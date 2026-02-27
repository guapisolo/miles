import logging
import os
import re
from pathlib import Path
from typing import Any

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from miles.utils import megatron_bridge_utils

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]

_MERGE_PATCHED = False
_ORIG_MERGE = None
_PADDING_PATCHED = False
_ORIG_LOAD_DP_RESHARDABLE = None


def _is_optimizer_param_state_key(key: tuple[object, ...]) -> bool:
    return len(key) >= 2 and key[0] == "optimizer" and key[1] == "param_state"


def _merge_with_optimizer_param_state_list_compat(
    x1: dict | list, x2: dict | list, key: tuple[object, ...] = ()
) -> dict | list:
    """Merge dicts/lists recursively, tolerating list length mismatch for optimizer param_state."""
    if isinstance(x1, dict) and isinstance(x2, dict):
        for k, v2 in x2.items():
            if k not in x1:
                x1[k] = v2
            else:
                x1[k] = _merge_with_optimizer_param_state_list_compat(x1[k], v2, key=key + (k,))
        return x1
    if isinstance(x1, list) and isinstance(x2, list):
        if len(x1) != len(x2) and _is_optimizer_param_state_key(key):
            # Allow length mismatch only for optimizer param_state lists.
            if len(x1) < len(x2):
                x1.extend([None] * (len(x2) - len(x1)))
            for i, v2 in enumerate(x2):
                if i >= len(x1) or x1[i] is None:
                    x1[i] = v2
                else:
                    x1[i] = _merge_with_optimizer_param_state_list_compat(x1[i], v2, key=key + (i,))
            return x1
        if len(x1) != len(x2):
            raise ValueError(
                f"Cannot merge two lists with different lengths ({len(x1)} and {len(x2)}, "
                f"encountered at level {key})"
            )
        for i, v2 in enumerate(x2):
            x1[i] = _merge_with_optimizer_param_state_list_compat(x1[i], v2, key=key + (i,))
        return x1
    raise ValueError(f"Duplicate non-dict and non-list values encountered: `{x1}` and `{x2}` (at level {key})")


def _patch_dp_reshardable_merge_compat() -> None:
    global _MERGE_PATCHED, _ORIG_MERGE
    if _MERGE_PATCHED:
        return
    from megatron.core.dist_checkpointing import dict_utils

    _ORIG_MERGE = dict_utils.merge
    dict_utils.merge = _merge_with_optimizer_param_state_list_compat
    _MERGE_PATCHED = True
    logger.info("[ckpt compat] Patched dist_checkpointing.merge for dp_reshardable optimizer param_state.")


def _patch_dp_reshardable_padding_compat() -> None:
    global _PADDING_PATCHED, _ORIG_LOAD_DP_RESHARDABLE
    if _PADDING_PATCHED:
        return
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    _ORIG_LOAD_DP_RESHARDABLE = DistributedOptimizer.load_parameter_state_from_dp_reshardable

    def _load_parameter_state_from_dp_reshardable_compat(self, state_dict):
        if state_dict is not None and "per_bucket_numel_unpadded" in state_dict:
            per_bucket_numel_unpadded_in_checkpoint = state_dict["per_bucket_numel_unpadded"]
            assert self.per_bucket_numel_unpadded == per_bucket_numel_unpadded_in_checkpoint, (
                f"Number of unpadded elements in each bucket need to be the same in current run "
                f"({self.per_bucket_numel_unpadded}) and checkpoint "
                f"({per_bucket_numel_unpadded_in_checkpoint})"
            )

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    bucket_state = state_dict[gbuf_idx][dtype][bucket_idx]
                    bucket_state = [
                        bucket_state_elem
                        for bucket_state_elem in bucket_state
                        if not bucket_state_elem.get("padding", False)
                    ]

                    param_count = len(gbuf_range_map["param_map"])
                    if len(bucket_state) != param_count:
                        logger.warning(
                            "[ckpt compat] dp_reshardable bucket_state length mismatch "
                            f"(bucket_state={len(bucket_state)}, param_map={param_count}); "
                            "loading by order with truncation if needed."
                        )
                        if len(bucket_state) > param_count:
                            bucket_state = bucket_state[:param_count]
                    for src_tensors, (model_param, _param_range_map) in zip(
                        bucket_state, gbuf_range_map["param_map"].items(), strict=False
                    ):
                        self._set_main_param_and_optimizer_states(model_param, src_tensors)

    DistributedOptimizer.load_parameter_state_from_dp_reshardable = _load_parameter_state_from_dp_reshardable_compat
    _PADDING_PATCHED = True
    logger.info("[ckpt compat] Patched dp_reshardable optimizer state loader for missing padding key.")


def _resolve_checkpoint_iteration_dir(load_path: str | Path, args) -> Path:
    path = Path(load_path)
    if path.name.startswith("iter_") and path.is_dir():
        return path
    if args.ckpt_step is not None:
        return path / f"iter_{int(args.ckpt_step):07d}"
    latest_path = path / "latest_checkpointed_iteration.txt"
    if latest_path.is_file():
        latest_text = latest_path.read_text().strip()
        if latest_text.isdigit():
            return path / f"iter_{int(latest_text):07d}"
    return path


def _maybe_patch_dp_reshardable_merge_compat(load_path: str | Path, args) -> None:
    try:
        iter_dir = _resolve_checkpoint_iteration_dir(load_path, args)
        common_path = iter_dir / "common.pt"
        if not common_path.is_file():
            return
        import torch

        common_state: dict[str, Any] = torch.load(common_path, map_location="cpu", weights_only=False)
        sharding_type = (
            common_state.get("optimizer", {}).get("param_state_sharding_type", None)
            if isinstance(common_state, dict)
            else None
        )
        if sharding_type == "dp_reshardable":
            _patch_dp_reshardable_merge_compat()
            _patch_dp_reshardable_padding_compat()
    except Exception as exc:
        logger.warning(f"[ckpt compat] Failed to inspect checkpoint for dp_reshardable patch: {exc}")


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        _maybe_patch_dp_reshardable_merge_compat(load_path, args)
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import miles_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
