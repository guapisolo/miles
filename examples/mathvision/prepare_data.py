"""Prepare MathLLMs/MathVision for Miles RL.

MathVision ships only ``test`` (3040) and ``testmini`` (304) splits and has no
training split, so we use ``test`` as the RL prompt source and ``testmini`` as
the held-out eval set.

Each row is rewritten into the columns the Miles data loader expects:
  - ``problem``  : the question text with a single ``<image>`` placeholder plus
                   an answer-format instruction (``--input-key problem``).
  - ``answer``   : the gold answer, a letter for multiple-choice rows and a
                   value for free-form rows (``--label-key answer``).
  - ``images``   : ``[abs_path]`` to the decoded figure on disk; paired with
                   ``--multimodal-keys '{"image": "images"}'`` the loader swaps
                   the ``<image>`` placeholder for this image.
  - ``metadata`` : ``{id, options, is_choice, subject, level}`` consumed by the
                   MathVision reward in ``reward.py`` (``--metadata-key metadata``).

MathVision questions reference images with ``<imageN>`` tags but the dataset
provides exactly one decoded figure per row, so rows that reference more than
one image are dropped (they cannot be rendered faithfully with a single image).
"""

import argparse
import io
import os

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

SRC_DIR = "/root/datasets/MathVision/data"
OUT_DIR = "/root/datasets/mathvision_miles"

# test is the large set -> RL prompts; testmini is the small held-out eval set.
SPLIT_TO_OUT = {"test": "train", "testmini": "eval"}

IMAGE_TAG = "image"  # Miles placeholder is "<image>" (MultimodalTypes.IMAGE)

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{{}}. {answer_hint}"


def _src_parquet(split: str) -> str:
    import glob

    matches = glob.glob(os.path.join(SRC_DIR, f"{split}-*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet for split={split} under {SRC_DIR}. "
            "Run `hf download --repo-type dataset MathLLMs/MathVision "
            f"--local-dir {os.path.dirname(SRC_DIR)}` first."
        )
    return matches[0]


def _to_pil(decoded_image) -> Image.Image:
    # pyarrow returns the HF Image() feature as {"bytes": ..., "path": ...}.
    if isinstance(decoded_image, Image.Image):
        return decoded_image
    if isinstance(decoded_image, dict) and decoded_image.get("bytes"):
        return Image.open(io.BytesIO(decoded_image["bytes"]))
    raise ValueError(f"Unexpected decoded_image payload: {type(decoded_image)}")


def _image_tag_occurrences(question: str) -> list[str]:
    import re

    # Count occurrences (not unique tags): a question may repeat the same
    # ``<imageN>`` tag, and each occurrence becomes one ``<image>`` placeholder
    # that the loader pairs with one image. We keep only single-image rows.
    return re.findall(r"<image\d+>", question)


def _normalize_question(question: str, tags: list[str]) -> str:
    """Replace the single ``<imageN>`` occurrence with ``<image>``; prepend it if absent."""
    if not tags:
        return f"<{IMAGE_TAG}>\n{question.strip()}"
    return question.replace(tags[0], f"<{IMAGE_TAG}>").strip()


def _build_problem(question: str, options: list[str]) -> str:
    parts = [question]
    if options:
        # Options are sometimes bare letters (choices live in the figure) and
        # sometimes the choice content; only list them when they carry content.
        letters = [chr(ord("A") + i) for i in range(len(options))]
        is_label_only = all(str(o).strip().upper() == letters[i] for i, o in enumerate(options))
        if not is_label_only:
            rendered = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
            parts.append("Options:\n" + rendered)
        answer_hint = "Your final answer should be the option's letter (e.g. \\boxed{A})."
    else:
        answer_hint = "Your final answer should be the value only (e.g. \\boxed{42})."
    parts.append(INSTRUCTION.format(answer_hint=answer_hint))
    return "\n\n".join(parts)


def prepare_split(split: str, out_name: str, limit: int | None) -> str:
    rows = pq.ParquetFile(_src_parquet(split)).read().to_pylist()
    img_dir = os.path.join(OUT_DIR, "images", out_name)
    os.makedirs(img_dir, exist_ok=True)

    records = []
    skipped_multi_image = 0
    for row in rows:
        tags = _image_tag_occurrences(row["question"])
        if len(tags) > 1:
            skipped_multi_image += 1
            continue

        img_path = os.path.abspath(os.path.join(img_dir, f"{row['id']}.png"))
        _to_pil(row["decoded_image"]).convert("RGB").save(img_path)

        options = list(row["options"]) if row["options"] is not None else []
        records.append(
            {
                "problem": _build_problem(_normalize_question(row["question"], tags), options),
                "answer": str(row["answer"]),
                "images": [img_path],
                "metadata": {
                    "id": str(row["id"]),
                    "options": options,
                    "is_choice": bool(options),
                    "subject": row.get("subject") or "",
                    "level": int(row.get("level") or 0),
                },
            }
        )
        if limit is not None and len(records) >= limit:
            break

    out_path = os.path.join(OUT_DIR, f"{out_name}.parquet")
    pd.DataFrame(records).to_parquet(out_path)
    print(
        f"[{split} -> {out_name}] wrote {len(records)} rows to {out_path} "
        f"(skipped {skipped_multi_image} multi-image rows)"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap rows per split (smoke test)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for split, out_name in SPLIT_TO_OUT.items():
        prepare_split(split, out_name, args.limit)


if __name__ == "__main__":
    main()
