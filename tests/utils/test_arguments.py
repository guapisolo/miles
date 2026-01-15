import sys
from unittest.mock import patch

import pytest

from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.misc import function_registry


class TestAddArgumentsSupport:

    def test_class_add_arguments_is_called_and_arg_is_parsed(self):
        class MyRolloutFn:
            @classmethod
            def add_arguments(cls, parser):
                parser.add_argument("--my-custom-arg", type=int, default=42)

        with function_registry.temporary("test:rollout_class", MyRolloutFn):
            with patch.object(sys, "argv", [
                "test",
                "--rollout-function-path", "test:rollout_class",
                "--my-custom-arg", "100",
            ]):
                import argparse
                parser = argparse.ArgumentParser()
                add_miles_arguments = get_miles_extra_args_provider()
                add_miles_arguments(parser)

                args, _ = parser.parse_known_args()
                assert args.my_custom_arg == 100

    def test_function_add_arguments_is_called_and_arg_is_parsed(self):
        def my_generate_fn():
            pass

        def add_arguments(parser):
            parser.add_argument("--my-gen-arg", type=str, default="default")

        my_generate_fn.add_arguments = add_arguments

        with function_registry.temporary("test:generate_fn", my_generate_fn):
            with patch.object(sys, "argv", [
                "test",
                "--custom-generate-function-path", "test:generate_fn",
                "--my-gen-arg", "custom_value",
            ]):
                import argparse
                parser = argparse.ArgumentParser()
                add_miles_arguments = get_miles_extra_args_provider()
                add_miles_arguments(parser)

                args, _ = parser.parse_known_args()
                assert args.my_gen_arg == "custom_value"

    def test_skips_function_without_add_arguments(self):
        def my_rollout_fn():
            pass

        with function_registry.temporary("test:rollout_fn", my_rollout_fn):
            with patch.object(sys, "argv", ["test", "--rollout-function-path", "test:rollout_fn"]):
                import argparse
                parser = argparse.ArgumentParser()
                add_miles_arguments = get_miles_extra_args_provider()
                add_miles_arguments(parser)
