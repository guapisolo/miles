import pytest

from miles.utils.misc import FunctionRegistry, function_registry, load_function


def _add_one(x):
    return x + 1


def _return_temp():
    return "temp"


def _return_registered():
    return "registered"


def _return_override():
    return "override"


class TestFunctionRegistry:
    def test_register_and_get(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _add_one):
            assert registry.get("my_fn") is _add_one

    def test_register_duplicate_raises(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _add_one):
            with pytest.raises(AssertionError):
                with registry.temporary("my_fn", _add_one):
                    pass

    def test_unregister(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _add_one):
            assert registry.get("my_fn") is _add_one
        assert registry.get("my_fn") is None

    def test_temporary_context_manager(self):
        registry = FunctionRegistry()
        with registry.temporary("temp_fn", _return_temp):
            assert registry.get("temp_fn") is _return_temp
        assert registry.get("temp_fn") is None

    def test_temporary_cleanup_on_exception(self):
        registry = FunctionRegistry()
        with pytest.raises(RuntimeError):
            with registry.temporary("temp_fn", _add_one):
                raise RuntimeError("test")
        assert registry.get("temp_fn") is None


class TestLoadFunction:
    def test_load_from_module(self):
        fn = load_function("os.path.join")
        import os.path
        assert fn is os.path.join

    def test_load_none_returns_none(self):
        assert load_function(None) is None

    def test_load_from_registry(self):
        with function_registry.temporary("test:my_fn", _return_registered):
            loaded = load_function("test:my_fn")
            assert loaded is _return_registered

    def test_registry_takes_precedence(self):
        with function_registry.temporary("os.path.join", _return_override):
            loaded = load_function("os.path.join")
            assert loaded is _return_override
