import pytest

from miles.utils.misc import FunctionRegistry, function_registry, load_function


class TestFunctionRegistry:
    def test_register_and_get(self):
        registry = FunctionRegistry()
        fn = lambda x: x + 1
        registry.register("my_fn", fn)
        assert registry.get("my_fn") is fn

    def test_register_duplicate_raises(self):
        registry = FunctionRegistry()
        registry.register("my_fn", lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("my_fn", lambda: None)

    def test_unregister(self):
        registry = FunctionRegistry()
        registry.register("my_fn", lambda: None)
        registry.unregister("my_fn")
        assert registry.get("my_fn") is None

    def test_unregister_nonexistent_no_error(self):
        registry = FunctionRegistry()
        registry.unregister("nonexistent")

    def test_temporary_context_manager(self):
        registry = FunctionRegistry()
        fn = lambda: "temp"
        with registry.temporary("temp_fn", fn):
            assert registry.get("temp_fn") is fn
        assert registry.get("temp_fn") is None

    def test_temporary_cleanup_on_exception(self):
        registry = FunctionRegistry()
        with pytest.raises(RuntimeError):
            with registry.temporary("temp_fn", lambda: None):
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
        my_fn = lambda: "registered"
        with function_registry.temporary("test:my_fn", my_fn):
            loaded = load_function("test:my_fn")
            assert loaded is my_fn

    def test_registry_takes_precedence(self):
        my_fn = lambda: "override"
        with function_registry.temporary("os.path.join", my_fn):
            loaded = load_function("os.path.join")
            assert loaded is my_fn
