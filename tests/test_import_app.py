import importlib.util

def test_app_module_exists():
    assert importlib.util.find_spec("app") is not None
