import importlib.util

def test_import_app():
    # Verifica che il modulo sia importabile
    assert importlib.util.find_spec("app") is not None
