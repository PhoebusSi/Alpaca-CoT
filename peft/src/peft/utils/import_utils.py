import importlib

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None

def is_bnb_4bit_available():
    if not is_bnb_available():
        return False

    import bitsandbytes as bnb
    return hasattr(bnb.nn, "Linear4bit")