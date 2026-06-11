from typing import Callable, Dict
from functools import wraps

def fwd_context_hook() -> Dict[str, any]:
    return {}

def modular_ctx(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = fwd_context_hook()
        return fn(*args, **{**kwargs, **ctx})  # kwargs as defaults, ctx can override
    return wrapper
