import matplotlib

_orig_use = matplotlib.use
def _safe_use(backend, force=False, **kwargs):
    if backend.lower().startswith("tk"):
        return
    return _orig_use(backend, force=force, **kwargs)
    
matplotlib.use = _safe_use
_orig_use("Agg", force=True)