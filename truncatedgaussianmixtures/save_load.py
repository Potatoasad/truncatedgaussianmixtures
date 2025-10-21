import h5py
import numpy as np
import pandas as pd

_NONE_ATTR = "__is_none__"

def save_dict_h5(filename, data):
    def _save_group(h5group, d):
        for k, v in d.items():
            if v is None:
                g = h5group.create_group(k)
                g.attrs[_NONE_ATTR] = True
            elif isinstance(v, dict):
                _save_group(h5group.create_group(k), v)
            elif isinstance(v, pd.DataFrame):
                g = h5group.create_group(k)
                g.create_dataset("columns", data=np.array(v.columns, dtype="S"))
                g.create_dataset("values", data=v.to_numpy())
            elif isinstance(v, (list, tuple)):
                arr = np.array(v)
                if arr.dtype.kind in {"U", "O"}:  # strings (or objects that are strings)
                    arr = arr.astype("S")
                h5group.create_dataset(k, data=arr)
            elif isinstance(v, str):
                h5group.create_dataset(k, data=np.bytes_(v))  # NumPy 2.0+
            else:
                h5group.create_dataset(k, data=np.array(v))
    with h5py.File(filename, "w") as f:
        _save_group(f, data)

def load_dict_h5(filename):
    def _load_group(h5group):
        out = {}
        for k, v in h5group.items():
            if isinstance(v, h5py.Group):
                # None sentinel?
                if v.attrs.get(_NONE_ATTR, False):
                    out[k] = None
                # DataFrame?
                elif "columns" in v and "values" in v:
                    cols = [c.decode() for c in v["columns"][()]]
                    out[k] = pd.DataFrame(v["values"][()], columns=cols)
                else:
                    out[k] = _load_group(v)
            else:
                arr = v[()]
                if arr.dtype.kind == "S":
                    # Return string arrays as Python lists of str; scalars as str
                    out[k] = arr.decode() if arr.ndim == 0 else [x.decode() for x in arr.flatten()]
                else:
                    out[k] = arr.tolist() if arr.ndim == 0 else arr
        return out
    with h5py.File(filename, "r") as f:
        return _load_group(f)
