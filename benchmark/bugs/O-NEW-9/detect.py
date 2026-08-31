"""
O-NEW-9: Token IDs silently truncated to 16-bit by modulo 2^16.

Bug: `input_ids = input_ids % (2**16)` wraps all token IDs into [0, 65535].
Fix: Remove the modulo operation.

Detection: Hook MemMapDataset.__getitem__, pass through data with IDs > 65536,
           check if they survive unchanged.
"""
import os, sys, torch
import numpy as np

OLMO_DIR = os.environ.get("OLMO_DIR", "")
if OLMO_DIR:
    sys.path.insert(0, OLMO_DIR)

def main():
    from olmo.data.memmap_dataset import MemMapDataset

    # Hook __getitem__ to intercept the modulo operation
    _orig_getitem = MemMapDataset.__getitem__

    # Monkey-patch _read_chunk_from_memmap to return known data
    _orig_read = MemMapDataset._read_chunk_from_memmap
    _test_data = np.array([100, 50000, 65535, 65536, 70000, 100000], dtype=np.uint32)
    _injected = [False]

    def _hooked_read(self, path, index):
        if not _injected[0]:
            _injected[0] = True
            return _test_data.copy()
        return _orig_read(self, path, index)

    MemMapDataset._read_chunk_from_memmap = _hooked_read

    # Also need to make __getitem__ not fail on missing files
    _getitem_results = []

    def _hooked_getitem(self, index):
        try:
            result = _orig_getitem(self, index)
            ids = result["input_ids"]
            if hasattr(ids, 'numpy'):
                ids = ids.numpy()
            _getitem_results.append(np.array(ids))
            return result
        except Exception as e:
            # If memmap file doesn't exist, create minimal one
            _getitem_results.append(None)
            raise

    MemMapDataset.__getitem__ = _hooked_getitem

    # Create a minimal memmap file
    import tempfile, struct
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp_path = f.name
        # Write enough uint32 data for one chunk
        for _ in range(10):  # 10 sequences worth
            _test_data.tofile(f)

    # Create the dataset
    try:
        dataset = MemMapDataset(tmp_path, chunk_size=len(_test_data), memmap_dtype=np.uint32)
        item = dataset[0]
        loaded = _getitem_results[-1] if _getitem_results else None
    except Exception as e:
        loaded = None
        print(f"  Dataset error: {e}")

    os.unlink(tmp_path)

    print(f"\n{'='*60}")
    print(f"[O-NEW-9] Token ID truncation check:")
    print(f"  Injected IDs: {list(_test_data)}")

    if loaded is not None:
        print(f"  Loaded IDs:   {list(loaded[:6])}")
        truncated = any(loaded[i] != _test_data[i] for i in range(min(len(loaded), len(_test_data))))
        if truncated:
            print(f"[O-NEW-9] BUG DETECTED: token IDs truncated")
            for i in range(min(len(loaded), len(_test_data))):
                if loaded[i] != _test_data[i]:
                    print(f"  ID {_test_data[i]} → {loaded[i]}")
        else:
            print(f"[O-NEW-9] CLEAN: token IDs preserved")
    else:
        # Direct test: simulate the buggy code path
        result = _test_data.copy()
        result_mod = result % (2**16)
        has_change = not np.array_equal(result, result_mod)

        # Check if MemMapDataset source has the modulo
        import importlib
        mod = importlib.import_module("olmo.data.memmap_dataset")
        src_file = mod.__file__
        with open(src_file) as f:
            src = f.read()
        has_modulo = "% (2**16)" in src or "% 65536" in src

        print(f"  Direct test: IDs > 65535 would be truncated: {has_change}")
        print(f"  Modulo in source: {has_modulo}")
        if has_modulo:
            print(f"[O-NEW-9] BUG DETECTED: modulo 2^16 truncates large token IDs")
            print(f"  70000 → {70000 % 65536}, 100000 → {100000 % 65536}")
        else:
            print(f"[O-NEW-9] CLEAN: no modulo truncation")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
