"""Compatibility entry point. Use extract_cfg_with_ida.py for VM extraction."""
import sys
import extract_cfg_with_ida as _extractor

if __name__ == '__main__':
    _extractor.main()
else:
    # Preserve the actual extractor path/hash for existing controllers.
    sys.modules[__name__] = _extractor
