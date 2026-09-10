import os
import re
import numpy as np
import lief
from sanitizer import get_sanitized_bytes

lief.logging.disable()

def safe_int(val):
    """Safely converts LIEF enum or integer types to native int."""
    if val is None:
        return 0
    if hasattr(val, 'value'):
        return int(val.value)
    try:
        return int(val)
    except Exception:
        return 0

def extract_byte_histogram(file_bytes):
    """Computes normalized 256-bin byte frequency distribution."""
    counts = np.bincount(np.frombuffer(file_bytes, dtype=np.uint8), minlength=256)
    total = len(file_bytes) if len(file_bytes) > 0 else 1
    return counts / total

def extract_byte_entropy_histogram(file_bytes, window_size=2048, step=1024):
    """Computes 256-bin joint byte-entropy distribution matrix."""
    if len(file_bytes) == 0:
        return np.zeros(256, dtype=np.float32)

    data = np.frombuffer(file_bytes, dtype=np.uint8)
    hist = np.zeros((16, 16), dtype=np.float32)

    for i in range(0, max(1, len(data) - window_size + 1), step):
        window = data[i:i + window_size]
        if len(window) == 0:
            continue
        counts = np.bincount(window, minlength=256)
        probs = counts / len(window)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        entropy_bin = min(15, int(entropy / 0.5))
        byte_bin = min(15, int(np.mean(window) / 16))
        hist[entropy_bin, byte_bin] += 1.0

    total = np.sum(hist)
    if total > 0:
        hist /= total
    return hist.flatten()

def extract_structural_features(file_bytes, file_path):
    """Extracts PE structural header metrics safely across all LIEF versions."""
    features = [float(len(file_bytes))]

    try:
        binary = lief.parse(file_path)
    except Exception:
        binary = None

    if binary is None or not isinstance(binary, lief.PE.Binary):
        return features + [0.0] * 20

    # 1. Optional Header Features
    try:
        opt = binary.optional_header
        sizeof_image = float(opt.sizeof_image)
        entrypoint = float(opt.addressof_entrypoint)
        subsystem = float(safe_int(opt.subsystem))
        dll_char = float(safe_int(opt.dll_characteristics))
    except Exception:
        sizeof_image, entrypoint, subsystem, dll_char = 0.0, 0.0, 0.0, 0.0

    # 2. File Header Features
    try:
        machine = float(safe_int(binary.header.machine))
        characteristics = float(safe_int(binary.header.characteristics))
    except Exception:
        machine, characteristics = 0.0, 0.0

    features.extend([sizeof_image, entrypoint, machine, characteristics, subsystem, dll_char])

    # 3. Imports & Exports Data
    imports, exports = [], []
    try:
        imports = list(binary.imports)
    except Exception:
        pass
    try:
        exports = list(binary.exported_functions)
    except Exception:
        pass

    # Flags
    has_imp_flag = 1.0 if len(imports) > 0 else 0.0
    has_exp_flag = 1.0 if len(exports) > 0 else 0.0
    has_res = 1.0 if getattr(binary, 'has_resources', False) or len(list(getattr(binary, 'resources', []))) > 0 else 0.0
    has_sig = 1.0 if getattr(binary, 'has_signatures', False) or len(list(getattr(binary, 'signatures', []))) > 0 else 0.0
    has_dbg = 1.0 if getattr(binary, 'has_debug', False) or len(list(getattr(binary, 'debug', []))) > 0 else 0.0
    has_rel = 1.0 if getattr(binary, 'has_relocations', False) or len(list(getattr(binary, 'relocations', []))) > 0 else 0.0
    has_tls = 1.0 if getattr(binary, 'has_tls', False) or getattr(binary, 'tls', None) is not None else 0.0

    features.extend([has_imp_flag, has_exp_flag, has_res, has_sig, has_dbg, has_rel, has_tls])

    # Import / Export Counts
    num_imports = 0
    for imp in imports:
        try:
            num_imports += len(list(imp.entries))
        except Exception:
            pass

    features.extend([float(num_imports), float(len(imports)), float(len(exports))])

    # Section Entropy & Size Stats
    sections = []
    try:
        sections = list(binary.sections)
    except Exception:
        pass

    features.append(float(len(sections)))
    if len(sections) > 0:
        entropies = []
        for s in sections:
            try:
                entropies.append(float(s.entropy))
            except Exception:
                pass
        if entropies:
            features.extend([float(np.mean(entropies)), float(np.max(entropies)), float(np.min(entropies))])
        else:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])

    return features

def extract_string_features(file_bytes):
    """Extracts printable ASCII string metadata."""
    strings = re.findall(b'[\x20-\x7e]{5,}', file_bytes)
    if not strings:
        return [0.0, 0.0, 0.0, 0.0]

    lengths = [len(s) for s in strings]
    total_chars = sum(lengths)
    return [
        float(len(strings)),
        float(np.mean(lengths)),
        float(np.max(lengths)),
        float(total_chars / (len(file_bytes) if len(file_bytes) > 0 else 1))
    ]

def extract_features_single(file_path, sanitize=False):
    """Extracts a unified static feature vector (537 total dimensions)."""
    file_bytes = get_sanitized_bytes(file_path) if sanitize else open(file_path, "rb").read()
    if not file_bytes:
        return None

    byte_hist = extract_byte_histogram(file_bytes)            # 256
    entropy_hist = extract_byte_entropy_histogram(file_bytes) # 256
    struct_feats = extract_structural_features(file_bytes, file_path) # 21
    string_feats = extract_string_features(file_bytes)        # 4

    return np.hstack([byte_hist, entropy_hist, struct_feats, string_feats])

def extract_features_from_folder(folder_path, label, sanitize=False):
    """Recursively parses all nested binaries into feature matrices."""
    X, y = [], []
    if not os.path.exists(folder_path):
        print(f"[!] Path not found: {folder_path}")
        return np.array(X), np.array(y)

    all_files = [
        os.path.join(root, filename)
        for root, _, files in os.walk(folder_path)
        for filename in files
    ]

    print(f"[*] Processing {len(all_files)} files in {folder_path} (Sanitize={sanitize})...")
    first_error = False
    for path in all_files:
        try:
            vec = extract_features_single(path, sanitize=sanitize)
            if vec is not None:
                X.append(vec)
                y.append(label)
        except Exception as e:
            if not first_error:
                print(f"[!] First extraction error on {os.path.basename(path)}: {e}")
                first_error = True
            continue

    print(f"[+] Successfully loaded {len(X)} samples from {folder_path}.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def prune_brittle_features(X_features):
    """Strips volatile byte/entropy histograms (0:512) and string stats (-4:)."""
    if len(X_features) == 0:
        return X_features
    mask = np.ones(X_features.shape[1], dtype=bool)
    mask[0:512] = False
    mask[-4:] = False
    return X_features[:, mask]