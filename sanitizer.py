import lief

lief.logging.disable()

def get_sanitized_bytes(file_path):
    """
    Parses PE headers with LIEF to detect and strip appended overlay bytes.
    """
    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        binary = lief.parse(file_path)
        if not binary:
            return raw_bytes

        sections = list(binary.sections)
        if not sections:
            return raw_bytes

        # Calculate true PE end byte using section boundaries
        pe_end = max(s.offset + s.size for s in sections)
        if 0 < pe_end < len(raw_bytes):
            return raw_bytes[:pe_end]
        return raw_bytes
    except Exception:
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except Exception:
            return None