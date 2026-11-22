import os


def fingerprint(path):
    """Cheap change detector for a source file, stored in the state file as a string."""
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"
