def bucket(flag_name, user_id, buckets=100):
    """Deterministically map (flag, user) to a bucket in [0, buckets)."""
    return hash(f"{flag_name}:{user_id}") % buckets
