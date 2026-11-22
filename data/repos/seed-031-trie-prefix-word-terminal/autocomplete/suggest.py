def suggest(trie, prefix, limit=5):
    """Up to `limit` completions for prefix, shortest first then alphabetical."""
    words = trie.words_with_prefix(prefix.lower())
    return sorted(words, key=lambda w: (len(w), w))[:limit]
