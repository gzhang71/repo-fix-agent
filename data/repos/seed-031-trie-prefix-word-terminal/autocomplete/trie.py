class _Node:
    __slots__ = ("children", "terminal")

    def __init__(self):
        self.children = {}
        self.terminal = False


class Trie:
    def __init__(self, words=()):
        self.root = _Node()
        self._count = 0
        for w in words:
            self.insert(w)

    def insert(self, word):
        node = self.root
        created = False
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _Node()
                created = True
            node = node.children[ch]
        if created:  # only count genuinely new words
            node.terminal = True
            self._count += 1

    def _walk(self, prefix):
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def __contains__(self, word):
        node = self._walk(word)
        return node is not None and node.terminal

    def __len__(self):
        return self._count

    def words_with_prefix(self, prefix):
        start = self._walk(prefix)
        out = []
        stack = [(start, prefix)] if start else []
        while stack:
            node, sofar = stack.pop()
            if node.terminal:
                out.append(sofar)
            for ch, child in node.children.items():
                stack.append((child, sofar + ch))
        return sorted(out)
