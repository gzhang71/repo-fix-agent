from .version import Version

UNRELEASED = "Unreleased"


class Changelog:
    def __init__(self):
        self.sections = [UNRELEASED]
        self.notes = {UNRELEASED: []}

    def add_release(self, text):
        """Promote the Unreleased notes to a new version section (newest first)."""
        v = Version(text)
        self.sections.insert(1, v)
        self.notes[v] = self.notes.pop(UNRELEASED)
        self.notes[UNRELEASED] = []
        return v

    def note(self, message, version=UNRELEASED):
        key = version if version == UNRELEASED else Version(version)
        self.notes[key].append(message)

    def position(self, version):
        """Index of a version's section; 0 is Unreleased."""
        return self.sections.index(Version(version))
