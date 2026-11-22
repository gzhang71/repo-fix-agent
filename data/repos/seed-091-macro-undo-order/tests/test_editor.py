from editor.document import Document
from editor.history import History
from editor.ops import delete_range, replace_all, type_text, wrap_selection


def test_type_undo_redo():
    h = History(Document())
    type_text(h, 0, "hello")
    type_text(h, 5, " world")
    h.undo()
    assert h.doc.text == "hello"
    h.redo()
    assert h.doc.text == "hello world"


def test_delete_undo():
    h = History(Document("abcdef"))
    delete_range(h, 1, 3)
    assert h.doc.text == "adef"
    h.undo()
    assert h.doc.text == "abcdef"


def test_replace_all_and_wrap():
    h = History(Document("cat hat cat"))
    assert replace_all(h, "cat", "dog") == 2
    assert h.doc.text == "dog hat dog"
    wrap_selection(h, 4, 7, "**", "**")
    assert h.doc.text == "dog **hat** dog"


def test_new_edit_clears_redo():
    h = History(Document())
    type_text(h, 0, "a")
    h.undo()
    type_text(h, 0, "b")
    assert h.redo() is False
