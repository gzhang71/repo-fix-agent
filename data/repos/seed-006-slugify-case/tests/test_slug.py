from blog.slug import slugify


def test_strips_punctuation():
    assert slugify("hello, world!") == "hello-world"


def test_collapses_separators():
    assert slugify("a  --  b") == "a-b"
