from mdlite import render
from mdlite.inline import render_inline


def test_single_link():
    assert render_inline("[docs](https://a.io)") == '<a href="https://a.io">docs</a>'


def test_bold_and_code():
    assert render_inline("**hi** `x<y`") == "<strong>hi</strong> <code>x&lt;y</code>"


def test_blocks():
    assert render("# Title\n\nline one\nline two") == "<h1>Title</h1>\n<p>line one line two</p>"
