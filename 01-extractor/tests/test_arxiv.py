import pytest

from extractor.arxiv import parse_atom

FEED = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<title>  A  Title
 Here </title><summary> Some
 abstract. </summary>
<author><name>Ada L</name></author><author><name>Bo K</name></author>
</entry></feed>"""


def test_parse_atom_normalises_whitespace():
    text = parse_atom(FEED, "1234.5678")
    assert "Title: A Title Here" in text
    assert "Authors: Ada L, Bo K" in text
    assert "Abstract: Some abstract." in text


def test_parse_atom_missing_entry_raises():
    with pytest.raises(ValueError, match="9999"):
        parse_atom(b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>', "9999")
