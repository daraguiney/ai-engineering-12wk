import pymupdf
import pytest

from rag.corpus import extract_text, pdf_url, validate_arxiv_id


def _make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)


def test_extract_text_returns_pages_joined(tmp_path):
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, ["Hello page one", "Second page"])
    text = extract_text(pdf)
    assert "Hello page one" in text and "Second page" in text
    assert text.index("Hello") < text.index("Second")


def test_extract_text_empty_pdf_raises(tmp_path):
    pdf = tmp_path / "blank.pdf"
    _make_pdf(pdf, [""])
    with pytest.raises(ValueError, match="no text"):
        extract_text(pdf)


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "1706.03762; rm", "abc"])
def test_validate_arxiv_id_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        validate_arxiv_id(bad)


def test_pdf_url_for_valid_id():
    assert pdf_url("1706.03762") == "https://arxiv.org/pdf/1706.03762"
