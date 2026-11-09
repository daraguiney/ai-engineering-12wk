from itertools import pairwise

import pytest

from rag.chunking import chunk_text
from rag.testset import normalize

TEXT = " ".join(f"word{i}" for i in range(2000))


def test_chunks_respect_size_and_cover_all_text():
    chunks = chunk_text(TEXT, size=500, overlap=100)
    assert all(len(c.text) <= 500 for c in chunks)
    assert normalize(chunks[0].text).startswith("word0")
    assert normalize(chunks[-1].text).endswith("word1999")


def test_every_window_up_to_overlap_length_fits_in_some_chunk():
    chunks = chunk_text(TEXT, size=500, overlap=120)
    words = TEXT.split()
    for start in range(0, len(words) - 12, 7):
        window = " ".join(words[start : start + 12])  # ~100 chars, below the overlap
        assert any(window in c.text for c in chunks), window


def test_chunks_start_and_end_on_word_boundaries():
    for c in chunk_text(TEXT, size=500, overlap=100):
        assert not c.text[0].isspace() and not c.text[-1].isspace()
        assert c.text.split()[0].startswith("word") and c.text.split()[-1].startswith("word")


def test_positions_are_sequential_and_indexed():
    chunks = chunk_text(TEXT, size=500, overlap=100)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(a.start < b.start for a, b in pairwise(chunks))


def test_short_text_is_one_chunk():
    assert [c.text for c in chunk_text("just a few words", size=500, overlap=100)] == [
        "just a few words"
    ]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1), (100, 150)])
def test_bad_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", size=size, overlap=overlap)


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n ", size=500, overlap=100) == []
