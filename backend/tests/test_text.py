import random

import pytest

from app.core.text import chunk_text, is_usable, normalize_whitespace, split_sentences

WHOLE, CHUNK, OVERLAP = 2000, 1200, 200


def make_passage(n_chars: int, seed: int = 0) -> str:
    """Realistic-ish abstract text: sentences of varied length, built to `n_chars`."""
    rng = random.Random(seed)
    words = "protein gene expression patients mutation receptor cells clinical study".split()
    sentences, total, i = [], 0, 0
    while total < n_chars:
        body = " ".join(rng.choice(words) for _ in range(rng.randint(8, 40)))
        sentence = f"Sentence {i} {body}."
        sentences.append(sentence)
        total += len(sentence) + 1
        i += 1
    return " ".join(sentences)[:n_chars].rsplit(" ", 1)[0] + "."


# --- cleaning -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "nan",
        " nan ",
        "NaN",
        "",
        "   ",
        "1.",
        None,
        "1. Am J Med. 1971 Jul;51(1):97-108. doi: 10.1016/0002-9343(71)90327-5.",
        "1. Cancer Discov. 2013 Mar;3(3):OF13. doi: 10.1158/2159-8290.CD-RW2013-028. "
        "Epub 2013 Feb 7.",
    ],
)
def test_unusable_passages_are_rejected(text):
    assert not is_usable(text, min_chars=50)


@pytest.mark.parametrize(
    "text",
    [
        "Pompe disease is caused by mutations in the GAA gene encoding acid alpha-glucosidase.",
        "1. Urinary incontinence is a costly and prevalent problem, affecting 15% to 39% of "
        "all community residing elderly people.",
    ],
)
def test_real_abstracts_are_kept(text):
    assert is_usable(text, min_chars=50)


def test_normalize_whitespace_collapses_line_breaks():
    assert normalize_whitespace("  The GAA\ngene  is\n\nmutated.\t ") == "The GAA gene is mutated."


# --- sentence splitting -------------------------------------------------------


def test_split_sentences_on_terminal_punctuation():
    text = "First sentence. Second one! Third? 4 patients were enrolled. (A) panel."
    assert split_sentences(text) == [
        "First sentence.",
        "Second one!",
        "Third?",
        "4 patients were enrolled.",
        "(A) panel.",
    ]


def test_split_sentences_keeps_lowercase_continuations():
    text = "Smith et al. reported this, e.g. in mice. Next sentence."
    assert split_sentences(text) == ["Smith et al. reported this, e.g. in mice.", "Next sentence."]


# --- chunking -----------------------------------------------------------------


def chunks_for(text: str) -> list[str]:
    return chunk_text(text, max_whole_chars=WHOLE, chunk_chars=CHUNK, overlap_chars=OVERLAP)


@pytest.mark.parametrize("n_chars", [60, 1200, 1999, 2000])
def test_short_passages_stay_whole(n_chars):
    text = make_passage(n_chars)
    assert len(text) <= WHOLE
    assert chunks_for(text) == [text]


@pytest.mark.parametrize("n_chars", [2050, 2600, 5000, 33_000])
def test_long_passages_are_split_within_size_limit(n_chars):
    text = make_passage(n_chars, seed=n_chars)
    assert len(text) > WHOLE
    chunks = chunks_for(text)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK for c in chunks)


@pytest.mark.parametrize("n_chars", [2600, 5000, 33_000])
def test_chunks_cover_every_sentence_in_order(n_chars):
    text = make_passage(n_chars, seed=n_chars)
    sentences = split_sentences(text)
    seen = []
    for chunk in chunks_for(text):
        for s in split_sentences(chunk):
            if not seen or s != seen[-1] and s not in seen[-5:]:
                seen.append(s)
    assert seen == sentences


def test_chunks_are_sentence_aligned():
    text = make_passage(5000, seed=1)
    sentences = set(split_sentences(text))
    for chunk in chunks_for(text):
        assert all(s in sentences for s in split_sentences(chunk))


def test_consecutive_chunks_overlap():
    text = make_passage(5000, seed=2)
    chunks = chunks_for(text)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        first_sentence = split_sentences(nxt)[0]
        assert first_sentence in split_sentences(prev)
        assert len(first_sentence) <= CHUNK // 2


def test_33k_passage_chunk_count_and_balance():
    text = make_passage(33_000, seed=3)
    chunks = chunks_for(text)
    # ~1,000 new chars per chunk -> about 33 chunks; allow slack for sentence alignment.
    assert 30 <= len(chunks) <= 45
    assert min(len(c) for c in chunks) >= 300


def test_passage_just_over_limit_splits_into_balanced_pair():
    text = make_passage(2100, seed=4)
    chunks = chunks_for(text)
    assert len(chunks) in (2, 3)
    assert max(len(c) for c in chunks) <= CHUNK


def test_single_huge_sentence_is_hard_split():
    text = " ".join(["word"] * 1000)  # 4,999 chars, no sentence boundary
    chunks = chunks_for(text)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK for c in chunks)
    assert " ".join(chunks).count("word") >= 1000
