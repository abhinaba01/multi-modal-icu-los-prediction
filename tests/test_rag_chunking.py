"""Stage 5 chunking: sentence splitting, the balanced partition, and passages from documents."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from transformers import BertTokenizer

from fate_icu.rag.chunking import (
    HFTokenCounter,
    WordCounter,
    chunk_document,
    chunk_documents,
    chunk_section,
    partition_balanced,
    split_long_unit,
    split_sentences,
)
from fate_icu.rag.corpus import Document, Section

WORDS = WordCounter()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Lactate rose. Creatinine fell.", ["Lactate rose.", "Creatinine fell."]),
        ("Was it sepsis? Yes! It was.", ["Was it sepsis?", "Yes!", "It was."]),
        ("See e.g. Table 2 for doses.", ["See e.g. Table 2 for doses."]),
        ("As Smith et al. Showed, mortality fell.", ["As Smith et al. Showed, mortality fell."]),
        ("Seen by Dr. Jones. She agreed.", ["Seen by Dr. Jones.", "She agreed."]),
        ("J. Smith and K. Lee wrote it.", ["J. Smith and K. Lee wrote it."]),
        (
            "E. coli and S. aureus grew. Both were treated.",
            ["E. coli and S. aureus grew.", "Both were treated."],
        ),
        ("The dose was 2.5 mg. Then 5 mg.", ["The dose was 2.5 mg.", "Then 5 mg."]),
        ("Results are in Fig. 3. They show a fall.", ["Results are in Fig. 3.", "They show a fall."]),
        ('He said "stop." Then he left.', ['He said "stop."', "Then he left."]),
        ("It rose (see above). [12] Later it fell.", ["It rose (see above).", "[12] Later it fell."]),
        (
            "RESULTS: Lactate was high. CONCLUSIONS: It matters.",
            ["RESULTS: Lactate was high.", "CONCLUSIONS: It matters."],
        ),
        ("ends with lowercase. next one stays", ["ends with lowercase. next one stays"]),
        ("No trailing stop", ["No trailing stop"]),
        ("", []),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_sentences_rebuild_the_text_exactly():
    rng = np.random.default_rng(0)
    pieces = [
        "Lactate rose.",
        "e.g.",
        "Dr.",
        "J.",
        "2.5",
        "Fig.",
        "(see)",
        "It fell!",
        "Why?",
        '"Yes."',
        "ok",
        "RESULTS:",
    ]
    for _ in range(200):
        text = " ".join(rng.choice(pieces, size=int(rng.integers(1, 12))))
        assert " ".join(split_sentences(text)) == text


def _brute_force(lengths, cap):
    best = None
    n = len(lengths)
    for cuts in itertools.product([False, True], repeat=n - 1):
        groups, start = [], 0
        for i, cut in enumerate(cuts, start=1):
            if cut:
                groups.append(sum(lengths[start:i]))
                start = i
        groups.append(sum(lengths[start:]))
        if max(groups) > cap:
            continue
        key = (len(groups), max(groups))
        best = key if best is None or key < best else best
    return best


def test_partition_is_optimal_against_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(1, 10))
        lengths = [int(x) for x in rng.integers(1, 40, size=n)]
        cap = int(rng.integers(max(lengths), 120))
        groups = partition_balanced(lengths, cap)
        assert groups[0][0] == 0 and groups[-1][1] == n
        assert all(a < b for a, b in groups) and all(
            b == c for (_, b), (c, _) in zip(groups, groups[1:], strict=False)
        )
        sums = [sum(lengths[a:b]) for a, b in groups]
        assert (len(groups), max(sums)) == _brute_force(lengths, cap)


def test_partition_edges():
    assert partition_balanced([], 10) == []
    assert partition_balanced([10, 10, 10], 10) == [(0, 1), (1, 2), (2, 3)]
    assert partition_balanced([3, 3, 3, 3], 9) == [(0, 2), (2, 4)]  # even: 6 + 6, not 9 + 3
    with pytest.raises(ValueError, match="more than max_tokens"):
        partition_balanced([5, 11], 10)


def test_split_long_unit():
    sentence = " ".join(f"w{i}" for i in range(25))
    pieces = split_long_unit(sentence, WORDS, 10)
    assert len(pieces) == 3 and all(n <= 10 for n in WORDS.count(pieces))
    assert " ".join(pieces) == sentence
    long_word = "x" * 50

    class Chars:
        name = "chars"

        def count(self, texts):
            return [len(t) for t in texts]

    pieces = split_long_unit("ab " + long_word, Chars(), 8)
    assert all(len(p) <= 8 for p in pieces) and "".join(pieces[1:]) == long_word


def _section(*paragraphs, title="Results", locators=()):
    return Section(title, tuple(paragraphs), tuple(locators))


def test_chunk_section_limits_joins_and_locators():
    para = " ".join(f"Sentence number {i} is here." for i in range(30))  # 30 sentences x 6 tokens
    chunks = chunk_section(_section(para, "Short tail.", locators=("3", "4")), WORDS, 50)
    assert all(c.n_tokens <= 50 for c in chunks)
    assert "\n".join(c.text for c in chunks).replace("\n", " ") == para + " Short tail."
    assert chunks[0].locator == "p. 3" and chunks[-1].locator in ("p. 4", "pp. 3-4")
    assert max(c.n_tokens for c in chunks) - min(c.n_tokens for c in chunks[:-1]) <= 6  # balanced
    joined = chunk_section(_section("One. Two.", "Three."), WORDS, 50)
    assert [c.text for c in joined] == ["One. Two.\nThree."] and joined[0].locator == ""


def test_chunk_section_with_a_counter_that_is_not_additive():
    class Penalty:
        """Every space-joined text costs one extra token per join."""

        name = "penalty"

        def count(self, texts):
            return [len(t.split()) + t.count(" ") for t in texts]

    para = " ".join(f"a{i} b{i}." for i in range(20))
    chunks = chunk_section(_section(para), Penalty(), 10)
    assert all(n <= 10 for n in Penalty().count([c.text for c in chunks]))
    assert " ".join(c.text for c in chunks) == para


def _doc(doc_id="1", sections=None, **kw):
    return Document(
        source=kw.pop("source", "statpearls"),
        doc_id=doc_id,
        title=kw.pop("title", "Lactic Acidosis"),
        sections=tuple(sections or ()),
        url="https://example.org",
        licence="test",
        citation="Test citation.",
        year=2025,
    )


def test_chunk_document_numbering_ids_and_fields():
    doc = _doc(
        "NBK1",
        [
            _section("Lactate is produced in tissues. It rises in shock.", title="Introduction"),
            _section("", "   ", title="Empty"),
            _section(" ".join(f"Point {i} follows." for i in range(20)), title="Treatment / Management"),
        ],
    )
    passages = chunk_document(doc, WORDS, max_tokens=20, min_tokens=1)
    assert [p.chunk for p in passages] == list(range(len(passages)))
    assert passages[0].passage_id == "statpearls:NBK1:0000" and passages[0].section == "Introduction"
    assert {p.section for p in passages[1:]} == {"Treatment / Management"}
    assert all(
        p.title == "Lactic Acidosis" and p.year == 2025 and p.citation == "Test citation." for p in passages
    )
    assert passages == chunk_document(doc, WORDS, max_tokens=20)  # deterministic
    dropped = chunk_document(doc, WORDS, max_tokens=20, min_tokens=12)
    assert all(p.n_tokens >= 12 for p in dropped) and [p.chunk for p in dropped] == list(range(len(dropped)))


def test_too_many_chunks_fail_loudly():
    doc = _doc("2", [_section(f"Word{i}.", title=f"S{i}") for i in range(10_001)])
    with pytest.raises(ValueError, match="more than 10000"):
        chunk_document(doc, WORDS, max_tokens=5)


def test_chunk_documents_equals_one_at_a_time():
    rng = np.random.default_rng(2)
    docs = []
    for d in range(7):
        sections = []
        for s in range(int(rng.integers(0, 4))):
            paras = [" ".join(f"Item {d}{s}{k} rises." for k in range(int(rng.integers(1, 30))))]
            sections.append(_section(*paras, title=f"Section {s}", locators=(str(s + 1),)))
        docs.append(_doc(str(d), sections))
    expected = [p for doc in docs for p in chunk_document(doc, WORDS, max_tokens=25)]
    for batch_size in (1, 2, 256):
        assert chunk_documents(docs, WORDS, max_tokens=25, batch_size=batch_size) == expected


def test_hf_token_counter_with_an_offline_tokenizer():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "lactate", "rose", "sep", "##sis", "."]
    tokenizer = BertTokenizer(vocab={token: i for i, token in enumerate(vocab)})
    counter = HFTokenCounter(tokenizer, name="hf:tiny", batch_size=2)
    assert counter.count(["Lactate rose.", "sepsis", "lactate lactate lactate", ""]) == [3, 2, 3, 0]
    assert counter.name == "hf:tiny"
