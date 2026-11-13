"""Stage 5 dense retrieval: encoder wrappers, resumable embeddings, exact FAISS search."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rag_tiny_models import save_tiny, tiny_cross_encoder, tiny_encoder, tiny_tokenizer

from fate_icu.rag.dense import EMBEDDINGS_FILE, DenseIndex, embed_corpus, load_embeddings
from fate_icu.rag.encoders import ArticleEncoder, CrossEncoder, ModelRef, QueryEncoder, load_pretrained

PAIRS = [
    ("Lactate clearance", "Early lactate clearance and length of stay in critically ill adults."),
    ("Kidney injury", "Serum creatinine rise within hours defines acute kidney injury stage."),
    ("Sedation", "Deep sedation prolongs mechanical ventilation."),
    ("Shock", "Norepinephrine dose in septic shock."),
    (
        "Effusion",
        "Pleural effusion on chest radiograph of ventilated patients was associated with longer stay.",
    ),
]


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_tokenizer()


@pytest.fixture(scope="module")
def article(tokenizer):
    return ArticleEncoder(tiny_encoder(), tokenizer, batch_size=2)


def test_cls_pooling_matches_a_manual_forward_pass(tokenizer):
    model = tiny_encoder()
    queries = ["lactate in septic shock", "creatinine"]
    with torch.inference_mode():
        encoded = tokenizer(queries, padding=True, truncation=True, max_length=64, return_tensors="pt")
        manual = model(**encoded).last_hidden_state[:, 0, :].numpy()
    got = QueryEncoder(model, tokenizer).encode(queries)
    assert got.dtype == np.float32 and got.shape == (2, 32)
    np.testing.assert_allclose(got, manual, atol=1e-5)


def test_article_pairs_order_and_batch_size_invariance(tokenizer, article):
    model = article.model
    with torch.inference_mode():
        encoded = tokenizer([list(PAIRS[1])], truncation=True, max_length=512, return_tensors="pt")
        single = model(**encoded).last_hidden_state[:, 0, :].numpy()[0]
    full = article.encode(PAIRS)
    np.testing.assert_allclose(full[1], single, atol=1e-5)  # pairs are tokenised as (title, text)
    for batch_size in (1, 3, 64):
        np.testing.assert_allclose(
            ArticleEncoder(model, tokenizer, batch_size=batch_size).encode(PAIRS), full, atol=1e-5
        )
    np.testing.assert_allclose(
        article.encode(PAIRS[::-1]), full[::-1], atol=1e-5
    )  # output follows input order
    assert article.encode([]).shape[0] == 0


def test_cross_encoder_logits(tokenizer):
    model = tiny_cross_encoder()
    texts = [text for _, text in PAIRS]
    with torch.inference_mode():
        encoded = tokenizer(
            [["lactate", t] for t in texts],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        manual = model(**encoded).logits[:, 0].numpy()
    got = CrossEncoder(model, tokenizer, batch_size=2).score("lactate", texts)
    np.testing.assert_allclose(got, manual, atol=1e-5)
    assert CrossEncoder(model, tokenizer).score("lactate", []).shape == (0,)


def test_model_refs_and_local_loading(tmp_path, tokenizer):
    with pytest.raises(ValueError):
        ModelRef(repo="ncbi/MedCPT-Query-Encoder", revision="main")
    folder = save_tiny(tmp_path / "tiny", tiny_encoder(), tokenizer)
    ref = ModelRef(repo=str(folder), revision="0" * 40)
    model, loaded_tokenizer = load_pretrained(ref, "encoder", local_files_only=True)
    encoder = QueryEncoder(model, loaded_tokenizer, ref=ref)
    np.testing.assert_allclose(
        encoder.encode(["lactate"]), QueryEncoder(tiny_encoder(), tokenizer).encode(["lactate"]), atol=1e-5
    )
    assert encoder.info["revision"] == "0" * 40 and encoder.info["precision"] == "fp32"
    with pytest.raises(ValueError, match="CUDA"):
        QueryEncoder(model, loaded_tokenizer, autocast_fp16=True)


class CrashingEncoder:
    """Encodes the first ``ok`` calls, then fails (a session that dies mid-run)."""

    def __init__(self, inner, ok):
        self.inner, self.ok, self.calls = inner, ok, 0

    def encode(self, pairs):
        self.calls += 1
        if self.calls > self.ok:
            raise RuntimeError("session died")
        return self.inner.encode(pairs)


def test_embeddings_resume_byte_identical(tmp_path, article):
    model = {"repo": "tiny", "revision": "0" * 40, "max_length": 512, "precision": "fp32"}
    pairs = PAIRS * 3  # 15 passages, shards of 4
    info = embed_corpus(pairs, article, tmp_path / "whole", corpus_sha256="c" * 64, model=model, shard_size=4)
    assert (info.n, info.dim) == (15, 32)
    crashing = CrashingEncoder(article, ok=2)
    with pytest.raises(RuntimeError, match="session died"):
        embed_corpus(pairs, crashing, tmp_path / "resumed", corpus_sha256="c" * 64, model=model, shard_size=4)
    resumed = CrashingEncoder(article, ok=10)
    again = embed_corpus(
        pairs, resumed, tmp_path / "resumed", corpus_sha256="c" * 64, model=model, shard_size=4
    )
    assert resumed.calls == 2  # shards 0 and 1 were reused
    assert again == info
    assert (tmp_path / "resumed" / EMBEDDINGS_FILE).read_bytes() == (
        tmp_path / "whole" / EMBEDDINGS_FILE
    ).read_bytes()
    loaded = load_embeddings(tmp_path / "whole", expect_corpus_sha256="c" * 64, expect_model=model)
    np.testing.assert_allclose(loaded, article.encode(pairs), atol=1e-5)
    other_model = {**model, "precision": "fp16-autocast"}
    redo = CrashingEncoder(article, ok=10)
    embed_corpus(pairs, redo, tmp_path / "resumed", corpus_sha256="c" * 64, model=other_model, shard_size=4)
    assert redo.calls == 4  # shards of another model are not reused


def test_load_embeddings_checks(tmp_path, article):
    model = {"repo": "tiny"}
    embed_corpus(PAIRS, article, tmp_path, corpus_sha256="a" * 64, model=model, shard_size=10)
    with pytest.raises(RuntimeError, match="corpus"):
        load_embeddings(tmp_path, expect_corpus_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="made with"):
        load_embeddings(tmp_path, expect_corpus_sha256="a" * 64, expect_model={"repo": "other"})
    assert load_embeddings(tmp_path, expect_corpus_sha256="a" * 64, mmap=True).shape == (5, 32)
    raw = bytearray((tmp_path / EMBEDDINGS_FILE).read_bytes())
    raw[-1] ^= 0xFF
    (tmp_path / EMBEDDINGS_FILE).write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_embeddings(tmp_path, expect_corpus_sha256="a" * 64)


def _brute(vectors, queries, k):
    out = []
    for q in queries:
        scores = vectors @ q
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        out.append([(int(i), float(scores[i])) for i in order])
    return out


def test_dense_index_matches_brute_force(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(300, 16)).astype(np.float32)
    queries = rng.normal(size=(7, 16)).astype(np.float32)
    index = DenseIndex.build(vectors)
    for got, expected in zip(index.search(queries, 25), _brute(vectors, queries, 25), strict=True):
        assert [i for i, _ in got] == [i for i, _ in expected]
        np.testing.assert_allclose([s for _, s in got], [s for _, s in expected], rtol=1e-5, atol=1e-5)
    index.save(tmp_path / "faiss.index")
    assert DenseIndex.load(tmp_path / "faiss.index").search(queries, 25) == index.search(queries, 25)


def test_dense_ties_and_small_indexes():
    identity = DenseIndex.build(np.eye(4, dtype=np.float32))
    assert identity.search(np.ones(4, dtype=np.float32), 3) == [
        [(0, 1.0), (1, 1.0), (2, 1.0)]
    ]  # FAISS gave 3, 2, 1, 0
    assert len(identity.search(np.ones(4), 10)[0]) == 4  # k is capped at the index size
    same = DenseIndex.build(np.ones((100, 4), dtype=np.float32))  # every score tied, across the cut
    assert [i for i, _ in same.search(np.ones(4), 5)[0]] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="dimension"):
        identity.search(np.ones(3), 2)
