"""``configs/rag/{smoke,full}.yaml``: one validated config for every Stage 5 command.

The smoke config builds the whole pipeline offline from the synthetic fixtures in
``tests/fixtures/rag/`` with the real MedCPT models on the CPU. The full config builds the
real corpus: PubMed through E-utilities, all of StatPearls, and the four guidelines.

Where things live (``RagPaths``): downloads are shared under ``<FATE_RAG_DIR>/raw/``; each
config builds under ``<FATE_RAG_DIR>/<run_name>/``, so a smoke run can never replace the
full index. ``FATE_RAG_DIR`` comes from ``.env``, never from the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fate_icu.rag.bm25 import BM25Params
from fate_icu.rag.encoders import MEDCPT_ARTICLE, MEDCPT_CROSS, MEDCPT_QUERY, ModelRef
from fate_icu.rag.sources.guidelines import GuidelineSpec
from fate_icu.rag.sources.statpearls import DEFAULT_EXCLUDED_SECTIONS, DEFAULT_LICENCE, STATPEARLS_URL
from fate_icu.utils.env import REPO_ROOT, Settings, under_onedrive

PUBMED_LICENCE = (
    "PubMed abstract: copyright of the publisher or authors; retrieved through NCBI E-utilities "
    "for research retrieval and short quotation with citation"
)
UNCONFIRMED = "TO CONFIRM"  # a guideline licence starting with this stops the full build


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PubmedSource(_Strict):
    enabled: bool = True
    queries: str = "configs/rag/pubmed_queries.yaml"
    fixtures: str | None = None  # a folder of PubMed XML files parsed instead of fetching (offline)
    max_records: int | None = None
    batch_size: int = 200
    max_workers: int = 1
    licence: str = PUBMED_LICENCE


class StatpearlsSource(_Strict):
    enabled: bool = True
    url: str = STATPEARLS_URL
    expected_size: int | None = None
    fixtures: str | None = None  # a folder of .nxml files read instead of the archive (offline)
    max_articles: int | None = None
    exclude_sections: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDED_SECTIONS))
    licence: str = DEFAULT_LICENCE


class GuidelineSources(_Strict):
    enabled: bool = True
    dir: str | None = None  # default <FATE_RAG_DIR>/raw/guidelines
    items: list[GuidelineSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> GuidelineSources:
        ids = [g.id for g in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("guideline ids must be unique")
        return self


class Sources(_Strict):
    pubmed: PubmedSource = PubmedSource()
    statpearls: StatpearlsSource = StatpearlsSource()
    guidelines: GuidelineSources = GuidelineSources()


class Chunking(_Strict):
    max_tokens: int = 300
    min_tokens: int = 5
    counter: Literal["medcpt", "words"] = "medcpt"  # medcpt = the article encoder's tokenizer


class Models(_Strict):
    query_encoder: ModelRef = MEDCPT_QUERY
    article_encoder: ModelRef = MEDCPT_ARTICLE
    cross_encoder: ModelRef = MEDCPT_CROSS


class Embedding(_Strict):
    batch_size: int = 64
    max_length: int = 512
    shard_size: int = 50_000
    autocast_fp16: bool = False  # CUDA only; recorded in the index version


class Retrieval(_Strict):
    depth: int = 50  # BM25 and dense lists fused by RRF
    rrf_k: int = 60
    rerank_depth: int = 50  # fused candidates scored by the cross-encoder
    top_k: int = 3  # passages per driver
    store_depth: int = 10  # passages stored per system and query in the query bank
    query_max_length: int = 64
    cross_max_length: int = 512
    cross_batch_size: int = 32
    cpu_check_queries: int = 2  # queries whose GPU rerank is recomputed on the laptop

    @model_validator(mode="after")
    def _depths(self) -> Retrieval:
        if not 1 <= self.top_k <= self.store_depth <= self.depth:
            raise ValueError("need 1 <= top_k <= store_depth <= depth")
        if self.rerank_depth < self.store_depth:
            raise ValueError("rerank_depth must be at least store_depth")
        return self


class Evaluation(_Strict):
    direction: Literal["longer", "shorter"] = "longer"
    pool_depth: int = 5
    n_boot: int = 1000
    n_perm: int = 10_000
    seed: int = 666
    labels: str = "configs/rag/relevance_labels.tsv"


class RagConfig(_Strict):
    """A whole ``configs/rag/*.yaml``."""

    run_name: str
    require_outside_onedrive: bool = True
    sources: Sources = Sources()
    chunking: Chunking = Chunking()
    bm25: BM25Params = BM25Params()
    models: Models = Models()
    embedding: Embedding = Embedding()
    retrieval: Retrieval = Retrieval()
    queries: str = "configs/rag/queries.yaml"
    evaluation: Evaluation = Evaluation()
    smoke_packets: str | None = None  # a folder of example packets retrieved after the bank is built
    seed: int = 666

    @model_validator(mode="after")
    def _pool_is_stored(self) -> RagConfig:
        if self.evaluation.pool_depth > self.retrieval.store_depth:
            raise ValueError(
                "evaluation.pool_depth must not exceed retrieval.store_depth (the bank stores no more)"
            )
        return self


def repo_path(path: str | Path) -> Path:
    """A config path: absolute, or relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> RagConfig:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return RagConfig.model_validate(yaml.safe_load(handle))


@dataclass(frozen=True)
class RagPaths:
    """The folder layout under ``FATE_RAG_DIR`` for one config."""

    root: Path
    run_name: str

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def raw_pubmed(self) -> Path:
        return self.raw / "pubmed"

    @property
    def raw_statpearls(self) -> Path:
        return self.raw / "statpearls"

    @property
    def raw_guidelines(self) -> Path:
        return self.raw / "guidelines"

    @property
    def build(self) -> Path:
        return self.root / self.run_name

    def corpus(self, corpus_id: str) -> Path:
        return self.build / "corpus" / corpus_id

    def index(self, version: str) -> Path:
        return self.build / "index" / version

    @property
    def eval(self) -> Path:
        return self.build / "eval"

    @property
    def runs(self) -> Path:
        return self.build / "runs"

    def pointer(self, kind: Literal["corpus", "index"]) -> Path:
        return self.build / f"{kind}.CURRENT"

    def current(self, kind: Literal["corpus", "index"]) -> str:
        """The id of the latest built corpus or index (fails loudly when none exists)."""
        pointer = self.pointer(kind)
        if not pointer.exists():
            step = "build_corpus" if kind == "corpus" else "build_index --part bm25"
            raise FileNotFoundError(f"no {kind} built for {self.run_name} yet: run fate_icu.rag.{step} first")
        return pointer.read_text(encoding="utf-8").strip()

    def set_current(self, kind: Literal["corpus", "index"], value: str) -> None:
        self.build.mkdir(parents=True, exist_ok=True)
        tmp = self.pointer(kind).with_suffix(".tmp")
        tmp.write_text(value + "\n", encoding="utf-8")
        tmp.replace(self.pointer(kind))


def rag_paths(config: RagConfig, settings: Settings) -> RagPaths:
    """The layout for this config, after the OneDrive check (CLAUDE.md: fail loudly).

    Raises:
        RuntimeError: ``require_outside_onedrive`` and ``FATE_RAG_DIR`` is inside OneDrive.
    """
    if config.require_outside_onedrive and under_onedrive(settings.rag_dir):
        raise RuntimeError(
            "FATE_RAG_DIR points inside a OneDrive folder, which would try to sync several GB. "
            "Set FATE_RAG_DIR in .env to a folder outside OneDrive, e.g. C:\\fate-data\\rag."
        )
    return RagPaths(settings.rag_dir, config.run_name)
