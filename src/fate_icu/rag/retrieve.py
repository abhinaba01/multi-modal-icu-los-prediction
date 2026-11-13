"""The query bank and ``retrieve(packet) -> passages`` (Stage 5).

``fate_icu.rag.query_bank`` runs every fixed query through BM25, dense retrieval, their
fusion and the reranked fusion, and stores the rankings with the passages they point to
in ``query_bank.json``. Retrieval for an evidence packet is then a lookup:

* each driver is read only by ``(modality, key, direction)``; its free-text fields
  (``label``, ``observed``, ``time``, ``effect``...) are never read, because they can hold
  patient values;
* drivers are taken in packet order; each gets the first ``top_k`` passages of its query;
* a passage gets one id, R1, R2, ... in order of first appearance, and lists every driver
  that retrieved it.

The result therefore depends only on the bank and on the drivers' keys and order: it is
identical on every machine and needs no model at explanation time.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fate_icu.rag.corpus import Passage

SYSTEMS = ("bm25", "dense", "hybrid", "hybrid_rerank")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RankedHit(_Strict):
    passage_id: str
    score: float


class BankQuery(_Strict):
    query_id: str
    modality: Literal["timeseries", "static", "image"]
    key: str
    direction: Literal["longer", "shorter"]
    text: str
    rankings: dict[str, list[RankedHit]]


class QueryBank(_Strict):
    """Every query's rankings by system, and the passages within the stored depth."""

    schema_version: int = 1
    index_version: str
    corpus_sha256: str
    query_spec_sha256: str
    template: str
    params: dict[str, Any]
    queries: dict[str, BankQuery]
    passages: dict[str, Passage]

    @model_validator(mode="after")
    def _complete(self) -> QueryBank:
        depth = int(self.params["store_depth"])
        for query_id, query in self.queries.items():
            if query.query_id != query_id:
                raise ValueError(f"query {query_id} is stored under another id")
            if set(query.rankings) != set(SYSTEMS):
                raise ValueError(
                    f"query {query_id} has rankings for {sorted(query.rankings)}, not {list(SYSTEMS)}"
                )
            for system, hits in query.rankings.items():
                missing = [h.passage_id for h in hits[:depth] if h.passage_id not in self.passages]
                if missing:
                    raise ValueError(
                        f"query {query_id}, {system}: passage {missing[0]} is not stored in the bank"
                    )
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def save_query_bank(bank: QueryBank, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(bank.canonical_json() + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_query_bank(path: str | Path) -> QueryBank:
    return QueryBank.model_validate_json(Path(path).read_text(encoding="utf-8"))


class DriverRef(BaseModel):
    """What retrieval reads of a packet's driver; every other field is ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(pattern=r"^D\d+$")
    modality: Literal["timeseries", "static", "image"]
    key: str
    direction: Literal["longer", "shorter"]


class RetrievedPassage(_Strict):
    rid: str
    passage_id: str
    source: str
    title: str
    section: str
    locator: str
    text: str
    url: str
    citation: str
    licence: str
    year: int | None
    drivers: tuple[str, ...]


class RetrievalResult(_Strict):
    index_version: str
    bank_sha256: str
    system: str
    top_k: int
    queries: dict[str, str]  # driver id -> query text
    by_driver: dict[str, tuple[str, ...]]  # driver id -> passage ids R1..Rn in rank order
    passages: tuple[RetrievedPassage, ...]


def _driver_refs(packet: Any) -> list[DriverRef]:
    drivers = packet.get("drivers") if isinstance(packet, Mapping) else getattr(packet, "drivers", None)
    if drivers is None:
        raise ValueError("the packet has no drivers")
    refs = []
    for driver in drivers:
        if isinstance(driver, Mapping):
            data = dict(driver)
        elif isinstance(driver, BaseModel):
            data = driver.model_dump()
        else:
            data = {name: getattr(driver, name, None) for name in DriverRef.model_fields}
        refs.append(DriverRef.model_validate(data))
    ids = [r.id for r in refs]
    if len(ids) != len(set(ids)):
        raise ValueError(
            f"duplicate driver ids in the packet: {sorted({i for i in ids if ids.count(i) > 1})}"
        )
    return refs


class Retriever:
    """``retrieve(packet)`` over a query bank (see the module docstring)."""

    def __init__(self, bank: QueryBank, *, system: str = "hybrid_rerank", top_k: int = 3):
        if system not in SYSTEMS:
            raise ValueError(f"unknown system {system!r}; choose one of {SYSTEMS}")
        if not 1 <= top_k <= int(bank.params["store_depth"]):
            raise ValueError(f"top_k must be between 1 and the stored depth {bank.params['store_depth']}")
        self.bank, self.system, self.top_k = bank, system, top_k
        self.bank_sha256 = bank.sha256

    @classmethod
    def from_file(cls, path: str | Path, **kw: Any) -> Retriever:
        return cls(load_query_bank(path), **kw)

    def retrieve(self, packet: Any) -> RetrievalResult:
        order: list[str] = []  # passage ids by first appearance
        drivers_of: dict[str, list[str]] = {}
        queries, by_driver = {}, {}
        for ref in _driver_refs(packet):
            query_id = f"{ref.modality}:{ref.key}:{ref.direction}"
            query = self.bank.queries.get(query_id)
            if query is None:
                raise KeyError(f"driver {ref.id}: no query {query_id} (unknown key or direction)")
            hits = query.rankings[self.system][: self.top_k]
            if not hits:
                raise ValueError(f"driver {ref.id}: query {query_id} has no {self.system} results")
            queries[ref.id] = query.text
            rids = []
            for hit in hits:
                if hit.passage_id not in drivers_of:
                    order.append(hit.passage_id)
                    drivers_of[hit.passage_id] = []
                if ref.id not in drivers_of[hit.passage_id]:
                    drivers_of[hit.passage_id].append(ref.id)
                rids.append(f"R{order.index(hit.passage_id) + 1}")
            by_driver[ref.id] = tuple(rids)
        passages = []
        for number, passage_id in enumerate(order, start=1):
            p = self.bank.passages[passage_id]
            passages.append(
                RetrievedPassage(
                    rid=f"R{number}",
                    passage_id=passage_id,
                    source=p.source,
                    title=p.title,
                    section=p.section,
                    locator=p.locator,
                    text=p.text,
                    url=p.url,
                    citation=p.citation,
                    licence=p.licence,
                    year=p.year,
                    drivers=tuple(drivers_of[passage_id]),
                )
            )
        return RetrievalResult(
            index_version=self.bank.index_version,
            bank_sha256=self.bank_sha256,
            system=self.system,
            top_k=self.top_k,
            queries=queries,
            by_driver=by_driver,
            passages=tuple(passages),
        )


def retrieve(packet: Any, retriever: Retriever) -> RetrievalResult:
    return retriever.retrieve(packet)


def passages_for_prompt(result: RetrievalResult) -> list[dict[str, str]]:
    """The fields of each retrieved passage an explanation prompt shows (Stage 6)."""
    return [
        {
            "rid": p.rid,
            "citation": p.citation,
            "title": p.title,
            "section": p.section,
            "locator": p.locator,
            "text": p.text,
        }
        for p in result.passages
    ]
