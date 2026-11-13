"""JATS/BITS XML to titled sections of paragraphs (StatPearls articles, PMC guideline XML).

Rules:

* the main body is the first ``body`` element; paragraphs outside every ``sec`` form a
  section titled "";
* each ``sec`` is a section titled by its ``title``; a nested ``sec`` is its own section
  titled "Parent: Child", so a passage never mixes a parent's text with a child's;
* a section whose own title is excluded (case-insensitive) is skipped with its subsections;
* paragraph text comes from every ``p`` (also inside lists, boxed text, definition lists
  and quotes), each emitted once in document order: a ``p`` holding a list gives its own
  text first and then each list item;
* citation markers (``xref``), footnotes, figures, tables, reference lists, display
  formulas, labels and object ids are dropped; leftover empty brackets and spaces before
  punctuation are removed; text is normalised.

Tags and attributes are matched by local name (namespaces are stripped), and nothing
depends on one fixed layout: the real StatPearls files may differ from the test fixtures.
ElementTree (expat) resolves no external entities.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Collection, Iterator

from fate_icu.rag.corpus import Section
from fate_icu.rag.text import normalise_text

DROPPED = frozenset(
    {
        "xref",
        "fn",
        "fn-group",
        "fig",
        "fig-group",
        "table-wrap",
        "table-wrap-group",
        "table",
        "ref-list",
        "disp-formula",
        "label",
        "object-id",
        "graphic",
        "media",
        "caption",
    }
)
_EMPTY_BRACKETS = re.compile(r"\[[\s,;\-–]*\]|\([\s,;\-–]*\)")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?)\]])")
_CC = re.compile(r"creativecommons\.org/licenses/([a-z-]+)/(\d+(?:\.\d+)?)", re.IGNORECASE)


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def strip_namespaces(root: ET.Element) -> ET.Element:
    """Drop namespace URIs from every tag and attribute name, in place."""
    for element in root.iter():
        element.tag = local(element.tag)
        if any("}" in name for name in element.attrib):
            element.attrib = {local(k): v for k, v in element.attrib.items()}
    return root


def parse_xml(data: bytes) -> ET.Element:
    return strip_namespaces(ET.fromstring(data))


def clean(text: str) -> str:
    text = _EMPTY_BRACKETS.sub("", normalise_text(text))
    return normalise_text(_SPACE_BEFORE_PUNCTUATION.sub(r"\1", text))


def _own_text(element: ET.Element) -> str:
    """Text of an element without its dropped parts, nested paragraphs or sections."""
    parts = [element.text or ""]
    for child in element:
        tag = local(child.tag)
        if tag not in DROPPED and tag not in ("p", "sec"):
            parts.append(_own_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _has_paragraph(element: ET.Element) -> bool:
    return any(local(e.tag) == "p" for e in element.iter() if e is not element)


def _paragraphs(element: ET.Element) -> Iterator[str]:
    """Paragraphs under ``element`` in document order, not entering sections."""
    for child in element:
        tag = local(child.tag)
        if tag in DROPPED or tag in ("sec", "title"):
            continue
        if tag == "p" or (tag == "list-item" and not _has_paragraph(child)):
            text = clean(_own_text(child))
            if text:
                yield text
        yield from _paragraphs(child)


def _child_sections(element: ET.Element) -> Iterator[ET.Element]:
    """The nearest ``sec`` descendants (not inside another ``sec`` or a dropped element)."""
    for child in element:
        tag = local(child.tag)
        if tag == "sec":
            yield child
        elif tag not in DROPPED:
            yield from _child_sections(child)


def _sections(sec: ET.Element, prefix: str, excluded: set[str]) -> list[Section]:
    title_element = next((c for c in sec if local(c.tag) == "title"), None)
    title = clean(_own_text(title_element)) if title_element is not None else ""
    if title.lower() in excluded:
        return []
    full = f"{prefix}: {title}" if prefix and title else (title or prefix)
    out = []
    paragraphs = tuple(_paragraphs(sec))
    if paragraphs:
        out.append(Section(full, paragraphs))
    for child in _child_sections(sec):
        out.extend(_sections(child, full, excluded))
    return out


def parse_jats_sections(root: ET.Element, *, exclude_titles: Collection[str] = ()) -> list[Section]:
    """The body of a JATS/BITS document as sections (see the module rules)."""
    body = next((e for e in root.iter() if local(e.tag) == "body"), None)
    if body is None:
        return []
    excluded = {t.strip().lower() for t in exclude_titles}
    sections = []
    lead = tuple(_paragraphs(body))
    if lead:
        sections.append(Section("", lead))
    for sec in _child_sections(body):
        sections.extend(_sections(sec, "", excluded))
    return sections


def _first(root: ET.Element, *paths: str) -> ET.Element | None:
    for path in paths:
        found = root.find(path)
        if found is not None:
            return found
    return None


def _meta(root: ET.Element) -> ET.Element:
    """The article's own metadata (BITS book-part-meta, JATS article-meta), else the root."""
    meta = _first(root, ".//book-part-meta", ".//article-meta")
    return root if meta is None else meta  # (an Element with no children is falsy)


def jats_title(root: ET.Element) -> str:
    title = _first(
        root,
        ".//book-part-meta/title-group/title",
        ".//article-meta/title-group/article-title",
        ".//title-group/title",
        ".//title-group/article-title",
    )
    return clean(_own_text(title)) if title is not None else ""


def _years(elements: list[ET.Element]) -> list[int]:
    out = []
    for element in elements:
        text = (element.findtext("year") or "").strip()
        if text.isdigit():
            out.append(int(text))
    return out


def jats_year(root: ET.Element) -> int | None:
    """The latest pub-history date ("last updated"), else the latest pub-date, else copyright-year."""
    meta = _meta(root)
    history = _years(meta.findall(".//pub-history//date"))
    if history:
        return max(history)
    published = _years(meta.findall(".//pub-date"))
    if published:
        return max(published)
    copyright_year = (meta.findtext(".//copyright-year") or "").strip()
    return int(copyright_year) if copyright_year.isdigit() else None


def jats_license(root: ET.Element) -> str | None:
    """A short licence name, e.g. "CC BY-NC-ND 4.0", or None when no licence is stated."""
    licence = _meta(root).find(".//permissions/license")
    if licence is None:
        licence = root.find(".//permissions/license")
    if licence is None:
        return None
    href = licence.get("href", "")
    match = _CC.search(href)
    if match:
        return f"CC {match.group(1).upper()} {match.group(2)}"
    text = clean(_own_text(licence))
    return text or licence.get("license-type") or None


def jats_authors(root: ET.Element) -> list[str]:
    """Authors as "Surname GI" (initials of the given names), in order."""
    meta = _meta(root)
    contribs = [
        c for c in meta.findall(".//contrib-group/contrib") if c.get("contrib-type", "author") == "author"
    ]
    names = []
    for contrib in contribs:
        name = contrib.find("name")
        if name is not None:
            surname = clean(name.findtext("surname") or "")
            given = clean(name.findtext("given-names") or "")
            initials = "".join(part[0].upper() for part in re.split(r"[\s.-]+", given) if part)
            text = " ".join(p for p in (surname, initials) if p)
        else:
            text = (
                clean(_own_text(contrib.find("string-name")))
                if contrib.find("string-name") is not None
                else ""
            )
        if text:
            names.append(text)
    return names
