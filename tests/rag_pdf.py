"""A tiny dependency-free PDF writer for the guideline tests (Helvetica text only).

``write_fixture_guideline`` writes the synthetic guideline committed as
``tests/fixtures/rag/guidelines/fixture_guideline.pdf`` (the smoke config parses it):
running header, page numbers, one two-column page, numbered and keyword headings, a
hyphenated line break, a reference list with a wrapped entry, an appendix after the
references, and an acknowledgements section. All text is invented.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

Item = tuple[float, float, str, float]  # x, y, text, font size


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(
    path: str | Path, pages: Sequence[Sequence[Item]], page_labels: Sequence[str] | None = None
) -> Path:
    """A PDF 1.4 file with one content stream per page; text is drawn in the given order."""
    n = len(pages)
    objects: dict[int, bytes] = {}
    labels = ""
    if page_labels:
        nums = " ".join(f"{i} << /P ({_escape(label)}) >>" for i, label in enumerate(page_labels))
        labels = f" /PageLabels << /Nums [{nums}] >>"
    kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(n))
    objects[1] = f"<< /Type /Catalog /Pages 2 0 R{labels} >>".encode("latin-1")
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode("latin-1")
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    for i, items in enumerate(pages):
        stream = "".join(
            f"BT /F1 {size:g} Tf 1 0 0 1 {x:g} {y:g} Tm ({_escape(text)}) Tj ET\n"
            for x, y, text, size in items
        ).encode("latin-1")
        objects[4 + 2 * i] = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream"
        objects[5 + 2 * i] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {4 + 2 * i} 0 R >>"
        ).encode("latin-1")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number in range(1, len(objects) + 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path = Path(path)
    path.write_bytes(bytes(out))
    return path


HEADER = "Synth Crit Care Guidel 2026; 1: 101-104"


def fixture_guideline_pages() -> list[list[Item]]:
    def page(number: int, *items: Item) -> list[Item]:
        return [(72, 760, HEADER, 8), *items, (300, 40, str(number), 8)]

    return [
        page(
            101,
            (72, 730, "Synthetic Guideline for Monitoring in Adult Intensive Care", 14),
            (72, 700, "Abstract", 11),
            (
                72,
                684,
                "This synthetic guideline was written for software tests. It describes how lactate,",
                10,
            ),
            (72, 670, "creatinine and urine output are monitored in critically ill adults.", 10),
            (72, 640, "1. Lactate monitoring", 11),
            (
                72,
                624,
                "Measure lactate within the first hour of admission in adults with suspected shock. A",
                10,
            ),
            (
                72,
                610,
                "lactate above 2 mmol/L signals tissue hypoperfusion and should prompt reassessment of",
                10,
            ),
            (72, 596, "fluid and vasopressor therapy.", 10),
        ),
        page(  # two columns: the content stream holds the left column first, as journal PDFs do
            102,
            (72, 730, "2. Kidney function", 11),
            (72, 714, "Serum creatinine and urine output define acute kidney in-", 10),
            (72, 700, "jury stages. A rise of 0.3 mg/dL within 48 hours is stage 1.", 10),
            (320, 730, "2.1 Urine output", 11),
            (320, 714, "Urine output below 0.5 mL/kg/h for 6 hours also defines", 10),
            (320, 700, "stage 1. Oliguria is associated with a longer stay.", 10),
        ),
        page(
            103,
            (72, 730, "Recommendation 3", 11),
            (
                72,
                714,
                "We suggest light rather than deep sedation in ventilated adults, because deep sedation",
                10,
            ),
            (72, 700, "prolongs mechanical ventilation and the intensive care stay.", 10),
            (72, 670, "References", 11),
            (72, 654, "1. Rao A, Mensah K. A synthetic reference about lactate. Synth J. 2020;1:1-10.", 9),
            (72, 640, "2. Silva P, Novak T, Okafor B, Haddad S, Lindqvist E", 9),
            (72, 626, "(2021) Another synthetic reference. Synth J 2:11-20", 9),
        ),
        page(
            104,
            (72, 730, "3. Appendix: Definitions", 11),
            (72, 714, "Shock is defined in this synthetic text as hypotension that needs vasopressors.", 10),
            (72, 684, "Acknowledgements", 11),
            (72, 668, "We thank the synthetic reviewers.", 10),
        ),
    ]


def write_fixture_guideline(path: str | Path) -> Path:
    return write_pdf(path, fixture_guideline_pages(), page_labels=["101", "102", "103", "104"])
