"""Grade the passages of a labelling sheet in your terminal (Stage 5; you run this, not Claude).

    python -m fate_icu.rag.label --config configs/rag/full.yaml
    python -m fate_icu.rag.label --sheet <path to labelling_sheet.tsv>

Shows one passage at a time with its query and asks for a grade (0, 1 or 2; see
docs/rag_grading_guide.md). Every answer is saved to the sheet at once, so you can stop with
"q" at any time and pick up later where you left off. The order is shuffled and shows no
system names, so the grading stays blind.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from fate_icu.rag.evaluate import SHEET_FILE, read_labelling_sheet, write_sheet_rows

RUBRIC = """\
2 = directly relevant: the passage addresses this driver's link to the ICU course, length of
    stay, severity or outcome in critically ill adults, or states a guideline recommendation
    about it.
1 = related background: about the driver or condition (definition, mechanism, normal range,
    management) but not about its link to ICU outcomes.
0 = off-topic: another subject, a population with nothing transferable, or unreadable text.
Judge the passage on its own; ignore who published it."""
PROMPT = "grade 0/1/2, s = skip, b = back, h = help, q = save and quit: "


def run(
    sheet: str | Path,
    *,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    width: int = 100,
) -> dict[str, int]:
    """The grading loop; ``ask`` and ``say`` stand in for the terminal (tests)."""
    sheet = Path(sheet)
    rows = read_labelling_sheet(sheet)
    total = len(rows)
    position = next((i for i, row in enumerate(rows) if row.grade is None), total)
    history: list[int] = []
    if position == total:
        say(f"All {total} passages are graded.")
    while position < total:
        row = rows[position]
        graded = sum(r.grade is not None for r in rows)
        place = ", ".join(part for part in (row.section, row.locator) if part)
        say("")
        say(f"[{graded}/{total} graded]  {row.query_id}  passage {row.item}")
        say(f"QUERY: {row.query}")
        say(f"TITLE: {row.title}" + (f"  ({place})" if place else ""))
        say(textwrap.fill(row.text, width=width))
        if row.grade is not None:
            say(f"(current grade: {row.grade})")
        answer = ask(PROMPT).strip().lower()
        if answer in ("0", "1", "2"):
            rows[position] = replace(row, grade=int(answer))
            write_sheet_rows(sheet, rows)
            history.append(position)
            ungraded = [i for i, r in enumerate(rows) if r.grade is None and i > position]
            position = (
                ungraded[0] if ungraded else next((i for i, r in enumerate(rows) if r.grade is None), total)
            )
        elif answer == "s":
            position += 1
        elif answer == "b":
            position = history.pop() if history else max(0, position - 1)
        elif answer == "h":
            say(RUBRIC)
        elif answer == "q":
            break
        else:
            say("Please type 0, 1, 2, s, b, h or q.")
    graded = sum(r.grade is not None for r in rows)
    say(f"Saved. {graded} of {total} passages graded.")
    return {"graded": graded, "total": total}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="configs/rag/full.yaml (finds the sheet)")
    parser.add_argument("--sheet", help="the labelling sheet")
    args = parser.parse_args(argv)
    if args.sheet:
        sheet = Path(args.sheet)
    elif args.config:
        from fate_icu.rag.config import load_config, rag_paths
        from fate_icu.utils.env import load_settings

        config = load_config(args.config)
        sheet = rag_paths(config, load_settings()).eval / SHEET_FILE
    else:
        parser.error("give --config or --sheet")
    run(sheet)


if __name__ == "__main__":
    main()
