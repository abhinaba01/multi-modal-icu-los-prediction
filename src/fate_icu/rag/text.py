"""Text normalisation shared by every source parser (Stage 5).

PDFs and XML bring ligatures, non-breaking spaces, soft hyphens and zero-width characters
that would split one word into several BM25 tokens. Every paragraph passes through
:func:`normalise_text` before chunking.
"""

from __future__ import annotations

import re
import unicodedata

# soft hyphen, zero-width space / non-joiner / joiner, word joiner, byte-order mark
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """NFKC, invisible and control characters removed, whitespace runs collapsed to one space.

    NFKC turns ligatures (``ﬁ`` -> ``fi``) and non-breaking spaces into plain characters.

    Args:
        text: Raw text from a parser.

    Returns:
        One line of text without leading or trailing whitespace.
    """
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    text = _CONTROL.sub(" ", text)
    return _SPACE.sub(" ", text).strip()
