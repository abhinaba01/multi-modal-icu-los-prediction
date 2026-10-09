"""Raw MIMIC-IV CSV(.gz) loading utilities.

Two access patterns:

* :meth:`MIMICLoader.load_table` reads a small table fully into memory
  (icustays, admissions, patients, diagnoses_icd).
* :meth:`MIMICLoader.iter_chunks` streams a large event table in row chunks so
  the 3.3 GB ``chartevents`` / 2.5 GB ``labevents`` never need to fit in RAM.
  Callers filter each chunk to the itemids / window they need and accumulate.

pandas infers gzip from the ``.csv.gz`` suffix, so the same code path serves both
compressed and plain CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import raw_table_path

LOGGER = logging.getLogger(__name__)


class MIMICLoader:
    """Loads MIMIC-IV tables from the configured raw CSV directory."""

    def __init__(self, config: dict[str, Any]):
        """Initialise the loader.

        Args:
            config: Parsed project configuration.
        """
        self.config = config
        self.chunksize = int(config["data"].get("csv_chunksize", 1_000_000))

    def path(self, table_name: str) -> Path:
        """Return the resolved path for a raw table (guards on existence).

        Args:
            table_name: Key under ``data.files``.

        Returns:
            Existing path to the table.
        """
        return raw_table_path(self.config, table_name)

    def load_table(
        self,
        table_name: str,
        usecols: list[str] | None = None,
        parse_dates: list[str] | None = None,
        dtype: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Read a full table into a DataFrame.

        Args:
            table_name: Key under ``data.files``.
            usecols: Optional subset of columns.
            parse_dates: Optional columns to parse as datetimes.
            dtype: Optional explicit dtypes.

        Returns:
            The table as a DataFrame.
        """
        path = self.path(table_name)
        LOGGER.info("Loading table %s from %s", table_name, path)
        return pd.read_csv(
            path,
            usecols=usecols,
            parse_dates=parse_dates,
            dtype=dtype,
            low_memory=False,
        )

    def iter_chunks(
        self,
        table_name: str,
        usecols: list[str] | None = None,
        chunksize: int | None = None,
        dtype: dict[str, Any] | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Iterate over a table in row chunks.

        Args:
            table_name: Key under ``data.files``.
            usecols: Optional subset of columns (strongly recommended for the big
                event tables to cut I/O and memory).
            chunksize: Rows per chunk; defaults to ``data.csv_chunksize``.
            dtype: Optional explicit dtypes.

        Yields:
            DataFrame chunks.
        """
        path = self.path(table_name)
        rows = chunksize or self.chunksize
        LOGGER.info("Streaming table %s from %s in chunks of %s rows", table_name, path, rows)
        yield from pd.read_csv(
            path,
            usecols=usecols,
            chunksize=rows,
            dtype=dtype,
            low_memory=False,
        )
