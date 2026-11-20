"""SHA-256 manifests for moving folders between machines: make, check, and what each catches."""

from __future__ import annotations

import hashlib

import pytest

from fate_icu.utils.checksums import (
    MANIFEST_NAME,
    check_manifest,
    list_files,
    main,
    make_manifest,
    read_manifest,
)
from fate_icu.utils.hashing import binary_sha256


def _tree(tmp_path):
    root = tmp_path / "data"
    (root / "raw" / "pubmed").mkdir(parents=True)
    (root / "raw" / "pubmed" / "a b.xml").write_bytes(b"alpha\r\nbeta")  # a space, and CRLF to keep as it is
    (root / "raw" / "top.txt").write_bytes(b"top")
    (root / "rag_dryrun_pubmed").mkdir()
    (root / "rag_dryrun_pubmed" / "x.bin").write_bytes(b"scratch")
    (root / "notes.tmp").write_bytes(b"temporary")
    return root


def test_hashes_are_of_the_raw_bytes(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"one\r\ntwo\r\n")
    assert binary_sha256(path) == hashlib.sha256(b"one\r\ntwo\r\n").hexdigest()  # CRLF is not normalised
    assert binary_sha256(path, chunk_size=3) == binary_sha256(path)


def test_make_lists_sorted_relative_files_without_the_manifest_or_excluded_paths(tmp_path):
    root = _tree(tmp_path)
    report = make_manifest(root, exclude=["rag_dryrun_*", "*.tmp"])
    assert report["files"] == 2 and report["manifest"] == str(root / MANIFEST_NAME)
    entries = read_manifest(root / MANIFEST_NAME)
    assert list(entries) == ["raw/pubmed/a b.xml", "raw/top.txt"]  # sorted, posix separators
    assert entries["raw/pubmed/a b.xml"] == hashlib.sha256(b"alpha\r\nbeta").hexdigest()
    again = make_manifest(root, exclude=["rag_dryrun_*", "*.tmp"])  # the manifest does not list itself
    assert again["files"] == 2
    assert [p.as_posix() for p in list_files(root)] == [
        "notes.tmp",
        "rag_dryrun_pubmed/x.bin",
        "raw/pubmed/a b.xml",
        "raw/top.txt",
    ]


def test_check_passes_on_an_intact_copy_and_ignores_extra_files(tmp_path):
    root = _tree(tmp_path)
    make_manifest(root, exclude=["rag_dryrun_*", "*.tmp"])
    result = check_manifest(root)
    assert result["ok"] and (result["files_listed"], result["files_ok"]) == (2, 2)
    assert (
        result["extra_files_not_in_manifest"] == 2
    )  # the scratch folder and notes.tmp: reported, not an error


def test_check_catches_a_changed_a_missing_and_a_truncated_file(tmp_path):
    root = _tree(tmp_path)
    make_manifest(root, exclude=["rag_dryrun_*", "*.tmp"])
    (root / "raw" / "top.txt").write_bytes(b"top!")  # changed
    (root / "raw" / "pubmed" / "a b.xml").unlink()  # missing
    result = check_manifest(root)
    assert (
        not result["ok"]
        and result["changed"] == ["raw/top.txt"]
        and result["missing"] == ["raw/pubmed/a b.xml"]
    )
    assert (result["n_changed"], result["n_missing"], result["files_ok"]) == (1, 1, 0)
    (root / "raw" / "pubmed" / "a b.xml").write_bytes(b"alpha\r\nbe")  # a copy cut short
    assert check_manifest(root)["changed"] == ["raw/pubmed/a b.xml", "raw/top.txt"]


def test_loud_failures(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(FileNotFoundError, match="copy the manifest"):
        check_manifest(root)
    with pytest.raises(FileNotFoundError):
        make_manifest(tmp_path / "nowhere")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        make_manifest(empty)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / MANIFEST_NAME).write_text("not a manifest line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_manifest(bad / MANIFEST_NAME)
    digest = "0" * 64
    (bad / MANIFEST_NAME).write_text(f"{digest}  a.txt\n{digest}  a.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        read_manifest(bad / MANIFEST_NAME)


def test_command_line_exit_status(tmp_path, capsys):
    root = _tree(tmp_path)
    main(["make", str(root), "--exclude", "rag_dryrun_*", "--exclude", "*.tmp"])
    assert '"files": 2' in capsys.readouterr().out
    main(["check", str(root)])
    assert '"ok": true' in capsys.readouterr().out
    (root / "raw" / "top.txt").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="does not match"):
        main(["check", str(root)])


def test_read_manifest_accepts_one_space_separator_as_physionet_writes_it(tmp_path):
    digest = "a" * 64
    manifest = tmp_path / "SHA256SUMS.txt"
    lines = [f"{digest} hosp/admissions.csv.gz", f"{digest}  icu/icustays.csv.gz", f"{digest} *LICENSE.txt"]
    manifest.write_text(chr(10).join(lines) + chr(10))
    assert read_manifest(manifest) == {
        "hosp/admissions.csv.gz": digest,
        "icu/icustays.csv.gz": digest,
        "LICENSE.txt": digest,
    }
