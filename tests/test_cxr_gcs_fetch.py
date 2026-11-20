"""The bucket copy step: commands, safety checks, resume and parallel-equals-serial, with no real copy."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from fate_icu.data.cxr import gcs_fetch as gf

PATHS = [f"files/p10/p10000{i:03d}/s5000{i:04d}/{i:08x}-aaaaaaaa.jpg" for i in range(60)]


class FakeBucket:
    """Stands in for ``gcloud storage cp``: writes the file, fails for chosen paths, records every call."""

    def __init__(self, fail=(), flaky=()):
        self.fail, self.flaky = set(fail), set(flaky)
        self.calls: list[list[str]] = []
        self.attempts: dict[str, int] = {}
        self.lock = threading.Lock()

    def __call__(self, command):
        source, target = command[-2], Path(command[-1])
        relative = source.split(".org/", 1)[1]
        with self.lock:
            self.calls.append(list(command))
            self.attempts[relative] = self.attempts.get(relative, 0) + 1
            attempt = self.attempts[relative]
        if relative in self.fail or (relative in self.flaky and attempt == 1):
            return 1, "ERROR: 403 does not have storage.objects.get access"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"jpeg bytes for " + relative.encode())
        return 0, ""


def cfg(**kw):
    return gf.FetchSettings(**{"billing_project": "fate-icu", "workers": 4, **kw})


def test_the_command_reads_the_exact_object_and_bills_the_project(tmp_path):
    bucket = FakeBucket()
    gf.fetch(PATHS[:1], tmp_path, cfg(), bucket)
    [command] = bucket.calls
    assert command[:3] == ["gcloud", "storage", "cp"]
    assert "--billing-project=fate-icu" in command
    assert command[-2] == f"gs://mimic-cxr-jpg-2.1.0.physionet.org/{PATHS[0]}"
    assert command[-1] == str(tmp_path / PATHS[0])
    assert not any("*" in part for part in command)  # PhysioNet allows exact names, not listing
    assert not any("@" in part for part in command)  # no account name is passed or logged


def test_the_folder_layout_is_kept_so_the_cache_step_finds_the_images(tmp_path):
    result = gf.fetch(PATHS, tmp_path, cfg(), FakeBucket())
    assert result.copied == len(PATHS) and not result.failed
    assert all((tmp_path / p).is_file() for p in PATHS)


def test_finished_files_are_skipped_so_an_interrupted_copy_resumes(tmp_path):
    first = FakeBucket()
    gf.fetch(PATHS[:20], tmp_path, cfg(), first)
    second = FakeBucket()
    result = gf.fetch(PATHS, tmp_path, cfg(), second)
    assert result.already_present == 20 and result.copied == 40
    assert len(second.calls) == 40 and not any(PATHS[i] in c[-2] for i in range(20) for c in second.calls)


def test_an_empty_file_is_not_a_finished_file(tmp_path):
    (tmp_path / PATHS[0]).parent.mkdir(parents=True)
    (tmp_path / PATHS[0]).write_bytes(b"")
    bucket = FakeBucket()
    assert gf.fetch(PATHS[:1], tmp_path, cfg(), bucket).copied == 1 and len(bucket.calls) == 1


def test_a_failed_copy_is_retried_then_reported_by_name_in_list_order(tmp_path):
    bucket = FakeBucket(fail={PATHS[7], PATHS[3]}, flaky={PATHS[5]})
    result = gf.fetch(PATHS[:10], tmp_path, cfg(retries=3), bucket)
    assert result.failed == [PATHS[3], PATHS[7]]  # in list order, not thread order
    assert bucket.attempts[PATHS[3]] == 3 and bucket.attempts[PATHS[5]] == 2
    assert result.copied == 8 and result.counts()["failed"] == 2


def test_parallel_copying_gives_the_same_result_as_serial(tmp_path):
    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"
    kw = dict(fail={PATHS[11]}, flaky={PATHS[2], PATHS[30]})
    serial = gf.fetch(PATHS, serial_dir, cfg(workers=1), FakeBucket(**kw))
    parallel = gf.fetch(PATHS, parallel_dir, cfg(workers=16), FakeBucket(**kw))
    assert serial.counts() == parallel.counts() and serial.failed == parallel.failed
    files = lambda d: {str(p.relative_to(d)): p.read_bytes() for p in d.rglob("*.jpg")}  # noqa: E731
    assert files(serial_dir) == files(parallel_dir)


def test_a_dry_run_copies_nothing_and_prints_a_template_not_a_patient_path(tmp_path, capsys):
    bucket = FakeBucket()
    result = gf.fetch(PATHS, tmp_path, cfg(dry_run=True), bucket)
    out = capsys.readouterr().out
    assert bucket.calls == [] and result.copied == 0 and "<image path>" in out and "p10000" not in out


@pytest.mark.parametrize(
    "bad",
    ["../etc/passwd", "files/../../x.jpg", "/files/p10/x.jpg", "files/p10/x.png", "other/p10/x.jpg",
     "files\\p10\\x.jpg", "files//p10/x.jpg"],
)  # fmt: skip
def test_paths_that_could_leave_the_destination_are_refused(bad, tmp_path):
    assert not gf.valid_relative(bad)
    listing = tmp_path / "list.txt"
    listing.write_text(f"{PATHS[0]}\n{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not image paths"):
        gf.read_list(listing)


def test_the_list_is_sorted_unique_and_ignores_blank_lines(tmp_path):
    listing = tmp_path / "list.txt"
    listing.write_text(f"\n{PATHS[3]}\n{PATHS[1]}\n\n{PATHS[3]}\n", encoding="utf-8")
    assert gf.read_list(listing) == [PATHS[1], PATHS[3]]


def test_the_requester_pays_project_is_required_and_the_counts_are_sane():
    with pytest.raises(ValueError, match="billing-project"):
        gf.FetchSettings(billing_project=" ")
    with pytest.raises(ValueError):
        gf.FetchSettings(billing_project="p", workers=0)


def test_the_metadata_file_is_copied_once(tmp_path):
    bucket = FakeBucket()
    assert gf.fetch_metadata(tmp_path, cfg(), bucket) is True
    assert bucket.calls[0][-2].endswith("/mimic-cxr-2.0.0-metadata.csv.gz")
    assert gf.fetch_metadata(tmp_path, cfg(), bucket) is True and len(bucket.calls) == 1


def test_credentialed_images_are_never_copied_on_the_laptop(settings, monkeypatch):
    laptop = settings.model_copy(update={"machine": "laptop"})
    with pytest.raises(SystemExit, match="never on the laptop"):
        gf.check_machine(laptop)
    for machine in ("gcp", "lab"):
        gf.check_machine(settings.model_copy(update={"machine": machine}))
    monkeypatch.setattr(gf.sys, "platform", "win32")
    with pytest.raises(SystemExit, match="GCP VM"):
        gf.main(["--billing-project", "fate-icu", "--list", "x.txt"])


def test_the_list_comes_from_a_link_run_or_a_file_but_not_both(settings, tmp_path):
    run = tmp_path / "cxr" / "run1"
    run.mkdir(parents=True)
    (run / "image_list.txt").write_text(PATHS[0] + "\n", encoding="utf-8")
    assert gf.resolve_list(settings, str(run), None) == run / "image_list.txt"
    with pytest.raises(SystemExit, match="exactly one"):
        gf.resolve_list(settings, str(run), run / "image_list.txt")
    with pytest.raises(SystemExit, match="exactly one"):
        gf.resolve_list(settings, None, None)
    with pytest.raises(FileNotFoundError, match="link step"):
        gf.resolve_list(settings, str(tmp_path), None)
