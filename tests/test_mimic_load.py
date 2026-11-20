"""The MIMIC loader: command construction, safety checks and skip logic, with no real download."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fate_icu.utils import mimic_load as ml


class Recorder:
    """A stand-in for subprocess: records every command and succeeds (or fails on demand)."""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[list[str], Path | None]] = []
        self.fail_on = fail_on

    def __call__(self, command, cwd=None):
        self.calls.append((list(command), cwd))
        return 1 if self.fail_on and self.fail_on in " ".join(command) else 0


def cfg(tmp_path, **kw):
    base = dict(dest=tmp_path / "mimiciv" / "3.1", code_dir=tmp_path / "mimic-code", user="alice")
    return ml.Settings(**{**base, **kw})


def test_download_asks_wget_for_the_password_and_never_takes_one(tmp_path):
    rec = Recorder()
    ml.download(cfg(tmp_path), rec)
    [(command, _)] = rec.calls
    assert (
        command[0] == "wget"
        and "--ask-password" in command
        and command[command.index("--user") + 1] == "alice"
    )
    assert command[-1] == "https://physionet.org/files/mimiciv/3.1/"
    assert {"-N", "-c", "-np"} <= set(command)  # resumable and checks timestamps
    assert not any("pass" in c.lower() and c != "--ask-password" for c in command)


def test_download_needs_a_username(tmp_path):
    with pytest.raises(SystemExit, match="--user"):
        ml.download(cfg(tmp_path, user=None), Recorder())


def test_a_dry_run_prints_commands_and_runs_nothing(tmp_path, capsys):
    rec = Recorder()
    ml.download(cfg(tmp_path, dry_run=True), rec)
    assert rec.calls == [] and "wget" in capsys.readouterr().out
    ml.verify(cfg(tmp_path, dry_run=True))  # no manifest needed


def test_a_failing_command_stops_the_run_with_its_name(tmp_path):
    with pytest.raises(SystemExit, match="exit code 1: wget"):
        ml.download(cfg(tmp_path), Recorder(fail_on="wget"))


def test_preflight_reports_missing_tools_and_a_small_disk(tmp_path):
    with pytest.raises(SystemExit, match="missing tools: wget"):
        ml.preflight(cfg(tmp_path), Recorder(), which=lambda t: None if t == "wget" else "/usr/bin/" + t)
    with pytest.raises(SystemExit, match="data disk mounted"):
        ml.preflight(cfg(tmp_path, min_free_gb=10**9), Recorder(), which=lambda t: "/usr/bin/" + t)


def test_preflight_clones_mimic_code_at_the_pinned_tag_only_when_missing(tmp_path):
    rec = Recorder()
    ml.preflight(cfg(tmp_path, min_free_gb=0), rec, which=lambda t: "/usr/bin/" + t)
    [(command, _)] = rec.calls
    assert command[:4] == ["git", "clone", "--branch", "v3.0.1"] and "--depth" in command
    settings = cfg(tmp_path, min_free_gb=0)
    settings.build_dir.mkdir(parents=True)
    again = Recorder()
    ml.preflight(settings, again, which=lambda t: "/usr/bin/" + t)
    assert again.calls == []


def test_load_runs_the_four_mimic_code_scripts_in_order_from_the_build_folder(tmp_path):
    settings = cfg(tmp_path)
    (settings.dest / "hosp").mkdir(parents=True)
    (settings.dest / "icu").mkdir()
    rec = Recorder()
    ml.load(settings, rec, count=lambda c, sql: None if "icustays" in sql else 1)  # new database, no stays
    commands = [c for c, _ in rec.calls]
    assert ["createdb", "mimiciv"] not in commands  # the database already exists in this case
    scripts = [c[c.index("-f") + 1] for c in commands if "-f" in c]
    assert scripts == ["create.sql", "load_gz.sql", "constraint.sql", "index.sql"]
    assert all("ON_ERROR_STOP=1" in c for c in commands if "psql" in c[0])
    assert any(f"mimic_data_dir={settings.dest}" in c for c in commands)
    assert all(cwd == settings.build_dir for c, cwd in rec.calls if "psql" in c[0])


def test_load_creates_the_database_only_when_it_does_not_exist(tmp_path):
    settings = cfg(tmp_path)
    (settings.dest / "hosp").mkdir(parents=True)
    (settings.dest / "icu").mkdir()
    absent = Recorder()
    ml.load(settings, absent, count=lambda c, sql: None)
    assert ["createdb", "mimiciv"] in [c for c, _ in absent.calls]
    present = Recorder()
    ml.load(settings, present, count=lambda c, sql: None if "icustays" in sql else 1)
    assert ["createdb", "mimiciv"] not in [c for c, _ in present.calls]


def test_a_finished_load_is_skipped_unless_forced(tmp_path, capsys):
    settings = cfg(tmp_path)
    rec = Recorder()
    ml.load(settings, rec, count=lambda c, sql: 94458)
    assert rec.calls == [] and "skipped" in capsys.readouterr().out
    (settings.dest / "hosp").mkdir(parents=True)
    (settings.dest / "icu").mkdir()
    forced = Recorder()
    ml.load(cfg(tmp_path, force=True), forced, count=lambda c, sql: 94458)
    assert forced.calls  # rebuilt


def test_load_needs_the_downloaded_folders(tmp_path):
    with pytest.raises(SystemExit, match="download step first"):
        ml.load(cfg(tmp_path), Recorder(), count=lambda c, sql: None)


def test_concepts_run_both_scripts_from_the_concepts_folder_and_skip_when_built(tmp_path):
    settings = cfg(tmp_path)
    rec = Recorder()
    ml.concepts(settings, rec, count=lambda c, sql: 0)
    scripts = [c[c.index("-f") + 1] for c, _ in rec.calls]
    assert scripts == ["postgres-functions.sql", "postgres-make-concepts.sql"]
    assert all(cwd == settings.concepts_dir for _, cwd in rec.calls)
    skipped = Recorder()
    ml.concepts(settings, skipped, count=lambda c, sql: 12258)
    assert skipped.calls == []


def test_the_check_prints_counts_and_fails_on_empty_tables_or_a_wrong_stay_count(tmp_path, capsys):
    good = {"icustays": 94458, "sofa": 5, "norepinephrine": 7}
    count = lambda c, sql: next(v for k, v in good.items() if k in sql)  # noqa: E731
    assert ml.check(cfg(tmp_path), count)["icu_stays"] == 94458
    assert "94458" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="sofa_rows"):
        ml.check(cfg(tmp_path), lambda c, sql: 0 if "sofa" in sql else 1)
    with pytest.raises(SystemExit, match="expected 94458"):
        ml.check(cfg(tmp_path), lambda c, sql: 140)  # the demo's count, not v3.1's


def test_verify_accepts_a_matching_list_and_rejects_damage(tmp_path):
    settings = cfg(tmp_path)
    (settings.dest / "hosp").mkdir(parents=True)
    data = b"subject_id\n1\n"
    (settings.dest / "hosp" / "patients.csv.gz").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (settings.dest / "SHA256SUMS.txt").write_text(f"{digest}  hosp/patients.csv.gz\n", encoding="utf-8")
    assert ml.verify(settings)["ok"] is True
    (settings.dest / "hosp" / "patients.csv.gz").write_bytes(b"damaged")
    with pytest.raises(SystemExit, match="damaged"):
        ml.verify(settings)


def test_the_loader_refuses_to_run_on_windows(monkeypatch):
    monkeypatch.setattr(ml.sys, "platform", "win32")
    with pytest.raises(SystemExit, match="GCP VM"):
        ml.main(["--step", "check"])


def test_unknown_steps_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="unknown step"):
        ml.run_steps(["explode"], cfg(tmp_path), Recorder())
