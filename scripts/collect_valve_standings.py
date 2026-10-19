"""Opt-in official global standings acquisition from first-addition Git blobs.

A full-metadata, blob-filtered no-checkout clone is temporary and bounded by one
wall-clock deadline. --repository uses an existing complete local repository
without cloning or changing it. No model code calls this collector. Git clocks
plus the seven-day lag are availability proxies, not publication guarantees.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time as monotonic_time

# Standalone `python scripts/collect_valve_standings.py` needs the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.features.valve_standings import (
    AVAILABILITY_POLICY, PROVENANCE_LIMIT, ROW_COLUMNS, SOURCE_PATTERN,
    StandingsHistory, _clock, _day, _iso, parse_standings,
)


OFFICIAL_REPOSITORY = "https://github.com/ValveSoftware/counter-strike_regional_standings"
MAX_BLOB_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024 * 1024


class _Git:
    def __init__(self, deadline, *, local):
        self.deadline = deadline
        self.environment = os.environ.copy()
        self.environment.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
        if local:
            # A supplied incomplete partial repository must fail, not fetch silently.
            self.environment["GIT_NO_LAZY_FETCH"] = "1"

    def run(self, *args, repository=None) -> bytes:
        remaining = self.deadline - monotonic_time.monotonic()
        if remaining <= 0:
            raise ValueError("standings acquisition deadline exceeded")
        command = ["git"]
        if repository is not None:
            command += ["-C", str(repository)]
        command += list(args)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=self.environment, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            raise ValueError("Git acquisition failed; complete metadata and original blobs are required") from exc
        if len(stdout) > MAX_METADATA_BYTES:
            raise ValueError("Git metadata exceeds acquisition size bound")
        return stdout


def _first_additions(git: _Git, repository: Path, years: tuple[int, ...]) -> dict:
    if git.run("rev-parse", "--is-shallow-repository", repository=repository).strip() != b"false":
        raise ValueError("shallow Git repositories do not prove first introduction")
    # Full-history prevents path simplification hiding side-branch additions.
    # Ancestors precede descendants, so removal/re-addition never replaces a first blob.
    metadata = git.run(
        "log", "--all", "--full-history", "--reverse", "--topo-order", "--no-renames",
        "--diff-filter=A", "--format=%H%x00%cI", "--name-status", "--",
        ":(glob)invitation/*/standings_global_*.md", repository=repository,
    ).decode("utf-8")
    first, commit, clock = {}, None, None
    for line in metadata.splitlines():
        if not line:
            continue
        if "\x00" in line:
            parts = line.split("\x00")
            if len(parts) != 2:
                raise ValueError("malformed Git history metadata")
            commit, raw_clock = parts
            clock = _clock(raw_clock)
            continue
        status, separator, source_path = line.partition("\t")
        if status != "A" or not separator or commit is None or clock is None:
            raise ValueError("unexpected first-addition Git history record")
        match = SOURCE_PATTERN.fullmatch(source_path)
        if not match or int(match[1]) not in years:
            continue
        if match[1] != match[2]:
            raise ValueError("dated global snapshot has inconsistent source year")
        snapshot_date = f"{match[2]}-{match[3]}-{match[4]}"
        _day(snapshot_date)
        if source_path in first:
            earliest = first[source_path]["commit"]
            ancestor = git.run("merge-base", earliest, commit, repository=repository).decode("ascii").strip()
            if ancestor != earliest:
                raise ValueError("ambiguous incomparable first introductions of global snapshot")
        first.setdefault(source_path, {"commit": commit, "first_commit_at": _iso(clock),
                                       "source_path": source_path, "snapshot_date": snapshot_date})
    if not first:
        raise ValueError("no dated global standings snapshots found in selected complete history")
    return first


def _acquire(git: _Git, repository: Path, years: tuple[int, ...]) -> tuple[list, list]:
    additions = _first_additions(git, repository, years)
    snapshots, all_rows = [], []
    for source_path, introduction in sorted(additions.items(), key=lambda item: item[1]["snapshot_date"]):
        blob_ref = f"{introduction['commit']}:{source_path}"
        size = int(git.run("cat-file", "-s", blob_ref, repository=repository).strip())
        if size <= 0 or size > MAX_BLOB_BYTES:
            raise ValueError("original standings blob is empty or exceeds acquisition size bound")
        raw = git.run("show", blob_ref, repository=repository)
        if len(raw) != size:
            raise ValueError("original standings blob size mismatch")
        blob_hash = git.run("rev-parse", blob_ref, repository=repository).decode("ascii").strip()
        lagged = datetime.combine(_day(introduction["snapshot_date"]), time(), timezone.utc) + timedelta(days=7)
        available_at = _iso(max(lagged, _clock(introduction["first_commit_at"])))
        sha256 = hashlib.sha256(raw).hexdigest()
        rows = parse_standings(raw, snapshot_date=introduction["snapshot_date"],
                               available_at=available_at, commit=introduction["commit"],
                               source_path=source_path, sha256=sha256)
        snapshot = {**introduction, "available_at": available_at, "sha256": sha256,
                    "git_blob_hash": blob_hash, "raw_path": "raw/" + source_path,
                    "row_count": len(rows)}
        snapshots.append((snapshot, raw))
        all_rows.extend(rows)
    # Validate cross-row provenance before creating any output files.
    return snapshots, StandingsHistory(all_rows).records


def collect_standings(output_dir, *, repository=None, years=(2024, 2025, 2026), timeout=600) -> dict:
    """Preserve original bytes, CSV rows and a hash/commit/clock manifest.

    An existing nonempty output directory is rejected to protect old artifacts.
    A local repository must contain full history; no current-snapshot fallback is
    permitted. The public clone does not use --depth, checkout, or submodules.
    """
    years = tuple(sorted(set(years)))
    if not years or any(isinstance(year, bool) or not isinstance(year, int) or year not in (2024, 2025, 2026) for year in years):
        raise ValueError("years must be a nonempty subset of 2024, 2025, 2026")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number of seconds")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("standings output directory must be absent or empty")
    git = _Git(monotonic_time.monotonic() + timeout, local=repository is not None)
    if repository is not None:
        source = Path(repository).resolve()
        snapshots, records = _acquire(git, source, years)
        acquisition_source = str(source)
        acquisition_method = "provided_local_full_history_repository"
    else:
        with tempfile.TemporaryDirectory(prefix="valve-standings-") as temporary:
            source = Path(temporary) / "repository"
            git.run("clone", "--filter=blob:none", "--no-checkout", "--",
                    OFFICIAL_REPOSITORY, str(source))
            snapshots, records = _acquire(git, source, years)
        acquisition_source = OFFICIAL_REPOSITORY
        acquisition_method = "temporary_public_blob_filtered_full_metadata_clone_no_checkout"
    output.mkdir(parents=True, exist_ok=True)
    for snapshot, raw in snapshots:
        path = output / snapshot["raw_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    csv_path = output / "standings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "roster": json.dumps(record["roster"], ensure_ascii=False)})
    manifest = {
        "schema_version": 1, "intended_official_repository": OFFICIAL_REPOSITORY,
        "acquisition_source": acquisition_source, "acquisition_method": acquisition_method,
        "collected_at": _iso(datetime.now(timezone.utc)), "years": list(years),
        "availability_policy": AVAILABILITY_POLICY, "provenance_limit": PROVENANCE_LIMIT,
        "organization_aggregation": "highest_points_then_lowest_rank_then_normalized_roster",
        "team_matching": "casefold_and_whitespace_only_no_aliases",
        "records_path": "standings.csv", "records_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "row_count": len(records), "snapshot_count": len(snapshots),
        "snapshots": [snapshot for snapshot, _ in snapshots],
        "years_without_snapshots": [year for year in years if not any(
            snapshot["snapshot_date"].startswith(str(year)) for snapshot, _ in snapshots)],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository", type=Path,
                        help="Existing complete local Git repository; skip public clone")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--timeout", type=float, default=600,
                        help="Total Git acquisition wall-clock deadline in seconds")
    args = parser.parse_args(argv)
    try:
        manifest = collect_standings(args.output_dir, repository=args.repository,
                                    years=args.years, timeout=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"output_dir": str(args.output_dir), "snapshot_count": manifest["snapshot_count"],
                      "row_count": manifest["row_count"], "availability_policy": manifest["availability_policy"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
