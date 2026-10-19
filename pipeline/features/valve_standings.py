"""Official global standings with conservative, explicitly proxy availability.

The collector preserves first-addition Git blobs, not later corrected snapshots.
Availability is max(snapshot midnight UTC + seven days, first committer clock),
which is NOT certified public release time. Organization strength is the fixed
maximum-points roster aggregate, not evidence of a match's actual starters.
Only casefold and whitespace normalization are used; aliases are never inferred.
This module performs no network access.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from collections import defaultdict


VRS_FEATURE_COLUMNS = [
    "vrs_points_diff", "vrs_log_rank_diff", "vrs_available", "vrs_age_days",
    "vrs_minimum_roster_count",
]
VRS_MIRROR_RULES = {
    **{name: {"operation": "negate", "source": name}
       for name in ("vrs_points_diff", "vrs_log_rank_diff")},
    **{name: {"operation": "identity", "source": name}
       for name in ("vrs_available", "vrs_age_days", "vrs_minimum_roster_count")},
}
ROW_COLUMNS = ["snapshot_date", "available_at", "team", "points", "rank",
               "roster", "commit", "source_path", "sha256"]
SOURCE_PATTERN = re.compile(
    r"invitation/(202[456])/standings_global_(202[456])_(\d{2})_(\d{2})\.md\Z")
AVAILABILITY_POLICY = "max(snapshot_date_midnight_utc_plus_7_days,first_addition_committer_utc)"
PROVENANCE_LIMIT = (
    "Git committer clocks and seven-day lag are retrospective availability proxies, "
    "not certified publication times or source-rights guarantees; first-addition "
    "blobs exclude later corrections. Maximum-points organization aggregates "
    "are not verified match starters."
)


def _day(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("date must be a canonical YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must be a canonical YYYY-MM-DD string") from exc
    if parsed.isoformat() != value:
        raise ValueError("date must be a canonical YYYY-MM-DD string")
    return parsed


def _clock(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("availability must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("availability must be a timezone-aware ISO timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValueError("availability must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_team(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("team must be a nonempty name")
    return " ".join(value.casefold().split())


def _provenance(snapshot_date, available_at, commit, source_path, sha256) -> dict:
    snapshot = _day(snapshot_date)
    clock = _clock(available_at)
    if clock < datetime.combine(snapshot, time(), timezone.utc) + timedelta(days=7):
        raise ValueError("availability precedes the conservative seven-day snapshot lag")
    if not isinstance(commit, str) or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
        raise ValueError("commit must be a full Git object hash")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("sha256 must be a full lowercase SHA256 hash")
    match = SOURCE_PATTERN.fullmatch(source_path) if isinstance(source_path, str) else None
    if not match or match[1] != match[2] or f"{match[2]}-{match[3]}-{match[4]}" != snapshot_date:
        raise ValueError("source_path must name the matching dated global snapshot")
    return {"snapshot_date": snapshot_date, "available_at": _iso(clock),
            "commit": commit, "source_path": source_path, "sha256": sha256}


def _validate_row(record: dict) -> dict:
    if not isinstance(record, dict) or any(key not in record for key in ROW_COLUMNS):
        raise ValueError("standing row lacks required values or provenance")
    result = _provenance(**{key: record[key] for key in
                           ("snapshot_date", "available_at", "commit", "source_path", "sha256")})
    normalize_team(record["team"])
    points, rank, roster = record["points"], record["rank"], record["roster"]
    if isinstance(points, bool) or not isinstance(points, (int, float)) or not math.isfinite(points) or points < 0:
        raise ValueError("standing points must be finite and nonnegative")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("standing rank must be a positive integer")
    if not isinstance(roster, (list, tuple)) or not roster:
        raise ValueError("standing roster must be a nonempty list")
    normalized = [normalize_team(player) for player in roster]
    if len(set(normalized)) != len(normalized):
        raise ValueError("standing roster contains duplicate players")
    result.update(team=record["team"].strip(), points=points, rank=rank, roster=list(roster))
    return result


def parse_standings(markdown, *, snapshot_date, available_at, commit, source_path, sha256) -> list[dict]:
    """Parse the official pipe table and reject malformed or duplicate evidence."""
    if isinstance(markdown, bytes):
        raw = markdown
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("standings markdown is not UTF-8") from exc
    elif isinstance(markdown, str):
        text, raw = markdown, markdown.encode("utf-8")
    else:
        raise ValueError("standings markdown must be text or original bytes")
    provenance = _provenance(snapshot_date, available_at, commit, source_path, sha256)
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("standings markdown SHA256 mismatch")
    rows, header, separator, finished = [], None, False, False
    for line in text.splitlines():
        if "|" not in line:
            if header is not None and separator:
                finished = True
            continue
        table_line = line.strip()
        if table_line.startswith("|"):
            table_line = table_line[1:]
        if table_line.endswith("|"):
            table_line = table_line[:-1]
        cells = [cell.strip() for cell in table_line.split("|")]
        # The final details header may be empty, but its column still exists.
        if header is None:
            if [cell.casefold() for cell in cells[:4]] == ["standing", "points", "team name", "roster"]:
                if len(cells) not in (4, 5):
                    raise ValueError("unexpected standings table columns")
                header = len(cells)
            continue
        if finished:
            raise ValueError("unexpected additional table after standings")
        if not separator:
            if len(cells) != header or any(re.fullmatch(r":?-+:?", cell) is None for cell in cells):
                raise ValueError("malformed standings table separator")
            separator = True
            continue
        if len(cells) != header:
            raise ValueError("malformed standings row column count")
        if re.fullmatch(r"[1-9][0-9]*", cells[0]) is None or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", cells[1]) is None:
            raise ValueError("malformed standings rank or points")
        points = float(cells[1]) if "." in cells[1] else int(cells[1])
        rows.append(_validate_row({**provenance, "rank": int(cells[0]), "points": points,
                                   "team": cells[2], "roster": [name.strip() for name in cells[3].split(",")]}))
    if not rows or not separator:
        raise ValueError("global standings table missing or empty")
    # Use the same duplicate and per-snapshot evidence validation as offline intake.
    return StandingsHistory(rows).records


class StandingsHistory:
    """Read-only global-snapshot lookup, with no per-team historical backfill."""

    def __init__(self, records, *, manifest=None):
        validated = [_validate_row(record) for record in records]
        grouped = defaultdict(list)
        for row in validated:
            grouped[row["snapshot_date"]].append(row)
        self._snapshots = []
        self._records = []
        for snapshot_date in sorted(grouped):
            rows = sorted(grouped[snapshot_date], key=lambda row: row["rank"])
            provenance = {(row["available_at"], row["commit"], row["source_path"], row["sha256"]) for row in rows}
            if len(provenance) != 1:
                raise ValueError("conflicting provenance within a global snapshot")
            ranks, identities, teams = set(), set(), defaultdict(list)
            for row in rows:
                name = normalize_team(row["team"])
                roster_key = tuple(sorted(normalize_team(player) for player in row["roster"]))
                identity = (name, roster_key)
                if row["rank"] in ranks or identity in identities:
                    raise ValueError("duplicate rank or organization roster within snapshot")
                ranks.add(row["rank"])
                identities.add(identity)
                teams[name].append(row)
            aggregates = {}
            for name, entries in teams.items():
                best = min(entries, key=lambda row: (-row["points"], row["rank"],
                           tuple(sorted(normalize_team(player) for player in row["roster"]))))
                aggregates[name] = (best["points"], best["rank"], len(entries))
            self._snapshots.append((_day(snapshot_date), _clock(rows[0]["available_at"]), aggregates))
            self._records.extend(rows)
        self._manifest = deepcopy(manifest) if manifest is not None else None

    @property
    def records(self) -> list[dict]:
        return deepcopy(self._records)

    @property
    def manifest(self) -> dict | None:
        return deepcopy(self._manifest)

    def features(self, team1: str, team2: str, day: str) -> dict:
        left, right = normalize_team(team1), normalize_team(team2)
        if left == right:
            raise ValueError("opposing normalized teams must be distinct")
        current = _day(day)
        origin = datetime.combine(current, time(), timezone.utc)
        result = dict.fromkeys(VRS_FEATURE_COLUMNS, 0)
        snapshot = next((item for item in reversed(self._snapshots) if item[1] < origin), None)
        if snapshot is None:
            return result
        snapshot_day, _, teams = snapshot
        result["vrs_age_days"] = (current - snapshot_day).days
        if left not in teams or right not in teams:
            return result
        p1, rank1, count1 = teams[left]
        p2, rank2, count2 = teams[right]
        result.update(vrs_points_diff=p1 - p2,
                      vrs_log_rank_diff=math.log1p(rank2) - math.log1p(rank1),
                      vrs_available=1, vrs_minimum_roster_count=min(count1, count2))
        return result


def _saved_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("saved provenance requires a relative file path")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("saved evidence path escapes artifact directory")
    return resolved


def load_standings(directory) -> StandingsHistory:
    """Load saved CSV plus manifest, verifying every raw first-publication blob.

    The CSV must exactly equal a fresh parse of the hash-verified raw artifacts.
    No network or Git repository is required for future inference.
    """
    root = Path(directory).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
                manifest.get("availability_policy") != AVAILABILITY_POLICY or
                not isinstance(manifest.get("snapshots"), list) or not manifest["snapshots"]):
            raise ValueError("invalid standings manifest schema or empty snapshots")
        records, dates = [], set()
        for snapshot in manifest["snapshots"]:
            if not isinstance(snapshot, dict):
                raise ValueError("invalid snapshot manifest entry")
            metadata = {key: snapshot[key] for key in
                        ("snapshot_date", "available_at", "commit", "source_path", "sha256")}
            _provenance(**metadata)
            first_commit_at = _clock(snapshot["first_commit_at"])
            lagged = datetime.combine(_day(snapshot["snapshot_date"]), time(), timezone.utc) + timedelta(days=7)
            if _clock(snapshot["available_at"]) != max(lagged, first_commit_at):
                raise ValueError("snapshot availability does not match declared proxy policy")
            if snapshot["snapshot_date"] in dates:
                raise ValueError("duplicate global snapshot in manifest")
            dates.add(snapshot["snapshot_date"])
            raw = _saved_path(root, snapshot["raw_path"]).read_bytes()
            blob_hash = snapshot["git_blob_hash"]
            if not isinstance(blob_hash, str) or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", blob_hash) is None:
                raise ValueError("invalid recorded Git blob hash")
            git_bytes = f"blob {len(raw)}\0".encode("ascii") + raw
            hash_function = hashlib.sha1 if len(blob_hash) == 40 else hashlib.sha256
            if hash_function(git_bytes).hexdigest() != blob_hash:
                raise ValueError("saved original Git blob hash mismatch")
            parsed = parse_standings(raw, **metadata)
            if len(parsed) != snapshot["row_count"]:
                raise ValueError("snapshot row count mismatch")
            records.extend(parsed)
        csv_path = _saved_path(root, manifest["records_path"])
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != manifest["records_sha256"]:
            raise ValueError("saved standings CSV SHA256 mismatch")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ROW_COLUMNS:
                raise ValueError("saved standings CSV schema mismatch")
            saved = []
            for row in reader:
                row["points"] = float(row["points"])
                row["rank"] = int(row["rank"])
                row["roster"] = json.loads(row["roster"])
                saved.append(row)
        history = StandingsHistory(records, manifest=manifest)
        if StandingsHistory(saved).records != history.records:
            raise ValueError("saved CSV differs from original snapshot records")
        if manifest["snapshot_count"] != len(dates) or manifest["row_count"] != len(records):
            raise ValueError("manifest aggregate counts do not match saved evidence")
        return history
    except (OSError, KeyError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("missing or corrupt saved standings evidence") from exc
