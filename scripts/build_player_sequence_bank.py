#!/usr/bin/env python3
"""Build and extract player sequence bank from historical CS2 match data.

This CLI script:
1. Discovers and reads available player match records from:
   - StrandedPond: data/raw/strandedpond/player_stats.csv & matches.csv (if present)
   - OpenCS2: data/raw/opencs2/maps.csv & player_scopes.csv (if present)
   - Match Lineups: data/processed/match_lineups.csv (if present)
2. Normalizes records chronologically by date into a unified event format.
3. Ingests events into PlayerHistoryTracker to maintain rolling deques per player.
4. Serializes the tracker state to data/04_feature/player_sequence_bank.pt.
5. Computes, prints, and saves summary statistics:
   - Total distinct players tracked
   - History length distribution (% and count with >= 1, >= 5, >= 10, >= 16 maps)
   - Date range of sequence bank (earliest date, latest date, total span).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.features.player_sequence_builder import (
    PlayerHistoryTracker,
    PlayerMapRecord,
)

logger = logging.getLogger(__name__)


def _to_clean_str(val: Any) -> str:
    """Return clean stripped string or empty string if None/NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def _to_clean_timestamp(val: Any) -> pd.Timestamp | None:
    """Parse date value to a timezone-naive pd.Timestamp, or None if invalid."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        ts = pd.to_datetime(val)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source Ingestion Loaders
# ---------------------------------------------------------------------------

def load_strandedpond_records(
    stats_path: Path,
    matches_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Load player map records from StrandedPond player_stats.csv and matches.csv.

    Merges match perspective metadata (score, winner, starting side) when matches.csv
    is available.
    """
    if not stats_path.exists():
        return [], None

    logger.info("Loading StrandedPond player stats from %s", stats_path)
    pstats = pd.read_csv(stats_path).drop_duplicates()
    if pstats.empty:
        return [], str(stats_path)

    # Attempt to join with matches.csv if present
    matches_df: pd.DataFrame | None = None
    if matches_path is not None and matches_path.exists():
        logger.info("Enriching StrandedPond stats with matches from %s", matches_path)
        matches_df = pd.read_csv(matches_path).drop_duplicates()
    else:
        alt_matches = stats_path.parent / "matches.csv"
        if alt_matches.exists():
            logger.info("Enriching StrandedPond stats with matches from %s", alt_matches)
            matches_df = pd.read_csv(alt_matches).drop_duplicates()

    merged_df: pd.DataFrame
    if matches_df is not None and not matches_df.empty:
        left_m = matches_df[
            [
                "event",
                "date",
                "map",
                "team_left",
                "winner",
                "left_team_started",
                "score_left",
                "score_right",
            ]
        ].copy()
        left_m = left_m.rename(
            columns={
                "team_left": "team",
                "score_left": "rounds_won",
                "score_right": "rounds_lost",
                "left_team_started": "side",
            }
        )
        left_m["won"] = (left_m["team"] == left_m["winner"]).astype(float)
        left_m = left_m.drop(columns=["winner"]).drop_duplicates(
            subset=["event", "date", "map", "team"]
        )

        right_m = matches_df[
            [
                "event",
                "date",
                "map",
                "team_right",
                "winner",
                "left_team_started",
                "score_left",
                "score_right",
            ]
        ].copy()
        right_m = right_m.rename(
            columns={
                "team_right": "team",
                "score_right": "rounds_won",
                "score_left": "rounds_lost",
            }
        )
        right_m["side"] = (
            right_m["left_team_started"].map({"CT": "T", "T": "CT"}).fillna("")
        )
        right_m["won"] = (right_m["team"] == right_m["winner"]).astype(float)
        right_m = right_m.drop(
            columns=["winner", "left_team_started"]
        ).drop_duplicates(subset=["event", "date", "map", "team"])

        team_matches = pd.concat(
            [left_m, right_m], ignore_index=True
        ).drop_duplicates(subset=["event", "date", "map", "team"])

        merged_df = pd.merge(
            pstats, team_matches, on=["event", "date", "map", "team"], how="left"
        )
    else:
        merged_df = pstats
        merged_df["won"] = 0.0
        merged_df["side"] = ""

    records: list[dict[str, Any]] = []
    for _, row in merged_df.iterrows():
        pname = _to_clean_str(row.get("player_name") or row.get("player_id"))
        if not pname:
            continue
        ts = _to_clean_timestamp(row.get("date"))
        if ts is None:
            continue

        row_dict = row.to_dict()
        won_val = float(row.get("won", 0.0)) if pd.notna(row.get("won")) else 0.0
        side_val = _to_clean_str(row.get("side"))
        map_val = _to_clean_str(row.get("map") or row.get("map_name"))

        records.append(
            {
                "player_id": pname,
                "date": ts,
                "map_name": map_val,
                "side": side_val,
                "won": won_val,
                "stats_dict": row_dict,
                "source": "strandedpond",
            }
        )

    logger.info("Extracted %d records from StrandedPond", len(records))
    return records, str(stats_path)


def load_opencs2_records(
    maps_path: Path,
    scopes_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Load player map records from OpenCS2 dataset exports.

    Checks maps.csv and player_scopes.csv (aggregating whole-map statistics).
    """
    default_maps = Path("data/raw/opencs2/maps.csv")
    resolved_maps_path: Path | None = None
    if maps_path != default_maps:
        resolved_maps_path = maps_path if maps_path.exists() else None
    else:
        for p in [maps_path, Path("data/exports/opencs2/maps.csv"), Path("data/01_raw/opencs2_maps.csv")]:
            if p is not None and p.exists():
                resolved_maps_path = p
                break

    default_scopes = Path("data/exports/opencs2/player_scopes.csv")
    resolved_scopes_path: Path | None = None
    if scopes_path is not None and scopes_path != default_scopes:
        resolved_scopes_path = scopes_path if scopes_path.exists() else None
    else:
        candidate_scopes = [
            scopes_path,
            Path("data/raw/opencs2/player_scopes.csv"),
            Path("data/exports/opencs2/player_scopes.csv"),
            (resolved_maps_path.parent / "player_scopes.csv") if resolved_maps_path else None,
        ]
        for p in candidate_scopes:
            if p is not None and p.exists():
                resolved_scopes_path = p
                break

    if resolved_maps_path is None and resolved_scopes_path is None:
        return [], None

    # Load maps metadata if available
    maps_df: pd.DataFrame | None = None
    if resolved_maps_path is not None and resolved_maps_path.exists():
        logger.info("Loading OpenCS2 maps metadata from %s", resolved_maps_path)
        maps_df = pd.read_csv(resolved_maps_path).drop_duplicates(subset=["row_id"])

    # If maps.csv already has individual player rows
    if maps_df is not None:
        player_cols = {"player_name", "player_id", "steamid", "name"}.intersection(
            maps_df.columns
        )
        if player_cols:
            records: list[dict[str, Any]] = []
            pcol = next(iter(player_cols))
            for _, row in maps_df.iterrows():
                pname = _to_clean_str(row.get(pcol))
                if not pname:
                    continue
                ts = _to_clean_timestamp(row.get("match_date") or row.get("date"))
                if ts is None:
                    continue
                records.append(
                    {
                        "player_id": pname,
                        "date": ts,
                        "map_name": _to_clean_str(row.get("map_name") or row.get("map")),
                        "side": _to_clean_str(row.get("side")),
                        "won": float(row.get("won", 0.0)) if pd.notna(row.get("won")) else 0.0,
                        "stats_dict": row.to_dict(),
                        "source": "opencs2",
                    }
                )
            logger.info("Extracted %d player records directly from OpenCS2 maps.csv", len(records))
            return records, str(resolved_maps_path)

    # Otherwise load from player_scopes.csv
    if resolved_scopes_path is None or not resolved_scopes_path.exists():
        logger.info(
            "OpenCS2 maps.csv has match-level rows and no player_scopes.csv found; skipping."
        )
        return [], str(resolved_maps_path) if resolved_maps_path else None

    logger.info("Loading OpenCS2 player scopes from %s", resolved_scopes_path)
    scopes_df = pd.read_csv(resolved_scopes_path)
    if "side" in scopes_df.columns:
        all_side_df = scopes_df[scopes_df["side"] == "all"].copy()
    else:
        all_side_df = scopes_df.copy()

    # Optionally join with maps metadata for score/winner
    if maps_df is not None and "row_id" in maps_df.columns and "row_id" in all_side_df.columns:
        m_subset = maps_df[["row_id", "team1", "team2", "winner", "score1", "score2"]]
        all_side_df = pd.merge(all_side_df, m_subset, on="row_id", how="left")

    records = []
    for _, row in all_side_df.iterrows():
        pname = _to_clean_str(row.get("name") or row.get("steamid"))
        if not pname:
            continue
        ts = _to_clean_timestamp(row.get("match_date") or row.get("date"))
        if ts is None:
            continue

        row_dict = row.to_dict()
        if "n_rounds" in row_dict and "total_rounds" not in row_dict:
            row_dict["total_rounds"] = row_dict["n_rounds"]

        records.append(
            {
                "player_id": pname,
                "date": ts,
                "map_name": _to_clean_str(row.get("map_name")),
                "side": "Neutral",
                "won": 0.0,
                "stats_dict": row_dict,
                "source": "opencs2",
            }
        )

    logger.info("Extracted %d records from OpenCS2 player scopes", len(records))
    return records, str(resolved_scopes_path)


def load_match_lineups_records(
    lineups_path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    """Load player participation records from match_lineups.csv."""
    default_lineups = Path("data/processed/match_lineups.csv")
    resolved_path: Path | None = None
    if lineups_path != default_lineups:
        resolved_path = lineups_path if lineups_path.exists() else None
    else:
        for p in [lineups_path, Path("data/02_intermediate/match_lineups.csv")]:
            if p is not None and p.exists():
                resolved_path = p
                break

    if resolved_path is None:
        return [], None

    logger.info("Loading match lineups from %s", resolved_path)
    df = pd.read_csv(resolved_path)
    if df.empty:
        return [], str(resolved_path)

    if "team1_p1_name" in df.columns:
        valid_df = df[df["team1_p1_name"].notna() & (df["team1_p1_name"] != "")].copy()
    else:
        valid_df = df.copy()

    records: list[dict[str, Any]] = []
    for _, row in valid_df.iterrows():
        ts = _to_clean_timestamp(row.get("date"))
        if ts is None:
            continue

        winner = _to_clean_str(row.get("winner"))
        s1 = float(row.get("score1", 0)) if pd.notna(row.get("score1")) else 0.0
        s2 = float(row.get("score2", 0)) if pd.notna(row.get("score2")) else 0.0

        # Team 1 players
        t1_won = 1.0 if winner == "team1" else 0.0
        for i in range(1, 6):
            pname = _to_clean_str(row.get(f"team1_p{i}_name"))
            if pname:
                records.append(
                    {
                        "player_id": pname,
                        "date": ts,
                        "map_name": "",
                        "side": "Neutral",
                        "won": t1_won,
                        "stats_dict": {
                            "rounds_won": s1,
                            "rounds_lost": s2,
                            "won": t1_won,
                            "total_rounds": s1 + s2,
                        },
                        "source": "match_lineups",
                    }
                )

        # Team 2 players
        t2_won = 1.0 if winner == "team2" else 0.0
        for i in range(1, 6):
            pname = _to_clean_str(row.get(f"team2_p{i}_name"))
            if pname:
                records.append(
                    {
                        "player_id": pname,
                        "date": ts,
                        "map_name": "",
                        "side": "Neutral",
                        "won": t2_won,
                        "stats_dict": {
                            "rounds_won": s2,
                            "rounds_lost": s1,
                            "won": t2_won,
                            "total_rounds": s1 + s2,
                        },
                        "source": "match_lineups",
                    }
                )

    logger.info("Extracted %d records from match lineups", len(records))
    return records, str(resolved_path)


# ---------------------------------------------------------------------------
# Normalization & Ingestion
# ---------------------------------------------------------------------------

def normalize_and_sort_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize records and sort strictly chronologically by date.

    Deduplicates exact matches for the same player, date, and map (retaining richer stats).
    """
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for r in records:
        pid = _to_clean_str(r.get("player_id"))
        dt = r.get("date")
        if not pid or not isinstance(dt, pd.Timestamp):
            continue

        m_name = _to_clean_str(r.get("map_name"))
        dedup_key = (pid.lower(), dt.isoformat(), m_name.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        cleaned.append(
            {
                "player_id": pid,
                "date": dt,
                "map_name": m_name,
                "side": _to_clean_str(r.get("side")),
                "won": float(r.get("won", 0.0)),
                "stats_dict": dict(r.get("stats_dict") or {}),
                "source": r.get("source", "unknown"),
            }
        )

    # Sort strictly chronologically by date, secondary sort by player_id
    cleaned.sort(key=lambda rec: (rec["date"], rec["player_id"]))
    return cleaned


def build_player_sequence_bank(
    records: Sequence[dict[str, Any]],
    *,
    max_history: int = 16,
    feature_dim: int = 18,
    include_map_side: bool = True,
    sources_loaded: Sequence[str] | None = None,
) -> tuple[PlayerHistoryTracker, dict[str, Any]]:
    """Ingest sorted chronological records into PlayerHistoryTracker and build summary.

    Args:
        records: Chronologically sorted list of record dictionaries.
        max_history: Rolling deque capacity per player (default 16).
        feature_dim: Feature dimension per token step (default 18).
        include_map_side: Whether token features include map_id & side_id.
        sources_loaded: Names of sources contributing records.

    Returns:
        tuple of (tracker, summary_statistics_dict).
    """
    tracker = PlayerHistoryTracker(
        max_history=max_history,
        feature_dim=feature_dim,
        include_map_side=include_map_side,
    )

    logger.info("Ingesting %d records into PlayerHistoryTracker...", len(records))
    for rec in records:
        tracker.add_player_map_record(
            player_id=rec["player_id"],
            date=rec["date"],
            map_name=rec["map_name"],
            side=rec["side"],
            stats_dict=rec["stats_dict"],
            won=rec["won"],
        )

    # Compute statistics
    total_distinct_players = tracker.num_players()
    history_lens = [len(h) for h in tracker.histories.values()]
    all_dates = [r.date for h in tracker.histories.values() for r in h]

    if total_distinct_players > 0 and all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        date_span_days = round((max_date - min_date).total_seconds() / 86400.0, 2)
        min_date_str = min_date.isoformat()
        max_date_str = max_date.isoformat()

        count_ge_1 = sum(1 for l in history_lens if l >= 1)
        count_ge_5 = sum(1 for l in history_lens if l >= 5)
        count_ge_10 = sum(1 for l in history_lens if l >= 10)
        count_ge_16 = sum(1 for l in history_lens if l >= 16)

        pct_ge_1 = round(count_ge_1 / total_distinct_players * 100.0, 2)
        pct_ge_5 = round(count_ge_5 / total_distinct_players * 100.0, 2)
        pct_ge_10 = round(count_ge_10 / total_distinct_players * 100.0, 2)
        pct_ge_16 = round(count_ge_16 / total_distinct_players * 100.0, 2)
    else:
        min_date_str = None
        max_date_str = None
        date_span_days = 0.0
        count_ge_1 = count_ge_5 = count_ge_10 = count_ge_16 = 0
        pct_ge_1 = pct_ge_5 = pct_ge_10 = pct_ge_16 = 0.0

    summary: dict[str, Any] = {
        "total_distinct_players": total_distinct_players,
        "total_records_ingested": len(records),
        "history_length_distribution": {
            "pct_ge_1": pct_ge_1,
            "pct_ge_5": pct_ge_5,
            "pct_ge_10": pct_ge_10,
            "pct_ge_16": pct_ge_16,
            "count_ge_1": count_ge_1,
            "count_ge_5": count_ge_5,
            "count_ge_10": count_ge_10,
            "count_ge_16": count_ge_16,
        },
        "date_range": {
            "min_date": min_date_str,
            "max_date": max_date_str,
            "total_days": date_span_days,
        },
        "sources_loaded": list(sources_loaded or []),
        "config": {
            "max_history": max_history,
            "feature_dim": feature_dim,
            "include_map_side": include_map_side,
        },
    }

    tracker.summary = summary
    return tracker, summary


def format_summary_report(summary: Mapping[str, Any]) -> str:
    """Format summary statistics dictionary into a clean CLI report."""
    dist = summary.get("history_length_distribution", {})
    drange = summary.get("date_range", {})
    sources = summary.get("sources_loaded", [])

    lines = [
        "=" * 68,
        "             PLAYER SEQUENCE BANK SUMMARY REPORT",
        "=" * 68,
        f"Total Distinct Players Tracked : {summary.get('total_distinct_players', 0):,}",
        f"Total Map Records Ingested     : {summary.get('total_records_ingested', 0):,}",
        f"Sources Loaded                 : {', '.join(sources) if sources else 'None'}",
        "-" * 68,
        "History Length Distribution (Max Rolling Deque: 16 maps):",
        f"  >=  1 maps : {dist.get('pct_ge_1', 0.0):6.2f}%  ({dist.get('count_ge_1', 0):,} players)",
        f"  >=  5 maps : {dist.get('pct_ge_5', 0.0):6.2f}%  ({dist.get('count_ge_5', 0):,} players)",
        f"  >= 10 maps : {dist.get('pct_ge_10', 0.0):6.2f}%  ({dist.get('count_ge_10', 0):,} players)",
        f"  >= 16 maps : {dist.get('pct_ge_16', 0.0):6.2f}%  ({dist.get('count_ge_16', 0):,} players)",
        "-" * 68,
        "Date Range of Sequence Bank:",
        f"  Earliest Record Date         : {drange.get('min_date', 'N/A')}",
        f"  Latest Record Date           : {drange.get('max_date', 'N/A')}",
        f"  Total Temporal Span          : {drange.get('total_days', 0.0):.1f} days",
        "=" * 68,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CS2 player sequence bank and serialize tracker state."
    )
    parser.add_argument(
        "--strandedpond-stats",
        type=Path,
        default=Path("data/raw/strandedpond/player_stats.csv"),
        help="Path to StrandedPond player_stats.csv",
    )
    parser.add_argument(
        "--strandedpond-matches",
        type=Path,
        default=Path("data/raw/strandedpond/matches.csv"),
        help="Path to StrandedPond matches.csv",
    )
    parser.add_argument(
        "--opencs2-maps",
        type=Path,
        default=Path("data/raw/opencs2/maps.csv"),
        help="Path to OpenCS2 maps.csv",
    )
    parser.add_argument(
        "--opencs2-scopes",
        type=Path,
        default=Path("data/exports/opencs2/player_scopes.csv"),
        help="Path to OpenCS2 player_scopes.csv",
    )
    parser.add_argument(
        "--match-lineups",
        type=Path,
        default=Path("data/processed/match_lineups.csv"),
        help="Path to processed match_lineups.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/04_feature/player_sequence_bank.pt"),
        help="Output path for serialized PlayerHistoryTracker state (.pt)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/04_feature/player_sequence_bank_summary.json"),
        help="Output path for summary statistics JSON",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=16,
        help="Maximum history steps per player rolling deque (default 16)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=18,
        help="Encoded token feature dimension per step (default 18)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    all_raw_records: list[dict[str, Any]] = []
    sources_loaded: list[str] = []

    # 1. StrandedPond
    sp_records, sp_source = load_strandedpond_records(
        stats_path=args.strandedpond_stats,
        matches_path=args.strandedpond_matches,
    )
    if sp_records:
        all_raw_records.extend(sp_records)
        sources_loaded.append(f"strandedpond ({len(sp_records)} records)")
    elif sp_source:
        logger.info("StrandedPond file present at %s but produced 0 records", sp_source)

    # 2. OpenCS2
    oc_records, oc_source = load_opencs2_records(
        maps_path=args.opencs2_maps,
        scopes_path=args.opencs2_scopes,
    )
    if oc_records:
        all_raw_records.extend(oc_records)
        sources_loaded.append(f"opencs2 ({len(oc_records)} records)")
    elif oc_source:
        logger.info("OpenCS2 file checked at %s but produced 0 records", oc_source)

    # 3. Match Lineups
    ml_records, ml_source = load_match_lineups_records(
        lineups_path=args.match_lineups,
    )
    if ml_records:
        all_raw_records.extend(ml_records)
        sources_loaded.append(f"match_lineups ({len(ml_records)} records)")
    elif ml_source:
        logger.info("Match lineups checked at %s but produced 0 records", ml_source)

    # Clean check if no data files were available
    if not all_raw_records:
        print("[WARNING] No player match records were found in the specified data sources.")
        print("Checked candidate paths:")
        print(f"  - StrandedPond stats : {args.strandedpond_stats}")
        print(f"  - OpenCS2 maps       : {args.opencs2_maps}")
        print(f"  - Match lineups      : {args.match_lineups}")
        print("Sequence bank generation skipped cleanly (0 records).")
        return 0

    # Normalize chronologically
    logger.info("Normalizing and sorting %d total raw records...", len(all_raw_records))
    normalized_records = normalize_and_sort_records(all_raw_records)

    # Ingest into PlayerHistoryTracker
    tracker, summary = build_player_sequence_bank(
        normalized_records,
        max_history=args.max_history,
        feature_dim=args.feature_dim,
        include_map_side=True,
        sources_loaded=sources_loaded,
    )

    # Save serialized tracker state (.pt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving serialized PlayerHistoryTracker state to %s", output_path)
    tracker.save(output_path, summary=summary)

    # Save summary statistics JSON
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving sequence bank summary statistics to %s", summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary report
    print(format_summary_report(summary))
    print(f"\n[OK] Serialized sequence bank successfully saved to: {output_path}")
    print(f"[OK] Summary statistics JSON successfully saved to: {summary_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
