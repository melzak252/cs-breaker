#!/usr/bin/env python3
"""Scrape player lineups for matches and map everything to native HLTV match ID.

Reads completed matches from data/raw/hltv/matches.csv and captures native player
IDs from individual HLTV match pages. Kaggle scrape-time roster snapshots are
not historical lineups and do not satisfy or skip a capture.

Outputs:
  data/processed/match_lineups.csv:
    hltv_match_id, date, tournament, format, team1, team2, score1, score2, winner,
    team1_p1_name, team1_p1_id, team1_p2_name, team1_p2_id, ... team2_p5_name, team2_p5_id,
    lineup_status, source, match_url
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import nodriver as uc
except ImportError:
    print("Error: 'nodriver' package not found. Install with: pip install nodriver", file=sys.stderr)
    sys.exit(1)


JS_MATCH_DETAILS = r"""
(() => {
    const t1 = document.querySelector('.team1-gradient .teamName')?.textContent?.trim() || '';
    const t2 = document.querySelector('.team2-gradient .teamName')?.textContent?.trim() || '';

    const boxes = Array.from(document.querySelectorAll('.lineups .lineup'));
    function extract(box) {
        if (!box) return [];
        const items = Array.from(box.querySelectorAll('.players .flagAlign, td.player .flagAlign'));
        return items.map(el => {
            const a = el.closest('a') || el.querySelector('a');
            const href = a ? a.getAttribute('href') : '';
            const m = href ? href.match(/\/player\/(\d+)\//) : null;
            const id = m ? m[1] : '';
            const name = el.querySelector('.text-ellipsis')?.textContent?.trim() || el.textContent?.trim() || '';
            return { id, name };
        }).filter(x => Boolean(x.name));
    }

    return JSON.stringify({
        t1,
        t2,
        t1_players: boxes.length >= 2 ? extract(boxes[0]) : [],
        t2_players: boxes.length >= 2 ? extract(boxes[1]) : []
    });
})()
"""

LINEUP_FIELDNAMES = [
    "hltv_match_id",
    "date",
    "tournament",
    "format",
    "team1",
    "team2",
    "score1",
    "score2",
    "winner",
    "team1_p1_name", "team1_p1_id",
    "team1_p2_name", "team1_p2_id",
    "team1_p3_name", "team1_p3_id",
    "team1_p4_name", "team1_p4_id",
    "team1_p5_name", "team1_p5_id",
    "team2_p1_name", "team2_p1_id",
    "team2_p2_name", "team2_p2_id",
    "team2_p3_name", "team2_p3_id",
    "team2_p4_name", "team2_p4_id",
    "team2_p5_name", "team2_p5_id",
    "team1_player_count", "team2_player_count",
    "lineup_status",
    "source",
    "match_url",
]


def complete_players(t1_players, t2_players):
    if any(len(players) != 5 for players in (t1_players, t2_players)):
        return False
    identities = [str(player.get("id", "")).strip() for player in t1_players + t2_players]
    return (len(set(identities)) == 10
            and all(identity.isascii() and identity.isdigit() and int(identity) > 0 for identity in identities)
            and all(str(player.get("name", "")).strip() for player in t1_players + t2_players))


def completed_record(row):
    if row.get("source") == "results_page" and row.get("lineup_status") == "forfeit":
        return True
    if row.get("source") != "hltv_live" or row.get("lineup_status") != "complete":
        return False
    if any(row.get(f"team{side}_player_count", "") not in ("", "5") for side in (1, 2)):
        return False
    players = [[{"id": row.get(f"team{side}_p{slot}_id", ""),
                 "name": row.get(f"team{side}_p{slot}_name", "")} for slot in range(1, 6)] for side in (1, 2)]
    return complete_players(*players)


async def scrape_missing_lineups(
    matches: List[Dict[str, str]],
    output_path: Path,
    delay: float = 3.5,
    jitter: float = 2.0,
    limit: Optional[int] = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")

    processed_ids: Set[str] = set()
    records: Dict[str, Dict[str, str]] = {}

    # If output file already exists, resume from it
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                mid = r.get("hltv_match_id", "").strip()
                if mid and r.get("source") != "kaggle":
                    records[mid] = r
                    if completed_record(r):
                        processed_ids.add(mid)
        print(f"Resumed from {len(records)} previously processed lineup records.", flush=True)

    # A forfeit needs no lineup; every other unverified record needs a capture.
    to_scrape: List[Dict[str, str]] = []
    for m in matches:
        mid = m["match_id"]
        if mid in processed_ids:
            continue

        # Check forfeit
        if m.get("format") == "def":
            row = create_lineup_row(m, [], [], status="forfeit", source="results_page")
            records[mid] = row
            processed_ids.add(mid)
            continue


        to_scrape.append(m)

    print(f"Retained {len(records)} capture/forfeit records. Need to scrape {len(to_scrape)} match pages.", flush=True)

    # Flush instant records to disk
    write_csv(records, temp_path, output_path)

    if not to_scrape or (limit is not None and limit <= 0):
        return

    if limit is not None:
        to_scrape = to_scrape[:limit]

    # Start browser for missing lineups
    browser = await uc.start(
        headless=False,
        browser_args=[
            "--no-sandbox",
            "--disable-gpu",
            "--window-size=1920,1080",
            "--window-position=-2400,-2400",
        ],
    )

    try:
        total = len(to_scrape)
        for idx, m in enumerate(to_scrape, start=1):
            mid = m["match_id"]
            url = m["match_url"]
            print(f"[{idx}/{total}] Fetching lineups for {mid}: {m['team1']} vs {m['team2']} ...", flush=True)

            wait_sec = delay + random.uniform(0, jitter)
            await asyncio.sleep(wait_sec)

            page = await browser.get(url)

            # Wait for Challenge or content
            for _ in range(12):
                await asyncio.sleep(1.5)
                title = await page.evaluate("document.title")
                if "Just a moment" not in str(title):
                    break

            raw = await page.evaluate(JS_MATCH_DETAILS)
            data = json.loads(raw) if isinstance(raw, str) else {}

            t1_p = data.get("t1_players", [])
            t2_p = data.get("t2_players", [])
            expected = (m["team1"].strip().casefold(), m["team2"].strip().casefold())
            captured = (str(data.get("t1", "")).strip().casefold(), str(data.get("t2", "")).strip().casefold())
            if captured == expected[::-1]:
                t1_p, t2_p = t2_p, t1_p
                captured = expected
            status = ("team_mismatch" if captured != expected else
                      "complete" if complete_players(t1_p, t2_p) else
                      "partial" if (t1_p or t2_p) else "missing")
            row = create_lineup_row(m, t1_p, t2_p, status=status, source="hltv_live")
            records[mid] = row
            processed_ids.add(mid)

            if idx % 10 == 0 or idx == total:
                write_csv(records, temp_path, output_path)
                print(f"--> Saved {len(records)} mapped match lineups to {output_path}", flush=True)

    finally:
        browser.stop()
        write_csv(records, temp_path, output_path)


def create_lineup_row(
    m: Dict[str, str],
    t1_p: List[Dict[str, str]],
    t2_p: List[Dict[str, str]],
    status: str,
    source: str,
) -> Dict[str, str]:
    if source == "hltv_live" and status == "complete" and not complete_players(t1_p, t2_p):
        status = "partial"
    row = {
        "hltv_match_id": m.get("match_id", ""),
        "date": m.get("date", ""),
        "tournament": m.get("tournament", ""),
        "format": m.get("format", ""),
        "team1": m.get("team1", ""),
        "team2": m.get("team2", ""),
        "score1": m.get("score1", ""),
        "score2": m.get("score2", ""),
        "winner": m.get("winner", ""),
        "lineup_status": status,
        "source": source,
        "match_url": m.get("match_url", ""),
        "team1_player_count": str(len(t1_p)),
        "team2_player_count": str(len(t2_p)),
    }
    for k in range(5):
        p1 = t1_p[k] if k < len(t1_p) else {"name": "", "id": ""}
        p2 = t2_p[k] if k < len(t2_p) else {"name": "", "id": ""}
        row[f"team1_p{k+1}_name"] = p1.get("name", "")
        row[f"team1_p{k+1}_id"] = p1.get("id", "")
        row[f"team2_p{k+1}_name"] = p2.get("name", "")
        row[f"team2_p{k+1}_id"] = p2.get("id", "")
    return row


def write_csv(records: Dict[str, Dict[str, str]], temp_path: Path, final_path: Path):
    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LINEUP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records.values())
    temp_path.replace(final_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=Path("data/raw/hltv/matches.csv"),
        help="Input scraped matches CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/match_lineups.csv"),
        help="Output mapped lineups CSV path",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default="2023-09-27",
        help="Filter matches on or after date (default: CS2 release 2023-09-27)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pages to scrape live",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.5,
        help="Base polite delay between match pages (default: 3.5)",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=2.0,
        help="Delay jitter (default: 2.0)",
    )
    args = parser.parse_args(argv)

    print(f"Loading matches from {args.matches_csv} (min_date: {args.min_date})...")
    matches: List[Dict[str, str]] = []
    with open(args.matches_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not args.min_date or r.get("date", "") >= args.min_date:
                matches.append(r)

    print(f"Total candidate matches: {len(matches)}")
    uc.loop().run_until_complete(
        scrape_missing_lineups(
            matches=matches,
            output_path=args.output,
            delay=args.delay,
            jitter=args.jitter,
            limit=args.limit,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
