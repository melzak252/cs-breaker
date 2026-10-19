#!/usr/bin/env python3
"""Run the NoDriver browser-based HLTV match results scraper.

Extracts completed match series from HLTV.org (results page) with
polite request pacing, randomized jitter, Cloudflare challenge handling,
and offset pagination.

Usage:
    python scripts/scrape_hltv.py [--pages 5] [--max-matches 500] [--all] [--delay 5.0] [--jitter 3.0] [--output data/raw/hltv/matches.csv]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import nodriver as uc
except ImportError:
    print("Error: 'nodriver' package not found. Install it with: pip install nodriver", file=sys.stderr)
    sys.exit(1)


JS_EXTRACT_MATCHES = """
(() => {
  const MONTHS_MAP = {
    january: '01', february: '02', march: '03', april: '04', may: '05', june: '06',
    july: '07', august: '08', september: '09', october: '10', november: '11', december: '12'
  };

  function parseDate(h) {
    if (!h) return null;
    const m = h.match(/(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d+)(?:st|nd|rd|th)\\s+(\\d{4})/i);
    if (!m) return null;
    return `${m[3]}-${MONTHS_MAP[m[1].toLowerCase()]}-${m[2].padStart(2, '0')}`;
  }

  const sublists = Array.from(document.querySelectorAll('.results-sublist'));
  const dates = sublists.map(s => parseDate(s.querySelector('.standard-headline')?.textContent?.trim()));
  const firstValidDate = dates.find(d => d !== null) || new Date().toISOString().slice(0, 10);

  const results = [];
  for (let i = 0; i < sublists.length; i++) {
    const sub = sublists[i];
    const dateStr = dates[i] || firstValidDate;
    const rows = sub.querySelectorAll('.result-con');
    
    for (const r of Array.from(rows)) {
      const a = r.querySelector('a.a-reset');
      const href = a?.getAttribute('href') || '';
      const idMatch = href.match(/\\/matches\\/(\\d+)\\//);
      if (!idMatch) continue;
      const matchId = idMatch[1];
      
      const team1Name = r.querySelector('.team1 .team')?.textContent?.trim() || '';
      const team2Name = r.querySelector('.team2 .team')?.textContent?.trim() || '';
      if (!team1Name || !team2Name) continue;
      
      const wonEl = r.querySelector('.score-won');
      const lostEl = r.querySelector('.score-lost');
      const scoreWon = wonEl ? parseInt(wonEl.textContent.trim(), 10) : null;
      const scoreLost = lostEl ? parseInt(lostEl.textContent.trim(), 10) : null;
      if (scoreWon === null || scoreLost === null) continue;
      
      const team1Won = r.querySelector('.team1 .team-won') !== null;
      const team2Won = r.querySelector('.team2 .team-won') !== null;
      
      let score1, score2, winner;
      if (team1Won) {
        score1 = scoreWon;
        score2 = scoreLost;
        winner = 'team1';
      } else if (team2Won) {
        score1 = scoreLost;
        score2 = scoreWon;
        winner = 'team2';
      } else {
        score1 = scoreWon;
        score2 = scoreLost;
        winner = scoreWon > scoreLost ? 'team1' : (scoreLost > scoreWon ? 'team2' : '');
      }
      
      const eventName = r.querySelector('.event-name')?.textContent?.trim() || '';
      const mapText = r.querySelector('.map-text')?.textContent?.trim() || 'bo3';
      
      results.push({
        match_id: matchId,
        date: `${dateStr} 12:00:00`,
        team1: team1Name,
        team2: team2Name,
        tournament: eventName,
        score1: String(score1),
        score2: String(score2),
        winner,
        format: mapText,
        match_url: `https://www.hltv.org${href}`
      });
    }
  }

  const paginationData = document.querySelector('.pagination-data')?.textContent?.trim() || '';
  const totalMatch = paginationData.match(/of\\s+([\\d,]+)/i);
  const totalAvailable = totalMatch ? parseInt(totalMatch[1].replace(/,/g, ''), 10) : null;

  return JSON.stringify({
    matches: results,
    totalAvailable: totalAvailable,
    hasNext: Boolean(document.querySelector('a.pagination-next'))
  });
})()
"""

FIELDNAMES = [
    "match_id",
    "date",
    "team1",
    "team2",
    "tournament",
    "score1",
    "score2",
    "winner",
    "format",
    "match_url",
]


async def run_scraper(
    pages: Optional[int],
    max_matches: Optional[int],
    all_pages: bool,
    delay: float,
    jitter: float,
    output_path: Path,
    headless: bool = False,
    start_offset: int = 0,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.parent / "manifest.json"

    existing_matches: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    mid = r.get("match_id", "").strip()
                    if mid:
                        seen_ids.add(mid)
                        existing_matches.append(r)
            print(f"Loaded {len(existing_matches)} existing matches from {output_path}", flush=True)
        except Exception as e:
            print(f"Warning reading existing {output_path}: {e}", flush=True)

    # Use off-screen window positioning so it doesn't interrupt desktop work
    # while naturally clearing Cloudflare challenge without detection
    browser_args = [
        "--no-sandbox",
        "--disable-gpu",
        "--window-size=1920,1080",
        "--window-position=-2400,-2400",
    ]
    if headless:
        browser_args.append("--headless=new")

    browser = await uc.start(
        headless=headless,
        browser_args=browser_args,
    )

    scraped_matches: List[Dict[str, str]] = list(existing_matches)
    offsets_scraped: List[int] = []
    offset = start_offset
    page_num = (start_offset // 100) + 1
    total_available = None

    try:
        # Initial navigation
        init_wait = delay + random.uniform(0.5, jitter) if jitter > 0 else delay
        page = await browser.get("https://www.hltv.org/results")
        await asyncio.sleep(init_wait)

        while True:
            if not all_pages and pages is not None and len(offsets_scraped) >= pages:
                break
            if max_matches is not None and len(scraped_matches) >= max_matches:
                break

            target_url = (
                "https://www.hltv.org/results"
                if offset == 0
                else f"https://www.hltv.org/results?offset={offset}"
            )

            # Polite human-like delay with random jitter
            current_delay = delay + random.uniform(0, jitter) if jitter > 0 else delay
            await asyncio.sleep(current_delay)
            page = await browser.get(target_url)

            # Wait for content to render or challenge to resolve (up to 30s)
            matches_count = 0
            for attempt in range(15):
                await asyncio.sleep(2.0)
                try:
                    eval_check = await page.evaluate(
                        'document.querySelectorAll(".result-con").length'
                    )
                    if isinstance(eval_check, int) and eval_check > 0:
                        matches_count = eval_check
                        break
                    # If challenged, wait for turnstile check to complete
                    title = await page.evaluate("document.title")
                    if "Just a moment" not in str(title):
                        # page loaded but 0 matches (actual end of results)
                        break
                except Exception:
                    pass

            raw_eval = await page.evaluate(JS_EXTRACT_MATCHES)
            data = json.loads(raw_eval) if isinstance(raw_eval, str) else {}
            matches = data.get("matches", [])
            total_available = data.get("totalAvailable")
            has_next = data.get("hasNext", False)

            new_count = 0
            for m in matches:
                mid = m["match_id"]
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    scraped_matches.append(m)
                    new_count += 1

            offsets_scraped.append(offset)
            total_info = f"/{total_available}" if total_available else ""
            print(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                f"Page {page_num} (offset {offset}): {len(matches)} matches on page, "
                f"+{new_count} new (Total: {len(scraped_matches)}{total_info})",
                flush=True,
            )

            # Atomic save so interrupts don't corrupt output
            temp_csv = output_path.with_suffix(".tmp")
            with open(temp_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(scraped_matches)
            temp_csv.replace(output_path)

            if not has_next or len(matches) == 0:
                print("No further pages found.", flush=True)
                break

            offset += 100
            page_num += 1

    finally:
        browser.stop()

    csv_bytes = output_path.read_bytes()
    sha256_hash = hashlib.sha256(csv_bytes).hexdigest()

    manifest = {
        "source": "hltv_scraped",
        "url": "https://www.hltv.org/results",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "pages_scraped": len(offsets_scraped),
        "offsets": offsets_scraped,
        "total_matches_available": total_available,
        "total_matches_scraped": len(scraped_matches),
        "file": {
            "path": str(output_path),
            "bytes": len(csv_bytes),
            "sha256": sha256_hash,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Completed: Saved {len(scraped_matches)} matches to {output_path} (manifest: {manifest_path})",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Number of 100-match pages to scrape",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Maximum total matches to collect before stopping",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all available pages until the end of results",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Base delay between page navigations in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=3.0,
        help="Random jitter added to delay in seconds (default: 3.0, so 5.0s - 8.0s interval)",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=None,
        help="Starting match offset (default: auto-detected from existing CSV)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force strict headless mode (note: Cloudflare Turnstile blocks pure headless)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/hltv/matches.csv"),
        help="Output CSV path (default: data/raw/hltv/matches.csv)",
    )
    args = parser.parse_args(argv)

    if not args.all and args.pages is None and args.max_matches is None:
        args.all = True

    # Auto-resume from existing row count if start_offset not explicitly set
    start_offset = args.start_offset
    if start_offset is None:
        if args.output.exists():
            try:
                with open(args.output, "r", encoding="utf-8") as f:
                    row_count = sum(1 for _ in f) - 1
                    # Round down to nearest 100
                    start_offset = max(0, (row_count // 100) * 100)
            except Exception:
                start_offset = 0
        else:
            start_offset = 0

    print(f"Starting scraper at offset {start_offset} (auto-resumed)")

    uc.loop().run_until_complete(
        run_scraper(
            pages=args.pages,
            max_matches=args.max_matches,
            all_pages=args.all,
            delay=args.delay,
            jitter=args.jitter,
            output_path=args.output,
            headless=args.headless,
            start_offset=start_offset,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
