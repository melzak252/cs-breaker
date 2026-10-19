"""Official Liquipedia API client with paced, disk-cached requests."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pipeline.contracts import MatchFixture, Tournament


_LOG = logging.getLogger(__name__)
_API_URL = "https://liquipedia.net/counterstrike/api.php"
_USER_AGENT = (
    "CSBreakerResearch/1.0 "
    "(https://github.com/melzak/cs-breaker; contact: research@example.com)"
)
_MIN_REQUEST_INTERVAL = 2.0


class LiquipediaClient:
    """Fetch Counter-Strike tournament data from Liquipedia's MediaWiki API.

    Responses are cached by their complete API URL (including parameters) under
    ``data/cache/liquipedia``. Cached data is usable without network access.
    If a query cannot be completed, the public getters return deterministic
    sample records so offline development does not depend on the API.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/liquipedia",
        timeout: float = 20.0,
        user_agent: str = _USER_AGENT,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.user_agent = user_agent
        self._request_lock = Lock()
        self._last_request_at = 0.0

    def get_tournaments(
        self, year: int = 2024, tier: Optional[str] = None
    ) -> List[Tournament]:
        """Return tournaments for a year, falling back to deterministic samples."""
        where = (
            f"Tournaments.Startdate>='{year}-01-01',"
            f"Tournaments.Startdate<='{year}-12-31'"
        )
        params: Dict[str, str] = {
            "action": "cargoquery",
            "tables": "Tournaments",
            "fields": ",".join(
                (
                    "Tournaments.Name",
                    "Tournaments.Tier",
                    "Tournaments.Startdate",
                    "Tournaments.Enddate",
                    "Tournaments.Location",
                )
            ),
            "where": where,
            "limit": "max",
            "format": "json",
        }
        try:
            rows = self._cargo_query(params)
            tournaments = [item for row in rows if (item := self._tournament_from_row(row))]
        except (OSError, ValueError, RuntimeError) as exc:
            _LOG.warning("Liquipedia tournament query failed; using offline fixtures: %s", exc)
            tournaments = []
        if not tournaments:
            tournaments = self._offline_tournaments(year)
        if tier:
            tournaments = [item for item in tournaments if item.tier.casefold() == tier.casefold()]
        return tournaments

    def get_tournament_matches(self, tournament_name: str) -> List[MatchFixture]:
        """Return tournament matches, or sample fixtures when the API is unavailable."""
        params = {
            "action": "cargoquery",
            "tables": "Match",
            "fields": ",".join(
                (
                    "Match.MatchId",
                    "Match.DateTime_UTC",
                    "Match.Team1",
                    "Match.Team2",
                    "Match.BestOf",
                    "Match.Team1Score",
                    "Match.Team2Score",
                    "Match.Winner",
                    "Match.Tournament",
                )
            ),
            "where": f'Match.Tournament="{self._cargo_escape(tournament_name)}"',
            "limit": "max",
            "format": "json",
        }
        try:
            rows = self._cargo_query(params)
            fixtures = [
                item
                for index, row in enumerate(rows)
                if (item := self._fixture_from_row(row, tournament_name, index))
            ]
        except (OSError, ValueError, RuntimeError) as exc:
            _LOG.warning(
                "Liquipedia match query failed for %r; using offline fixtures: %s",
                tournament_name,
                exc,
            )
            fixtures = []
        return fixtures or self._offline_matches(tournament_name)

    @staticmethod
    def save_fixtures_manifest(
        fixtures: List[MatchFixture], output_path: str
    ) -> None:
        """Write fixtures as a UTF-8 JSON array using contract serialization."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump([fixture.to_dict() for fixture in fixtures], stream, indent=2)
            stream.write("\n")

    def _cargo_query(self, params: Dict[str, str]) -> List[Dict[str, Any]]:
        payload = self._request_json(params)
        query = payload.get("cargoquery", []) if isinstance(payload, dict) else []
        if not isinstance(query, list):
            raise ValueError("Liquipedia API response has invalid cargoquery data")
        return [entry["title"] for entry in query if isinstance(entry, dict) and isinstance(entry.get("title"), dict)]

    def _request_json(self, params: Dict[str, str]) -> Dict[str, Any]:
        url = f"{_API_URL}?{urlencode(params)}"
        cache_path = self._cache_path(url)
        if cache_path.is_file():
            try:
                with cache_path.open("r", encoding="utf-8") as stream:
                    cached = json.load(stream)
                if isinstance(cached, dict):
                    return cached
            except (OSError, json.JSONDecodeError):
                _LOG.warning("Ignoring corrupt Liquipedia cache entry %s", cache_path)

        for attempt in range(3):
            self._pace_request()
            request = Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = self._decode_body(response.read(), response.headers.get("Content-Encoding", ""))
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Liquipedia API returned non-object JSON")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = cache_path.with_suffix(".tmp")
                with temporary_path.open("w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False)
                os.replace(temporary_path, cache_path)
                return payload
            except HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = max(float(retry_after), 2.0 * (2**attempt)) if retry_after else 2.0 * (2**attempt)
                    except ValueError:
                        delay = 2.0 * (2**attempt)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Liquipedia API returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(f"Liquipedia API request failed: {exc}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Liquipedia API returned invalid JSON") from exc
        raise RuntimeError("Liquipedia API remained throttled after retries")

    def _pace_request(self) -> None:
        # Keep pacing and request-start timestamp atomic across concurrent callers.
        with self._request_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < _MIN_REQUEST_INTERVAL:
                time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
            self._last_request_at = time.monotonic()

    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _decode_body(body: bytes, encoding: str) -> bytes:
        encoding = encoding.lower().strip()
        if encoding == "gzip":
            return gzip.decompress(body)
        if encoding == "deflate":
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
        return body

    @staticmethod
    def _cargo_escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _normalized_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {re.sub(r"[^a-z0-9]", "", str(key).split(".")[-1].lower()): value for key, value in row.items()}

    def _tournament_from_row(self, row: Dict[str, Any]) -> Optional[Tournament]:
        fields = self._normalized_row(row)
        name = self._text(fields, "name")
        if not name:
            return None
        tier = self._text(fields, "tier") or "Unknown"
        start = self._date(fields.get("startdate"))
        end = self._date(fields.get("enddate"))
        location = self._text(fields, "location")
        return Tournament(
            tournament_id=self._slug(name),
            name=name,
            tier=tier,
            start_date=start,
            end_date=end,
            location=location,
            liquipedia_url=f"https://liquipedia.net/counterstrike/{quote(name.replace(' ', '_'), safe='/()_')}",
        )

    def _fixture_from_row(
        self, row: Dict[str, Any], tournament_name: str, index: int
    ) -> Optional[MatchFixture]:
        fields = self._normalized_row(row)
        team1 = self._text(fields, "team1")
        team2 = self._text(fields, "team2")
        if not team1 or not team2:
            return None
        timestamp = self._datetime(fields.get("datetimeutc") or fields.get("date") or fields.get("matchday"))
        score1 = self._integer(fields.get("team1score"))
        score2 = self._integer(fields.get("team2score"))
        winner_value = self._text(fields, "winner")
        if winner_value and winner_value.casefold() in {team1.casefold(), "1", "team1"}:
            winner = "team1"
        elif winner_value and winner_value.casefold() in {team2.casefold(), "2", "team2"}:
            winner = "team2"
        elif score1 is not None and score2 is not None and score1 != score2:
            winner = "team1" if score1 > score2 else "team2"
        else:
            winner = None
        best_of = self._integer(fields.get("bestof"))
        return MatchFixture(
            match_id=self._text(fields, "matchid") or f"{self._slug(tournament_name)}-{index + 1}",
            tournament_name=tournament_name,
            timestamp=timestamp,
            team1=team1,
            team2=team2,
            format=f"bo{best_of}" if best_of in (1, 3, 5) else "bo3",
            score1=score1,
            score2=score2,
            winner=winner,
            demo_url=self._text(fields, "demourl"),
            maps=self._maps(fields),
        )

    @staticmethod
    def _text(fields: Dict[str, Any], key: str) -> Optional[str]:
        value = fields.get(key)
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @staticmethod
    def _date(value: Any) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        return str(value).strip()[:10]

    @staticmethod
    def _datetime(value: Any) -> datetime:
        if value:
            text = str(value).strip().replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(text)
                return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            except ValueError:
                pass
        return datetime(2024, 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _integer(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None and str(value).strip() else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _maps(cls, fields: Dict[str, Any]) -> List[str]:
        maps: List[str] = []
        for key in sorted(fields):
            if re.fullmatch(r"map\d+", key) and fields[key]:
                maps.append(str(fields[key]))
        return maps

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
        return slug or "unknown"

    @classmethod
    def _offline_tournaments(cls, year: int) -> List[Tournament]:
        return [
            Tournament(
                tournament_id=f"offline-{year}-major",
                name=f"CS2 Major Championship {year}",
                tier="S-Tier",
                start_date=f"{year}-06-01",
                end_date=f"{year}-06-15",
                location="Offline sample",
                liquipedia_url="https://liquipedia.net/counterstrike/",
            ),
            Tournament(
                tournament_id=f"offline-{year}-regional",
                name=f"CS2 Regional Invitational {year}",
                tier="A-Tier",
                start_date=f"{year}-09-01",
                end_date=f"{year}-09-07",
                location="Offline sample",
                liquipedia_url="https://liquipedia.net/counterstrike/",
            ),
        ]

    @staticmethod
    def _offline_matches(tournament_name: str) -> List[MatchFixture]:
        return [
            MatchFixture(
                match_id=f"offline-{LiquipediaClient._slug(tournament_name)}-1",
                tournament_name=tournament_name,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                team1="Team Alpha",
                team2="Team Bravo",
                format="bo3",
                maps=["Mirage", "Inferno", "Nuke"],
            )
        ]
