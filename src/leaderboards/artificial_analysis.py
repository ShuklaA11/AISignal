"""Artificial Analysis provider.

Scrapes the embedded payload at https://artificialanalysis.ai/ and emits
three Snapshots: intelligence (AA's quality index), output_speed
(tokens/sec, median), and price (blended $/Mtok). AA does not expose a
free public API, so the data lives inside Next.js RSC chunks.

This parser is the brittle bit of the provider. It looks for JSON-object
literals carrying the schema's distinctive markers (slug, name,
intelligence_index, median_output_speed, price, creator) and treats each
match as one model. When AA renames a field or restructures their app,
this is the place to update.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable

import httpx

from src.leaderboards.base import LeaderboardProvider, Ranking, Snapshot

logger = logging.getLogger(__name__)


DEFAULT_URL = "https://artificialanalysis.ai/"

PROVIDER = "artificial_analysis"
METRIC_INTELLIGENCE = "intelligence"
METRIC_SPEED = "output_speed"
METRIC_PRICE = "price"


@dataclass(frozen=True)
class _ModelRow:
    slug: str
    name: str
    organization: str | None
    intelligence: float | None
    speed: float | None
    price: float | None


# AA's embedded payload escapes JSON twice: the page-level RSC string carries
# JSON whose quotes are written as \". So in the served HTML, every JSON
# delimiter is a backslash + quote (regex: \\\")  — kept in a constant so the
# escape level is visible in one place.
_Q = r'\\"'

# Anchor: every model record ends with a hosts_url that contains the slug.
# That gives us a reliable per-model boundary.
_HOSTS_URL_RE = re.compile(
    _Q + r"hosts_url" + _Q + r":" + _Q + r"/models/(?P<slug>[a-z0-9-]+)/providers"
)

# Within the ~3000 chars before a hosts_url, the model's own object has these
# fields. We pull them with individual regexes so a single missing field
# doesn't drop the whole record.
_NAME_RE = re.compile(_Q + r"name" + _Q + r":" + _Q + r"(?P<name>[^\"\\]{1,120}?)" + _Q)
_INTEL_RE = re.compile(_Q + r"intelligence_index" + _Q + r":(?P<v>-?[0-9.]+|null)")
_SPEED_RE = re.compile(
    _Q + r"timescaleData" + _Q + r":\{[^}]{0,400}?"
    + _Q + r"median_output_speed" + _Q + r":(?P<v>-?[0-9.]+|null)"
)
_PRICE_RE = re.compile(
    _Q + r"price_1m_blended_0_3_1" + _Q + r":(?P<v>-?[0-9.]+|null)"
)
_CREATOR_RE = re.compile(
    _Q + r"model_creators" + _Q + r":\{[^}]{0,400}?"
    + _Q + r"name" + _Q + r":" + _Q + r"(?P<v>[^\"\\]{1,80}?)" + _Q
)


def _safe_float(v: str | None) -> float | None:
    if v is None or v == "null":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_models(html: str) -> list[_ModelRow]:
    """Extract model rows from the page HTML.

    Anchor on each hosts_url match (one per model). For each, look at the
    preceding ~6000-char window for the four scalar fields and the creator
    name. Last-write-wins per slug, but rows with non-null intelligence
    take priority so we don't overwrite with the zero-rows that appear in
    summary widgets elsewhere on the page.
    """
    seen: dict[str, _ModelRow] = {}
    for m in _HOSTS_URL_RE.finditer(html):
        slug = m.group("slug")
        # Window ends just before "hosts_url" — the model fields all precede it.
        end = m.start()
        start = max(0, end - 6000)
        window = html[start:end]

        # The model's own display name is the LAST "name" field before creator
        # in the window. Iterate and take the one closest to the end.
        creator_m = list(_CREATOR_RE.finditer(window))
        creator = creator_m[-1].group("v") if creator_m else None
        creator_pos = creator_m[-1].start() if creator_m else len(window)
        name_candidates = [
            mm.group("name") for mm in _NAME_RE.finditer(window) if mm.end() <= creator_pos
        ]
        name = name_candidates[-1] if name_candidates else slug

        intel_m = list(_INTEL_RE.finditer(window))
        speed_m = list(_SPEED_RE.finditer(window))
        price_m = list(_PRICE_RE.finditer(window))

        row = _ModelRow(
            slug=slug,
            name=name,
            organization=creator,
            intelligence=_safe_float(intel_m[-1].group("v") if intel_m else None),
            speed=_safe_float(speed_m[-1].group("v") if speed_m else None),
            price=_safe_float(price_m[-1].group("v") if price_m else None),
        )

        prev = seen.get(slug)
        if prev is None or (prev.intelligence is None and row.intelligence is not None):
            seen[slug] = row
    return list(seen.values())


def _rank_by(
    rows: list[_ModelRow],
    score_fn,
    ascending: bool = False,
    drop_if_zero: bool = True,
) -> list[Ranking]:
    """Build a Ranking list from rows. score_fn returns score or None.

    ascending=True orders smallest-first (used for price — cheaper is better).
    drop_if_zero excludes rows with None or 0 scores (price=0 means unlisted).
    """
    scored = []
    for r in rows:
        s = score_fn(r)
        if s is None:
            continue
        if drop_if_zero and s == 0:
            continue
        scored.append((r, s))
    scored.sort(key=lambda x: x[1], reverse=not ascending)
    return [
        Ranking(
            model=row.name,
            rank=i + 1,
            score=round(score, 4),
            organization=row.organization,
            extras={"slug": row.slug},
        )
        for i, (row, score) in enumerate(scored)
    ]


def build_snapshots(rows: list[_ModelRow]) -> list[Snapshot]:
    """Turn a list of model rows into one Snapshot per metric."""
    return [
        Snapshot(
            provider=PROVIDER,
            metric=METRIC_INTELLIGENCE,
            rankings=_rank_by(rows, lambda r: r.intelligence),
            source_url=DEFAULT_URL,
            notes="Artificial Analysis Intelligence Index (higher = better)",
        ),
        Snapshot(
            provider=PROVIDER,
            metric=METRIC_SPEED,
            rankings=_rank_by(rows, lambda r: r.speed),
            source_url=DEFAULT_URL,
            notes="Median output tokens/sec (higher = faster)",
        ),
        Snapshot(
            provider=PROVIDER,
            metric=METRIC_PRICE,
            rankings=_rank_by(rows, lambda r: r.price, ascending=True),
            source_url=DEFAULT_URL,
            notes="Blended price $/Mtok (lower = cheaper)",
        ),
    ]


class ArtificialAnalysisProvider(LeaderboardProvider):
    """Provider for artificialanalysis.ai's intelligence / speed / price boards."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return PROVIDER

    async def fetch_snapshot(self) -> Iterable[Snapshot]:
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            resp = await client.get(self.url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; AISignalBot/1.0)",
            })
            resp.raise_for_status()
            html = resp.text

        rows = parse_models(html)
        if not rows:
            logger.warning(
                f"[{PROVIDER}] parser found 0 models — AA page format may have changed"
            )
            return []
        logger.info(f"[{PROVIDER}] parsed {len(rows)} model rows")
        return build_snapshots(rows)
