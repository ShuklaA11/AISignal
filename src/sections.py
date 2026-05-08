"""Section taxonomy for the newsletter.

Sections are organized by reading intent, not topic. The taxonomy is
defined in code so that the rest of the codebase can validate values
without coupling to a specific config or database state.

Section weight profiles (per-role weighting across sections) live in
config (see config.yaml.example) so deployments can tune defaults
without code changes.
"""

from __future__ import annotations

from typing import Final

SECTION_RESEARCH: Final[str] = "research"
SECTION_RELEASES: Final[str] = "releases"
SECTION_BUILDER: Final[str] = "builder"
SECTION_INDUSTRY: Final[str] = "industry"
SECTION_LEARN: Final[str] = "learn"

ALL_SECTIONS: Final[tuple[str, ...]] = (
    SECTION_RESEARCH,
    SECTION_RELEASES,
    SECTION_BUILDER,
    SECTION_INDUSTRY,
    SECTION_LEARN,
)

SECTION_LABELS: Final[dict[str, str]] = {
    SECTION_RESEARCH: "Research",
    SECTION_RELEASES: "Releases",
    SECTION_BUILDER: "Builder",
    SECTION_INDUSTRY: "Industry",
    SECTION_LEARN: "Learn",
}

SECTION_DESCRIPTIONS: Final[dict[str, str]] = {
    SECTION_RESEARCH: "Deep-dives, papers, and mechanistic work",
    SECTION_RELEASES: "Lab releases, model launches, and frontier announcements",
    SECTION_BUILDER: "Practitioner blogs, tools, and tool releases",
    SECTION_INDUSTRY: "Business, funding, products, and strategy",
    SECTION_LEARN: "Explainers, lectures, and tutorials",
}


def is_valid_section(value: str | None) -> bool:
    """Return True if the value is a recognized section identifier."""
    return value in ALL_SECTIONS
