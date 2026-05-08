"""Tests for the section taxonomy."""
import pytest

from src.sections import (
    ALL_SECTIONS,
    SECTION_BUILDER,
    SECTION_DESCRIPTIONS,
    SECTION_INDUSTRY,
    SECTION_LABELS,
    SECTION_LEARN,
    SECTION_RELEASES,
    SECTION_RESEARCH,
    is_valid_section,
)


@pytest.mark.unit
def test_all_sections_has_five_entries() -> None:
    assert len(ALL_SECTIONS) == 5


@pytest.mark.unit
def test_all_sections_contains_each_constant() -> None:
    assert set(ALL_SECTIONS) == {
        SECTION_RESEARCH,
        SECTION_RELEASES,
        SECTION_BUILDER,
        SECTION_INDUSTRY,
        SECTION_LEARN,
    }


@pytest.mark.unit
def test_section_labels_cover_all_sections() -> None:
    assert set(SECTION_LABELS.keys()) == set(ALL_SECTIONS)


@pytest.mark.unit
def test_section_descriptions_cover_all_sections() -> None:
    assert set(SECTION_DESCRIPTIONS.keys()) == set(ALL_SECTIONS)


@pytest.mark.unit
@pytest.mark.parametrize("value", list(ALL_SECTIONS))
def test_is_valid_section_accepts_known_values(value: str) -> None:
    assert is_valid_section(value) is True


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "Research", "RESEARCH", "frontier", "discourse", None])
def test_is_valid_section_rejects_unknown_values(value: str | None) -> None:
    assert is_valid_section(value) is False


@pytest.mark.unit
def test_section_constants_are_lowercase_strings() -> None:
    for section in ALL_SECTIONS:
        assert section == section.lower()
        assert isinstance(section, str)
