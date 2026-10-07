"""Cheap checks on the data files. They run before any paid eval does."""
from src.dataset import load_golden, validate
from src.prompts import PLACEHOLDER, list_versions, load_prompt


def test_golden_dataset_is_valid():
    cases = load_golden()
    assert len(cases) >= 50
    assert validate(cases) == []


def test_every_prompt_version_loads():
    versions = list_versions()
    assert versions
    for version in versions:
        assert PLACEHOLDER in load_prompt(version)["template"]