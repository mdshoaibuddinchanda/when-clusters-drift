"""Test that the dataset registry contains exactly 40 real and 8 synthetic datasets."""

import pytest
from clusterdrift.data.registry import (
    get_dataset_spec,
    get_synthetic_spec,
    list_controlled_real,
    list_datasets,
    list_natural_shift,
    list_synthetic,
)


def test_real_dataset_counts():
    all_real = list_datasets()
    controlled = list_controlled_real()
    natural = list_natural_shift()

    assert len(all_real) == 40, f"Expected 40 real datasets, got {len(all_real)}"
    assert len(controlled) == 30, f"Expected 30 controlled real datasets, got {len(controlled)}"
    assert len(natural) == 10, f"Expected 10 natural shift datasets, got {len(natural)}"
    assert set(controlled).isdisjoint(set(natural)), "Controlled and natural datasets must not overlap."


def test_synthetic_family_counts():
    syn = list_synthetic()
    assert len(syn) == 8, f"Expected 8 synthetic families, got {len(syn)}"


def test_unique_identifiers():
    all_real = list_datasets()
    assert len(all_real) == len(set(all_real)), "Duplicate real dataset slugs detected."
    syn = list_synthetic()
    assert len(syn) == len(set(syn)), "Duplicate synthetic slugs detected."


def test_required_metadata_fields():
    for slug in list_datasets():
        spec = get_dataset_spec(slug)
        assert spec.id == slug
        assert spec.dataset_group in ["controlled_real", "natural_shift"]
        assert spec.source_provider in ["sklearn", "openml", "whyshift", "tableshift", "uci", "direct"]
        assert spec.download_mode in ["auto", "auth_required", "manual_license_acceptance", "unavailable"]
        assert len(spec.license) > 0
        assert spec.expected_min_rows > 0
        assert spec.expected_min_features > 0

    for syn_slug in list_synthetic():
        spec = get_synthetic_spec(syn_slug)
        assert spec.family_id == syn_slug
        assert spec.generator_seed > 0
        assert spec.K >= 2
        assert spec.n >= 100
        assert spec.d >= 2
