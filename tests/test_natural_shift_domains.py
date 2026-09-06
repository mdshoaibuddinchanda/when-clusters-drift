"""Test that natural shift datasets preserve distinct domain partitions and metadata."""

from clusterdrift.data.registry import get_dataset_spec, list_natural_shift


def test_natural_shift_domain_separation():
    natural_slugs = list_natural_shift()
    assert len(natural_slugs) == 9

    for slug in natural_slugs:
        spec = get_dataset_spec(slug)
        assert spec.dataset_group == "natural_shift"
        assert spec.domain_column is not None, f"Natural shift dataset {slug} missing domain_column specification."
        assert "domain_metadata" in spec.__dict__, f"Missing domain_metadata in {slug}"
        assert len(spec.domain_metadata) > 0, f"Empty domain_metadata in {slug}"

        if slug.startswith("whyshift_acs"):
            domains = spec.domain_metadata.get("domains", [])
            assert len(domains) >= 2, f"WhyShift ACS dataset {slug} must define multiple state domains."
            assert "CA" in domains
            assert spec.domain_metadata.get("type") == "spatial"
