import pytest
from clusterdrift.signals.hashing import compute_quality_record_sha256
from clusterdrift.signals.result import QualityResult

def test_target_signed_unclamped():
    # Example where shifted clustering actually improves over clean
    ari_clean = 0.50
    ari_condition = 0.70
    delta_ari = ari_clean - ari_condition
    assert delta_ari == pytest.approx(-0.20)  # Signed negative preserved, not clamped at 0.0

    rec_for_hash = {
        "dataset_id": "iris",
        "outer_fold": 0,
        "condition": "location_mild",
        "method": "fcm_adaptive",
        "seed": 1,
        "quality_target": "delta_ari",
        "ari_clean": ari_clean,
        "ari_condition": ari_condition,
        "delta_ari": delta_ari,
        "nmi_condition": 0.75,
        "ami_condition": 0.74,
        "n_evaluation_rows": 100,
    }
    q_sha = compute_quality_record_sha256(rec_for_hash)

    q = QualityResult(
        dataset_id="iris",
        outer_fold=0,
        condition="location_mild",
        method="fcm_adaptive",
        seed=1,
        quality_target="delta_ari",
        ari_clean=ari_clean,
        ari_condition=ari_condition,
        delta_ari=delta_ari,
        nmi_condition=0.75,
        ami_condition=0.74,
        n_evaluation_rows=100,
        quality_record_sha256=q_sha,
    )
    d = q.to_dict()
    assert d["delta_ari"] == pytest.approx(-0.20)