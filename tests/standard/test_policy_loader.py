import pytest

from standard.policy.loader import PolicyIntegrityError, PolicyLoader

POLICY_BODY = """\
policy_version: "food_integrity_1.0.0"
mask:
  min_iou: 0.995
  max_area_delta: 0.01
high_frequency:
  base_threshold: 0.04
  contrast_weight: 0.35
  sharpness_weight: 0.50
food:
  max_saturation_gain: 0.05
  max_luminance_gain: 0.10
layout:
  max_scale_delta_per_object: 0.03
  max_relative_scale_ratio_change: 0.02
  preserve_object_area_rank: true
  max_occlusion_change: 0.15
  preserve_front_back_order: true
severity:
  mask:
    min_iou: {{reject_multiplier: null}}
    max_area_delta: {{reject_multiplier: null}}
  high_frequency:
    base_threshold: {{reject_multiplier: null}}
  food:
    max_saturation_gain: {{reject_multiplier: null}}
    max_luminance_gain: {{reject_multiplier: null}}
  layout:
    max_scale_delta_per_object: {{reject_multiplier: null}}
    max_relative_scale_ratio_change: {{reject_multiplier: null}}
    max_occlusion_change: {{reject_multiplier: {reject_multiplier}}}
"""


def _write_policy(path, reject_multiplier="null"):
    path.write_text(POLICY_BODY.format(reject_multiplier=reject_multiplier), encoding="utf-8")


def test_load_without_registration_is_refused(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    loader = PolicyLoader(tmp_path / "policy.lock.json")

    with pytest.raises(PolicyIntegrityError, match="not registered"):
        loader.load(policy_path)


def test_register_then_load_succeeds_with_matching_hash(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    loader = PolicyLoader(tmp_path / "policy.lock.json")

    registered = loader.register(policy_path)
    loaded = loader.load(policy_path)

    assert loaded.version == "food_integrity_1.0.0"
    assert loaded.hash == registered.hash
    assert loaded.hash.startswith("sha256:")
    assert loaded.threshold("mask", "min_iou") == 0.995


def test_editing_a_registered_policy_in_place_is_rejected_on_load(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    loader = PolicyLoader(tmp_path / "policy.lock.json")
    loader.register(policy_path)

    _write_policy(policy_path, reject_multiplier="1.5")  # tamper with a locked version's content

    with pytest.raises(PolicyIntegrityError, match="content changed"):
        loader.load(policy_path)


def test_reregistering_same_content_is_idempotent(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    loader = PolicyLoader(tmp_path / "policy.lock.json")

    first = loader.register(policy_path)
    second = loader.register(policy_path)

    assert first.hash == second.hash


def test_reregistering_changed_content_under_same_version_is_refused(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    loader = PolicyLoader(tmp_path / "policy.lock.json")
    loader.register(policy_path)

    _write_policy(policy_path, reject_multiplier="1.5")  # same policy_version, different content

    with pytest.raises(PolicyIntegrityError, match="refusing to re-register"):
        loader.register(policy_path)


def test_hash_changes_when_content_changes(tmp_path):
    import hashlib

    policy_a = tmp_path / "a.yaml"
    policy_b = tmp_path / "b.yaml"
    _write_policy(policy_a, reject_multiplier="null")
    _write_policy(policy_b, reject_multiplier="2.0")
    loader = PolicyLoader(tmp_path / "policy.lock.json")

    hash_a = loader.register(policy_a).hash
    expected_hash_b = f"sha256:{hashlib.sha256(policy_b.read_bytes()).hexdigest()}"

    assert hash_a != expected_hash_b


def test_reason_lookup_helpers(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path, reject_multiplier="1.5")
    loader = PolicyLoader(tmp_path / "policy.lock.json")
    policy = loader.register(policy_path)

    assert policy.reject_multiplier("layout", "max_occlusion_change") == 1.5
    assert policy.reject_multiplier("mask", "min_iou") is None
