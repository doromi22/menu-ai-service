from standard.shadow.profiles import CONTACT_SHADOW_PROFILES, resolve_contact_profile


def test_known_profile_names_resolve_to_their_own_params():
    assert resolve_contact_profile("heavy") == CONTACT_SHADOW_PROFILES["heavy"]
    assert resolve_contact_profile("glass") == CONTACT_SHADOW_PROFILES["glass"]


def test_unknown_profile_name_falls_back_to_default():
    assert resolve_contact_profile("nonexistent") == CONTACT_SHADOW_PROFILES["default"]


def test_heavy_profile_is_denser_than_glass():
    heavy = CONTACT_SHADOW_PROFILES["heavy"]
    glass = CONTACT_SHADOW_PROFILES["glass"]
    assert heavy.opacity > glass.opacity
    assert heavy.blur > glass.blur
