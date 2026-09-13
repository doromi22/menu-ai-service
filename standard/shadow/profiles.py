"""
Per-object contact-shadow profiles (spec §4: "Teishoku/Set Meal: 단일 giant
shadow 금지. Object별 contact shadow + 전체 ambient shadow 조합").

`FoodObject.shadow_profile` (spec §2) names one of these - independent of
the template's own `shadow` block (§9: `{"opacity": 0.12, "blur": 42,
"offset_y": 18}`), which drives the *ambient* shadow instead (see
standard/shadow/engine.py). This is what gives shadow_profile an actual
job: a donburi bowl and a small side dish can cast visibly different
contact shadows in the same frame, under the same background template.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactShadowParams:
    blur: float
    opacity: float
    offset_y: int


CONTACT_SHADOW_PROFILES: dict[str, ContactShadowParams] = {
    "default": ContactShadowParams(blur=4.0, opacity=0.55, offset_y=4),
    "heavy": ContactShadowParams(blur=6.0, opacity=0.70, offset_y=6),  # e.g. a large donburi bowl
    "glass": ContactShadowParams(blur=2.0, opacity=0.30, offset_y=2),  # e.g. a drink glass - light, tight shadow
}


def resolve_contact_profile(name: str) -> ContactShadowParams:
    return CONTACT_SHADOW_PROFILES.get(name, CONTACT_SHADOW_PROFILES["default"])
