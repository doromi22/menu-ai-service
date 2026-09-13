"""
Immutable Policy YAML loader (spec §6).

`load()` computes a sha256 hash of the policy file's raw bytes - this
becomes the §8 `policy_hash` metadata field stamped on every processed
image - and checks it against `policy.lock.json`, a small version->hash
registry that is this policy's code-level "git tag/digest" (spec §6:
"정책 파일은 immutable artifact, git tag/digest로 고정"). A policy_version
must be deliberately pinned with `register()` before `load()` will accept
it, and `load()` refuses to proceed if a previously-registered version's
file content ever drifts - i.e. someone edited an "immutable" artifact in
place instead of bumping policy_version.

The hash can't be stored inside the policy file itself (a file cannot
correctly hash its own content, since the hash field would have to include
its own value), which is why it lives in this sibling lock file instead.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from standard.policy.schema import Policy, validate_schema


class PolicyIntegrityError(Exception):
    pass


def _hash_bytes(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


class PolicyLoader:
    def __init__(self, lock_path: str | Path) -> None:
        self._lock_path = Path(lock_path)

    def _read_lock(self) -> dict:
        if not self._lock_path.exists():
            return {}
        return json.loads(self._lock_path.read_text(encoding="utf-8"))

    def _write_lock(self, lock: dict) -> None:
        self._lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _read_and_validate(path: Path) -> tuple[dict, bytes, str]:
        raw_bytes = path.read_bytes()
        data = yaml.safe_load(raw_bytes) or {}
        validate_schema(data)
        return data, raw_bytes, data["policy_version"]

    def load(self, path: str | Path, *, require_locked: bool = True) -> Policy:
        data, raw_bytes, version = self._read_and_validate(Path(path))
        computed_hash = _hash_bytes(raw_bytes)

        locked_hash = self._read_lock().get(version)
        if locked_hash is None:
            if require_locked:
                raise PolicyIntegrityError(
                    f"policy_version {version!r} is not registered in {self._lock_path.name}. "
                    "Call PolicyLoader.register(path) deliberately before using it in a "
                    "pipeline run, so its content can never silently change (spec §6)."
                )
        elif locked_hash != computed_hash:
            raise PolicyIntegrityError(
                f"policy_version {version!r} content changed since it was locked "
                f"(locked {locked_hash}, current {computed_hash}). An immutable policy "
                "artifact must not be edited in place - bump policy_version instead (spec §6)."
            )

        return Policy(version=version, hash=computed_hash, raw=data)

    def register(self, path: str | Path) -> Policy:
        """Deliberately pin a policy file's current content as the locked
        artifact for its policy_version - the code-level analogue of
        tagging it in git (spec §6)."""
        data, raw_bytes, version = self._read_and_validate(Path(path))
        computed_hash = _hash_bytes(raw_bytes)

        lock = self._read_lock()
        existing = lock.get(version)
        if existing is not None and existing != computed_hash:
            raise PolicyIntegrityError(
                f"refusing to re-register policy_version {version!r} with different content "
                f"(locked {existing}, new {computed_hash}). Bump policy_version instead."
            )

        lock[version] = computed_hash
        self._write_lock(lock)
        return Policy(version=version, hash=computed_hash, raw=data)
