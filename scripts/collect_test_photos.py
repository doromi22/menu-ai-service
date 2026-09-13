"""
One-off collector (not a pipeline module): pulls freely-licensed food photos
from Wikimedia Commons into photos/<category>/ using the harness's
category-prefix naming (<category>_NNN.ext), and writes a license ledger
next to them so every downloaded file's source/author/license is recorded.

Only free licenses are accepted (CC0/PD/CC BY/CC BY-SA) - anything else is
skipped, matching the licensing bar already applied to model weights
(see standard/segmentation/README.md's rejection of CC BY-NC bria-rmbg).

Run:
    ./venv/Scripts/python.exe scripts/collect_test_photos.py donburi
    ./venv/Scripts/python.exe scripts/collect_test_photos.py donburi dessert pasta
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "menu-ai-test-corpus/1.0 (local research harness; contact: repo owner)"

# Free licenses only. Substring match against Commons' LicenseShortName.
ALLOWED_LICENSE_MARKERS = ("cc0", "public domain", "cc by", "cc-by")
# Explicitly rejected even though they contain "cc by".
REJECTED_LICENSE_MARKERS = ("nc", "nd", "non-commercial", "noderivs")

SEARCH_TERMS: dict[str, list[str]] = {
    "donburi": ["donburi", "gyudon beef bowl", "katsudon", "oyakodon", "kaisendon"],
    "dessert": ["cake slice plate", "parfait glass", "dessert plate restaurant", "ice cream sundae"],
    "pasta": ["spaghetti plate", "carbonara pasta", "penne dish", "pasta restaurant plate"],
    "ramen": ["ramen bowl", "tonkotsu ramen", "shoyu ramen"],
}

MIN_PIXELS = 800  # shorter side; below this the mask/render comparison isn't meaningful
MAX_BYTES = 12_000_000
TARGET_PER_CATEGORY = 13  # matches the existing real ramen sample size
THUMB_WIDTH = 1600  # plenty for segmentation QA, and what Commons asks clients to fetch
DOWNLOAD_PAUSE_SECONDS = 2.0
MAX_DOWNLOAD_ATTEMPTS = 4


def _get(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _download(url: str) -> bytes | None:
    """Fetch one image, backing off on Commons' 429 rate limiting."""
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < MAX_DOWNLOAD_ATTEMPTS:
                wait = DOWNLOAD_PAUSE_SECONDS * (2**attempt)
                print(f"    429 from Commons, waiting {wait:.0f}s (attempt {attempt})")
                time.sleep(wait)
                continue
            print(f"    HTTP {exc.code}")
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"    {type(exc).__name__}: {exc}")
            return None
    return None


def _license_ok(short_name: str) -> bool:
    lowered = short_name.lower()
    if any(marker in lowered for marker in REJECTED_LICENSE_MARKERS):
        return False
    return any(marker in lowered for marker in ALLOWED_LICENSE_MARKERS)


def _search(term: str, limit: int = 30) -> list[dict]:
    data = _get(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": term,
            "gsrnamespace": 6,  # File:
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            # Commons' 429 response explicitly asks clients to take thumbnails
            # rather than full-size originals; this makes `thumburl` available.
            "iiurlwidth": THUMB_WIDTH,
        }
    )
    return list(data.get("query", {}).get("pages", {}).values())


def _plain(extmetadata: dict, key: str) -> str:
    raw = extmetadata.get(key, {}).get("value", "")
    # Commons returns HTML in several of these fields; keep it to one flat line.
    text = raw.replace("\n", " ")
    while "<" in text and ">" in text:
        start = text.index("<")
        end = text.index(">", start)
        text = text[:start] + text[end + 1 :]
    return " ".join(text.split())


def collect(category: str, output_root: Path, already_have: set[str]) -> list[dict]:
    output_dir = output_root / category
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob(f"{category}_*"))
    next_index = 1
    if existing:
        numbers = []
        for path in existing:
            stem_number = path.stem.rsplit("_", 1)[-1]
            if stem_number.isdigit():
                numbers.append(int(stem_number))
        next_index = max(numbers) + 1 if numbers else 1

    ledger: list[dict] = []
    # Seeded from the existing ledger, not just this run: an earlier partial
    # run (e.g. cut short by Commons' rate limiting) would otherwise be
    # re-downloaded under new filenames as duplicates.
    seen_titles: set[str] = set(already_have)

    for term in SEARCH_TERMS.get(category, [category]):
        if len(ledger) >= TARGET_PER_CATEGORY:
            break
        print(f"[{category}] searching: {term}")
        try:
            pages = _search(term)
        except Exception as exc:  # noqa: BLE001 - one bad search must not kill the run
            print(f"  search failed: {type(exc).__name__}: {exc}")
            continue

        for page in pages:
            if len(ledger) >= TARGET_PER_CATEGORY:
                break
            title = page.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)

            info = (page.get("imageinfo") or [{}])[0]
            extmetadata = info.get("extmetadata", {})
            url = info.get("url")
            if not url:
                continue

            suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if min(info.get("width", 0), info.get("height", 0)) < MIN_PIXELS:
                continue
            if info.get("size", 0) > MAX_BYTES:
                continue

            license_name = _plain(extmetadata, "LicenseShortName")
            if not _license_ok(license_name):
                continue

            filename = f"{category}_{next_index:03d}{suffix}"
            destination = output_dir / filename
            download_url = info.get("thumburl") or url
            payload = _download(download_url)
            if payload is None:
                print(f"  download failed ({title})")
                continue

            destination.write_bytes(payload)
            next_index += 1
            ledger.append(
                {
                    "file": filename,
                    "category": category,
                    "commons_title": title,
                    "license": license_name,
                    "author": _plain(extmetadata, "Artist"),
                    "credit": _plain(extmetadata, "Credit"),
                    "source_url": url,
                    "description_page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                    "width": info.get("width"),
                    "height": info.get("height"),
                }
            )
            print(f"  saved {filename}  [{license_name}]  {info.get('width')}x{info.get('height')}")
            time.sleep(DOWNLOAD_PAUSE_SECONDS)  # be polite to Commons

    return ledger


def main() -> None:
    categories = sys.argv[1:] or ["donburi", "dessert", "pasta"]
    output_root = Path(__file__).resolve().parents[1] / "photos"
    ledger_path = output_root / "photo_license_ledger.csv"

    all_rows: list[dict] = []
    if ledger_path.exists():
        with open(ledger_path, newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))

    already_have = {row.get("commons_title", "") for row in all_rows}
    for category in categories:
        new_rows = collect(category, output_root, already_have)
        already_have.update(row["commons_title"] for row in new_rows)
        all_rows.extend(new_rows)

    fieldnames = [
        "file",
        "category",
        "commons_title",
        "license",
        "author",
        "credit",
        "source_url",
        "description_page",
        "width",
        "height",
    ]
    with open(ledger_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"\nLicense ledger: {ledger_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
