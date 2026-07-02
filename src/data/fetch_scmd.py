"""
Fetch NHSBSA Secondary Care Medicines Data (SCMD) with Indicative Price.

The NHSBSA Open Data Portal exposes datasets via a CKAN-compatible API.
We query the package list, find the SCMD dataset, and download the most
recent CSV resource.
"""

import os
import sys
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PORTAL_BASE = "https://opendata.nhsbsa.net"
API_BASE = f"{PORTAL_BASE}/api/3/action"

KNOWN_SCMD_IDS = [
    "secondary-care-medicines-data-indicative-price",
    "scmd",
]


def find_scmd_package(session):
    r = session.get(f"{API_BASE}/package_list", timeout=30)
    r.raise_for_status()
    packages = r.json().get("result", [])
    for pid in packages:
        if any(kw in pid.lower() for kw in ["scmd", "secondary-care-med"]):
            return pid
    # fallback: try known IDs directly
    for pid in KNOWN_SCMD_IDS:
        try:
            r2 = session.get(f"{API_BASE}/package_show?id={pid}", timeout=30)
            if r2.ok and r2.json().get("success"):
                return pid
        except Exception:
            pass
    return None


def download_resource(session, url, dest_path):
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "NHS-Carbon-ML-Research/1.0"

    print("Querying NHSBSA Open Data Portal for SCMD dataset…")
    try:
        pkg_id = find_scmd_package(session)
    except Exception as e:
        print(f"  ERROR reaching portal: {e}")
        pkg_id = None

    if not pkg_id:
        print("\n  Could not locate SCMD package via API.")
        print("  MANUAL DOWNLOAD INSTRUCTIONS:")
        print("  1. Go to: https://opendata.nhsbsa.net/dataset")
        print("  2. Search for: 'Secondary Care Medicines Data Indicative Price'")
        print("  3. Click the most recent month's dataset.")
        print("  4. Download the CSV resource.")
        print("  5. Save to: data/raw/scmd_latest.csv")
        sys.exit(1)

    print(f"  Found package: {pkg_id}")
    r = session.get(f"{API_BASE}/package_show?id={pkg_id}", timeout=30)
    r.raise_for_status()
    resources = r.json()["result"]["resources"]

    csv_resources = [
        res for res in resources
        if res.get("format", "").upper() in ("CSV", "")
        and res.get("url", "").lower().endswith(".csv")
    ]
    if not csv_resources:
        csv_resources = resources  # fallback: take whatever is there

    # Sort by created/last_modified descending to get the most recent
    csv_resources.sort(key=lambda x: x.get("created", ""), reverse=True)
    target = csv_resources[0]
    url = target["url"]
    name = target.get("name", "scmd_latest").replace(" ", "_") + ".csv"
    dest = os.path.join(RAW_DIR, f"scmd_{name}")

    print(f"  Downloading: {url}")
    try:
        download_resource(session, url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  SUCCESS: saved to {dest} ({size_mb:.1f} MB)")

        import csv
        with open(dest, newline="", encoding="utf-8-sig") as f:
            rows = sum(1 for _ in csv.reader(f)) - 1
        print(f"  Rows (excl. header): {rows:,}")
    except Exception as e:
        print(f"  DOWNLOAD FAILED: {e}")
        print("  MANUAL DOWNLOAD INSTRUCTIONS:")
        print(f"  1. Open: {PORTAL_BASE}/dataset/{pkg_id}")
        print("  2. Click the CSV download link for the most recent month.")
        print("  3. Save to: data/raw/scmd_latest.csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
