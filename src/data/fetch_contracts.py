"""
Fetch NHSBSA contract / procurement data from the Open Data Portal.

The NHSBSA portal hosts several contract-related datasets. We query the
package list and download any CSV resources whose package name matches
contract-related keywords.
"""

import os
import sys
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
API_BASE = "https://opendata.nhsbsa.net/api/3/action"
PORTAL_BASE = "https://opendata.nhsbsa.net"

CONTRACT_KEYWORDS = ["contract", "procurement", "spend", "dispensing", "epact"]


def find_contract_packages(session):
    r = session.get(f"{API_BASE}/package_list", timeout=30)
    r.raise_for_status()
    packages = r.json().get("result", [])
    return [p for p in packages if any(kw in p.lower() for kw in CONTRACT_KEYWORDS)]


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

    print("Querying NHSBSA Open Data Portal for contract/procurement datasets…")
    try:
        matches = find_contract_packages(session)
    except Exception as e:
        print(f"  ERROR reaching portal: {e}")
        matches = []

    if not matches:
        print("\n  No contract datasets found via API.")
        print("  MANUAL DOWNLOAD INSTRUCTIONS:")
        print("  1. Go to: https://opendata.nhsbsa.net/dataset")
        print("  2. Search for: 'contract' or 'spend'")
        print("  3. Download the relevant CSV files.")
        print("  4. Save to: data/raw/contracts_<name>.csv")
        sys.exit(1)

    print(f"  Found {len(matches)} matching package(s): {matches}")
    downloaded = []

    for pkg_id in matches[:3]:  # cap at 3 to avoid very large downloads
        try:
            r = session.get(f"{API_BASE}/package_show?id={pkg_id}", timeout=30)
            r.raise_for_status()
            resources = r.json()["result"]["resources"]
        except Exception as e:
            print(f"  Could not fetch resources for {pkg_id}: {e}")
            continue

        csv_res = [
            res for res in resources
            if res.get("format", "").upper() == "CSV"
            or res.get("url", "").lower().endswith(".csv")
        ]
        if not csv_res:
            print(f"  No CSV resources in {pkg_id}, skipping.")
            continue

        # Take only the most recent resource per package
        csv_res.sort(key=lambda x: x.get("created", ""), reverse=True)
        res = csv_res[0]
        url = res["url"]
        safe_name = pkg_id.replace("-", "_")[:60]
        dest = os.path.join(RAW_DIR, f"contracts_{safe_name}.csv")

        print(f"  Downloading {pkg_id}: {url}")
        try:
            download_resource(session, url, dest)
            size_mb = os.path.getsize(dest) / 1e6
            import csv
            with open(dest, newline="", encoding="utf-8-sig") as f:
                rows = sum(1 for _ in csv.reader(f)) - 1
            print(f"  SUCCESS: {dest} ({size_mb:.1f} MB, {rows:,} rows)")
            downloaded.append(dest)
        except Exception as e:
            print(f"  DOWNLOAD FAILED for {pkg_id}: {e}")

    if not downloaded:
        print("\n  All downloads failed.")
        print("  MANUAL DOWNLOAD INSTRUCTIONS:")
        print("  1. Go to: https://opendata.nhsbsa.net/dataset")
        print("  2. Filter or search for 'contract' datasets.")
        print("  3. Download CSV resources and save to data/raw/.")
        sys.exit(1)
    else:
        print(f"\n  Done. {len(downloaded)} file(s) saved.")


if __name__ == "__main__":
    main()
