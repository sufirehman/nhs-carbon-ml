"""
Fetch DEFRA/DESNZ 2025 Greenhouse Gas Conversion Factors spreadsheet.

The official "Government conversion factors for company reporting" is
published annually on GOV.UK. We try several known URL patterns and fall
back to a GOV.UK search if none succeed.
"""

import os
import sys
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# GOV.UK attachment URLs follow a stable pattern; try 2025 first, then 2024.
CANDIDATE_URLS = [
    # 2025 — confirmed live as of June 2025 (DESNZ published 2025-06-10)
    "https://assets.publishing.service.gov.uk/media/6846a4f55e92539572806125/ghg-conversion-factors-2025-full-set.xlsx",
    "https://assets.publishing.service.gov.uk/media/6846b6ea57f3515d9611f0dd/ghg-conversion-factors-2025-flat-format.xlsx",
    # 2024 fallback
    "https://assets.publishing.service.gov.uk/media/66a7d4a020aad89b5561a5b2/Conversion_Factors_2024_-_Condensed_set__for_most_users_.xlsx",
    "https://assets.publishing.service.gov.uk/media/65ddb0d0fc2843001ddf7b36/Conversion_Factors_2024_-_Full_set_.xlsx",
]

GOV_UK_PAGE = (
    "https://www.gov.uk/government/collections/"
    "government-conversion-factors-for-company-reporting"
)


def try_download(session, url, dest):
    r = session.get(url, stream=True, timeout=60)
    if r.status_code == 200 and len(r.content) > 10_000:
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return True
    return False


def scrape_govuk_for_xlsx(session):
    """Scrape the GOV.UK collection page for the latest conversion factors xlsx."""
    try:
        r = session.get(GOV_UK_PAGE, timeout=30)
        r.raise_for_status()
        import re
        # Find all .xlsx hrefs on the page
        hrefs = re.findall(r'href="(https://assets\.publishing[^"]+\.xlsx)"', r.text)
        # Prefer 2025, then 2024
        for year in ("2025", "2024", "2023"):
            for h in hrefs:
                if year in h:
                    return h
        return hrefs[0] if hrefs else None
    except Exception:
        return None


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "NHS-Carbon-ML-Research/1.0"

    print("Attempting to download DEFRA/DESNZ GHG Conversion Factors…")

    # First: try scraping the GOV.UK collection page for a live URL
    print("  Step 1: Scraping GOV.UK collection page for xlsx links…")
    live_url = scrape_govuk_for_xlsx(session)
    if live_url:
        print(f"  Found URL on GOV.UK: {live_url}")
        year_tag = "2025" if "2025" in live_url else "2024"
        dest = os.path.join(RAW_DIR, f"ghg_conversion_factors_{year_tag}.xlsx")
        try:
            if try_download(session, live_url, dest):
                size_mb = os.path.getsize(dest) / 1e6
                print(f"  SUCCESS: {dest} ({size_mb:.2f} MB)")
                return
        except Exception as e:
            print(f"  Scrape-URL download failed: {e}")

    # Second: try hardcoded candidate URLs
    print("  Step 2: Trying known candidate URLs…")
    for url in CANDIDATE_URLS:
        year_tag = "2025" if "2025" in url else "2024"
        dest = os.path.join(RAW_DIR, f"ghg_conversion_factors_{year_tag}.xlsx")
        print(f"    Trying: {url}")
        try:
            if try_download(session, url, dest):
                size_mb = os.path.getsize(dest) / 1e6
                print(f"  SUCCESS: {dest} ({size_mb:.2f} MB)")
                return
            else:
                print(f"    Not found or too small (status may be 404).")
        except Exception as e:
            print(f"    Failed: {e}")

    # All automated attempts failed
    print("\n  Automated download unsuccessful.")
    print("  MANUAL DOWNLOAD INSTRUCTIONS:")
    print("  1. Go to:")
    print(f"     {GOV_UK_PAGE}")
    print("  2. Click the most recent year's guidance page (e.g. '2025').")
    print("  3. Under 'Conversion factor documents', download the Excel file")
    print("     labelled 'Conversion Factors [year] - Full set'.")
    print("  4. Save to: data/raw/ghg_conversion_factors_2025.xlsx")
    print()
    print("  Alternatively, search GOV.UK for:")
    print("  'government conversion factors company reporting 2025'")
    sys.exit(1)


if __name__ == "__main__":
    main()
