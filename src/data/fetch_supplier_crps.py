"""
TODO: Collect NHS Supplier Carbon Reduction Plans (CRPs).

Supplier CRPs are NOT centrally downloadable. Each NHS Evergreen-assessed
supplier publishes their CRP as an individual PDF on their own website or
via the NHS Supply Chain portal. There is no bulk download endpoint.

Plan:
- Manually collect 15-20 priority supplier CRPs (see MANUAL_CRP_COLLECTION.md).
- Save PDFs to data/raw/crps/<supplier_name>.pdf
- Use pdfplumber (already in requirements.txt) to extract text + tables.
- Build a structured dataset of commitments, baselines, and targets.

Once PDFs are collected, run:
    python src/data/parse_supplier_crps.py   # (to be written)
"""

raise NotImplementedError(
    "Supplier CRP collection is manual. "
    "See data/raw/MANUAL_CRP_COLLECTION.md for instructions."
)
