# Manual Collection: NHS Supplier Carbon Reduction Plans (CRPs)

## Why manual?

NHS Evergreen Sustainable Supplier Assessments require suppliers to submit
Carbon Reduction Plans (CRPs) as a condition of contracting. However, these
are published as individual PDFs per supplier — there is no central bulk
download. They appear across:

- Supplier corporate websites (usually under "Sustainability" or "ESG")
- NHS Supply Chain supplier portal pages
- Individual procurement portals (e.g. Find a Tender Service attachments)

## Target: 15–20 priority suppliers

Focus on high-spend, high-emission categories first (medical devices,
pharmaceuticals, PPE, logistics). Suggested starting list:

| Priority | Supplier | Category | Where to look |
|----------|----------|----------|---------------|
| 1 | Baxter Healthcare | Pharmaceuticals/IV | baxter.com > Sustainability |
| 2 | BD (Becton Dickinson) | Medical devices | bd.com > ESG |
| 3 | B. Braun | Medical devices | bbraun.com > Sustainability |
| 4 | Cardinal Health | Medical/surgical | cardinalhealth.com |
| 5 | DHL Supply Chain (NHS) | Logistics | dhl.com > Sustainability |
| 6 | Fresenius Kabi | Pharmaceuticals | fresenius-kabi.com |
| 7 | GE Healthcare | Imaging/diagnostics | gehealthcare.com |
| 8 | Johnson & Johnson MedTech | Medical devices | jnj.com > ESG |
| 9 | Lohmann & Rauscher | Wound care/PPE | lrmed.com |
| 10 | Medtronic | Medical devices | medtronic.com > Sustainability |
| 11 | Philips Healthcare | Diagnostics | philips.com > Sustainability |
| 12 | Reckitt (healthcare div.) | Hygiene/PPE | reckitt.com |
| 13 | Siemens Healthineers | Diagnostics | siemens-healthineers.com |
| 14 | Smith+Nephew | Wound/orthopaedic | smith-nephew.com |
| 15 | Stryker | Orthopaedics | stryker.com > Sustainability |

## How to find CRPs

1. **NHS Supply Chain portal**: https://www.nhssupplychain.nhs.uk  
   Search the supplier name — some list their CRP PDF directly.

2. **Find a Tender Service**: https://www.find-tender.service.gov.uk  
   Search for the supplier in awarded contracts; attachments sometimes include CRPs.

3. **Supplier website** → Sustainability / ESG / Corporate Responsibility section.  
   Look for a document titled "Carbon Reduction Plan" (required wording for PPN 06/21 compliance).

4. **Google**: `"[Supplier Name]" "carbon reduction plan" filetype:pdf site:[supplier domain]`

## Where to save

Save each PDF to: `data/raw/crps/<supplier_slug>.pdf`

Example: `data/raw/crps/medtronic_crp.pdf`

## Next step

Once 15+ PDFs are collected, run `src/data/parse_supplier_crps.py` (to be written)
to extract:
- Baseline year and scope 1/2/3 emissions
- Reduction targets and timelines
- Net-zero commitment date
