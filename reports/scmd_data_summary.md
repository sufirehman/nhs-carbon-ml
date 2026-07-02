# SCMD Data Summary

**Source file:** `scmd_SCMD_FINAL_202502.csv`  
**Rows:** 313,375  
**Columns:** 8  
**Date range:** 202502 to 202502  

## Columns

| Column | dtype | Non-null count |
|--------|-------|---------------|
| `YEAR_MONTH` | int64 | 313,375 |
| `ODS_CODE` | object | 313,375 |
| `VMP_SNOMED_CODE` | int64 | 313,375 |
| `VMP_PRODUCT_NAME` | object | 313,375 |
| `UNIT_OF_MEASURE_IDENTIFIER` | int64 | 313,375 |
| `UNIT_OF_MEASURE_NAME` | object | 313,375 |
| `TOTAL_QUANITY_IN_VMP_UNIT` | float64 | 313,375 |
| `INDICATIVE_COST` | float64 | 300,836 |

## Key Counts

- **NHS organisations (ODS_CODE):** 194
- **Unique VMP products (SNOMED):** 8,473
- **Total indicative spend:** £1,813,459,454

## Important: Missing BNF Classification

The SCMD dataset does **not** include BNF chapter, section, or paragraph
columns. The dataset uses SNOMED VMP codes for product identification.

BNF-like classification is derived in `src/features/build_defra_mapping.py`
using product-name pattern matching against BNF chapter keyword lists.
This is an approximation; a more accurate mapping would use the NHS dm+d
(Dictionary of Medicines and Devices) SNOMED-to-BNF lookup, available from
NHS TRUD (requires free registration at https://isd.digital.nhs.uk/trud).

## Top 20 Products by Total Indicative Cost

| Product | Total £ |
|---------|---------|
| Generic Kaftrio 75mg/50mg/100mg tablets | £44,031,651 |
| Pembrolizumab 100mg/4ml solution for infusion vials | £39,999,670 |
| Adalimumab 40mg/0.8ml solution for injection pre-filled disposable devices | £38,521,257 |
| Ivacaftor 150mg tablets | £38,090,750 |
| Tafamidis 61mg capsules | £37,076,950 |
| Daratumumab 1.8g/15ml solution for injection vials | £36,832,320 |
| Adalimumab 40mg/0.4ml solution for injection pre-filled disposable devices | £35,850,669 |
| Aflibercept 3.6mg/90microlitres solution for injection pre-filled syringes | £34,755,888 |
| Enzalutamide 40mg tablets | £29,348,991 |
| Infliximab 100mg powder for solution for infusion vials | £27,878,102 |
| Faricimab 28.8mg/0.24ml solution for injection vials | £27,421,572 |
| Emtricitabine 200mg / Tenofovir disoproxil 245mg tablets | £18,847,300 |
| Ustekinumab 90mg/1ml solution for injection pre-filled syringes | £18,073,446 |
| Pertuzumab 600mg/10ml / Trastuzumab 600mg/10ml solution for injection vials | £17,936,703 |
| Nivolumab 240mg/24ml solution for infusion vials | £17,013,788 |
| Ocrelizumab 300mg/10ml solution for infusion vials | £16,161,460 |
| Bictegravir 50mg / Emtricitabine 200mg / Tenofovir alafenamide 25mg tablets | £15,966,569 |
| Etanercept 50mg/1ml solution for injection pre-filled disposable devices | £15,381,724 |
| Acalabrutinib 100mg tablets | £14,950,103 |
| Lenalidomide 10mg capsules | £14,875,740 |

## Top 20 NHS Organisations by Spend

| ODS Code | Total £ |
|----------|---------|
| RAL | £89,273,079 |
| RJ1 | £65,946,981 |
| RR8 | £53,458,776 |
| RRV | £48,330,209 |
| R0A | £46,215,014 |
| RHQ | £45,341,984 |
| RRK | £44,059,236 |
| RTD | £40,644,337 |
| R1H | £38,151,588 |
| RTH | £36,137,902 |
| RX1 | £35,122,790 |
| RHM | £34,018,746 |
| RJZ | £32,080,851 |
| RA7 | £31,723,505 |
| RM3 | £28,664,512 |
| RWE | £27,012,792 |
| RYJ | £26,662,892 |
| RGT | £26,457,227 |
| REN | £23,890,589 |
| RYR | £23,535,642 |