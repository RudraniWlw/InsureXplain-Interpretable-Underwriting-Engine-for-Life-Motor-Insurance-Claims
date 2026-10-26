# InsureXplain - Interpretable Underwriting Engine (Motor Insurance)

Microproject 7 | Module 1: Theoretical foundations of XAI (global vs local explanations, SHAP marginal contribution)

## Run
    pip install -r requirements.txt
    python insurexplain.py
Open `output/InsureXplain_report.html` in a browser.

## Deliverables -> where they are
1. Training pipeline + explanation module: `insurexplain.py` (train_model, explain_global, explain_local)
2. Adverse Action Notice generator: `generate_notices()` -> `output/notices/*.txt`, `output/reason_codes.csv`
3. Global compliance report: `compliance_audit()` -> report section "Regulatory compliance report", `output/proxy_audit.csv`

## Design choices
- Gender and marital status are excluded from training; they are used only for auditing.
- A positive SHAP value is an adverse factor; the top three become the notice reasons.
- LIME is used as a cross-check on the SHAP reasons.
- Data are synthetic. Notices are templates for coursework, not legal advice.
