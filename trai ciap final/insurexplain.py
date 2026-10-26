"""
InsureXplain - Interpretable Underwriting Engine for Motor Insurance
Microproject 7 | Module 1: XAI foundations (global vs local, SHAP marginal contribution)

Pipeline
  1. make_data()            synthetic motor-insurance applicants (no real PII)
  2. train_model()          gradient-boosted risk model (protected attributes EXCLUDED)
  3. explain_global()       SHAP summary / importance / interaction plots
  4. explain_local()        SHAP force plot + LIME cross-check for one applicant
  5. generate_notices()     plain-language Adverse Action Notices (top-3 adverse factors)
  6. compliance_audit()     disparate-impact + proxy-feature audit -> HTML report

Run:  python insurexplain.py        (outputs go to ./output)
"""
import os, io, base64, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")
SEED = 42
OUT = "output"
FIG = os.path.join(OUT, "figures")
NOTICES = os.path.join(OUT, "notices")
for d in (OUT, FIG, NOTICES):
    os.makedirs(d, exist_ok=True)

# ---- colour theme (pink / sage / cream) ------------------------------------
PINK, HOT, SAGE, DEEP, CREAM, INK = "#f7c6dc", "#e8789f", "#a9bf7a", "#6f8240", "#fffbe6", "#5c6b36"
CMAP = LinearSegmentedColormap.from_list("pinksage", [SAGE, "#f3ecb8", HOT])
plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
    "axes.edgecolor": DEEP, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10,
})

PROTECTED = ["gender", "marital_status"]          # never given to the model
FEATURES = ["age", "driving_experience", "vehicle_age", "vehicle_value", "annual_mileage",
            "night_driving_pct", "prior_claims", "traffic_violations", "credit_score",
            "sports_vehicle", "urban_garaging"]


# =============================================================================
# 1. DATA
# =============================================================================
def make_data(n=8000, seed=SEED):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["F", "M"], n)
    marital = rng.choice(["single", "married", "divorced"], n, p=[.4, .45, .15])
    age = np.clip(rng.normal(41, 14, n), 18, 80).round()
    exp = np.clip(age - 18 - rng.integers(0, 6, n), 0, None)
    vehicle_age = rng.integers(0, 19, n)
    vehicle_value = np.clip(rng.normal(22000, 9000, n), 3000, 80000).round(-2)
    mileage = np.clip(rng.normal(12000, 5000, n), 2000, 40000).round(-2)
    night = np.clip(rng.normal(12, 8, n), 0, 60).round()
    prior = rng.poisson(0.35 + 0.35 * (age < 25), n)
    viol = rng.poisson(0.5, n)
    credit = np.clip(rng.normal(690, 80, n), 300, 850).round()
    # NOTE: sports cars are more common among men in this data -> a realistic proxy
    sports = (rng.random(n) < np.where(gender == "M", 0.22, 0.07)).astype(int)
    urban = (rng.random(n) < 0.55).astype(int)

    z = (0.55 * prior + 0.35 * viol - 0.035 * exp - 0.006 * (credit - 690)
         + 0.03 * night + 0.02 * vehicle_age + 0.5 * sports + 0.3 * urban
         + 0.00003 * (vehicle_value - 22000)
         + 0.00012 * mileage * (age < 25))                 # young x high-mileage interaction
    lo, hi = -10, 10
    for _ in range(60):                                    # calibrate to ~25% claim rate
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > 0.25:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(z + mid)))
    y = (rng.random(n) < p).astype(int)

    return pd.DataFrame(dict(
        age=age, driving_experience=exp, vehicle_age=vehicle_age, vehicle_value=vehicle_value,
        annual_mileage=mileage, night_driving_pct=night, prior_claims=prior,
        traffic_violations=viol, credit_score=credit, sports_vehicle=sports,
        urban_garaging=urban, gender=gender, marital_status=marital, high_risk=y))


# =============================================================================
# 2. MODEL
# =============================================================================
def train_model(df):
    X, y = df[FEATURES], df["high_risk"]
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.25, random_state=SEED, stratify=y)
    model = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=SEED).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    threshold = float(np.quantile(proba, 0.80))            # decline the riskiest ~20%
    metrics = dict(auc=roc_auc_score(yte, proba),
                   accuracy=accuracy_score(yte, proba >= threshold),
                   threshold=threshold, n_train=len(Xtr), n_test=len(Xte))
    return model, Xtr, Xte, ytr, yte, idx_te, proba, metrics


# =============================================================================
# 3. GLOBAL EXPLANATIONS
# =============================================================================
def save_fig(name):
    path = os.path.join(FIG, name)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


def explain_global(explainer, Xte):
    sv = explainer.shap_values(Xte)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, Xte, cmap=CMAP, show=False)
    plt.title("Global SHAP summary (each dot = one applicant)", color=INK)
    p1 = save_fig("global_summary.png")

    imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns).sort_values()
    plt.figure(figsize=(7, 4.2))
    plt.barh(imp.index, imp.values, color=[HOT if v >= imp.values[-3] else SAGE for v in imp.values])
    plt.xlabel("mean |SHAP value| (log-odds)")
    plt.title("Global feature importance", color=INK)
    p2 = save_fig("global_importance.png")

    sub = Xte.sample(400, random_state=SEED)
    iv = explainer.shap_interaction_values(sub)
    mat = np.abs(iv).mean(0)
    np.fill_diagonal(mat, 0)
    plt.figure(figsize=(7, 6))
    plt.imshow(mat, cmap=CMAP)
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=60, ha="right")
    plt.yticks(range(len(FEATURES)), FEATURES)
    plt.colorbar(label="mean |interaction|")
    plt.title("SHAP interaction strength (off-diagonal)", color=INK)
    p3 = save_fig("interactions.png")

    pairs = [(FEATURES[i], FEATURES[j], mat[i, j] + mat[j, i])
             for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))]
    pairs = sorted(pairs, key=lambda t: -t[2])[:5]
    return sv, imp.sort_values(ascending=False), pairs, (p1, p2, p3)


# =============================================================================
# 4. LOCAL EXPLANATIONS (SHAP force plot + LIME cross-check)
# =============================================================================
def top_adverse(sv_row, cols, k=3):
    """Features that PUSH RISK UP, largest first (positive SHAP = adverse)."""
    s = pd.Series(sv_row, index=cols)
    return list(s[s > 0].sort_values(ascending=False).head(k).index)


def lime_top3(lime_exp, model, row, cols):
    e = lime_exp.explain_instance(row.values, model.predict_proba,
                                  num_features=len(cols), num_samples=1500)
    w = sorted(e.as_map()[1], key=lambda t: -t[1])
    return [cols[i] for i, wt in w if wt > 0][:3], e


def explain_local(explainer, model, Xtr, Xte, sv, declined_pos):
    cols = list(Xte.columns)
    pos = declined_pos[0]
    row = Xte.iloc[pos]
    base = float(np.ravel(explainer.expected_value)[0])
    plt.figure()
    shap.force_plot(base, sv[pos], row.round(1), matplotlib=True, show=False,
                    figsize=(15, 3.4), text_rotation=15, plot_cmap=[HOT, SAGE])
    p_force = save_fig("local_force_plot.png")

    lime = LimeTabularExplainer(Xtr.values, feature_names=cols, class_names=["ok", "high risk"],
                                mode="classification", random_state=SEED)
    # agreement between SHAP and LIME top-3 across the first 25 declined applicants
    overlaps = []
    for q in declined_pos[:25]:
        l3, _ = lime_top3(lime, model, Xte.iloc[q], cols)
        s3 = top_adverse(sv[q], cols)
        overlaps.append(len(set(l3) & set(s3)) / 3)
    l3, e = lime_top3(lime, model, row, cols)
    s3 = top_adverse(sv[pos], cols)
    mp = dict(e.as_map()[1])
    plt.figure(figsize=(7, 4))
    x = np.arange(len(cols))
    plt.bar(x - 0.2, sv[pos], 0.4, color=HOT, label="SHAP")
    lw = np.array([mp.get(i, 0) for i in range(len(cols))])
    plt.bar(x + 0.2, lw, 0.4, color=SAGE, label="LIME (scaled)")
    plt.xticks(x, cols, rotation=60, ha="right")
    plt.axhline(0, color=DEEP, lw=.8)
    plt.legend()
    plt.title("Local explanation: SHAP vs LIME", color=INK)
    p_cmp = save_fig("local_shap_vs_lime.png")
    return pos, s3, l3, float(np.mean(overlaps)), (p_force, p_cmp)


# =============================================================================
# 5. ADVERSE ACTION NOTICE GENERATOR
# =============================================================================
REASONS = {
    "prior_claims": (lambda v: f"Your record shows {int(v)} claim(s) in the last 3 years, which is linked to a higher chance of future claims.",
                     "Claims fall out of our rating window over time; a claim-free period will improve your profile."),
    "traffic_violations": (lambda v: f"Your driving record shows {int(v)} traffic violation(s) in the last 3 years.",
                           "Violations drop off after 3 years; completing an approved defensive-driving course may help."),
    "credit_score": (lambda v: f"Your credit-based insurance score ({int(v)}) is below the level used for our standard terms.",
                     "Paying bills on time and lowering card balances can raise this score. You may request a free copy of your report."),
    "driving_experience": (lambda v: f"You have {int(v)} year(s) of licensed driving experience, and experience is a strong predictor of safe driving.",
                           "This improves automatically with time on the road."),
    "annual_mileage": (lambda v: f"Your estimated annual mileage ({int(v):,} miles) is high, which means more time exposed to risk.",
                       "If your mileage estimate is wrong, tell us and we will re-assess."),
    "night_driving_pct": (lambda v: f"About {int(v)}% of your driving takes place at night, when claim rates are higher.",
                          "A lower share of night driving can improve your assessment."),
    "vehicle_age": (lambda v: f"The insured vehicle is {int(v)} year(s) old, and older vehicles have fewer safety features.",
                    "A vehicle with newer safety technology may qualify for better terms."),
    "vehicle_value": (lambda v: f"The insured vehicle's value (${int(v):,}) increases the potential cost of a claim.",
                      "A different vehicle choice may change the terms offered."),
    "sports_vehicle": (lambda v: "The insured vehicle is classed as a high-performance model.",
                       "A standard-class vehicle may qualify for different terms."),
    "urban_garaging": (lambda v: "The vehicle is kept in a high-traffic urban area, where claim frequency is higher.",
                       "Secure off-street parking may improve your assessment."),
    "age": (lambda v: "Your age group, combined with your mileage and experience, is associated with a higher claim rate.",
            "You may ask for a manual review of this combined factor."),
}


def build_notice(app_id, row, factors):
    today = datetime.date.today().strftime("%B %d, %Y")
    lines = [
        "=" * 66,
        "  ADVERSE ACTION NOTICE  |  InsureXplain Underwriting",
        "=" * 66,
        f"Date: {today}      Reference: {app_id}",
        "",
        "Dear Applicant,",
        "",
        "We are unable to offer you motor insurance on the standard terms you applied",
        "for. This decision was made with the help of an automated risk model.",
        "",
        "The three factors that contributed most to this decision were:",
        "",
    ]
    for i, f in enumerate(factors, 1):
        reason_fn, hint = REASONS[f]
        lines += [f"  {i}. {reason_fn(row[f])}", f"     What may help: {hint}", ""]
    lines += [
        "Your rights",
        "  - You can ask for a human underwriter to review this decision.",
        "  - You can ask for a copy of the information we used and correct any errors.",
        "  - Characteristics such as gender and marital status are NOT used in this model.",
        "",
        "Sincerely,",
        "InsureXplain Underwriting Team",
        "(Template for academic demonstration; not legal advice.)",
    ]
    return "\n".join(lines)


def generate_notices(Xte, sv, declined_pos, idx_te, limit=30):
    cols = list(Xte.columns)
    log, first = [], None
    for q in declined_pos[:limit]:
        row, app_id = Xte.iloc[q], f"APP-{idx_te[q]:05d}"
        factors = top_adverse(sv[q], cols)
        text = build_notice(app_id, row, factors)
        assert not set(factors) & {"gender", "marital_status"}   # protected attrs never cited as reasons
        with open(os.path.join(NOTICES, f"{app_id}.txt"), "w") as fh:
            fh.write(text)
        log.append(dict(applicant=app_id, reason_1=factors[0], reason_2=factors[1] if len(factors) > 1 else "",
                        reason_3=factors[2] if len(factors) > 2 else ""))
        if first is None:
            first = text
    pd.DataFrame(log).to_csv(os.path.join(OUT, "reason_codes.csv"), index=False)
    return first


# =============================================================================
# 6. COMPLIANCE AUDIT
# =============================================================================
def compliance_audit(df, Xte, idx_te, proba, thr, sv, imp):
    meta = df.loc[idx_te, PROTECTED].reset_index(drop=True)
    declined = (proba >= thr).astype(int)
    audit = {}
    # (a) disparate impact on approval rate (four-fifths rule)
    rates = {}
    for attr in PROTECTED:
        approval = pd.Series(1 - declined).groupby(meta[attr]).mean()
        ratio = approval.min() / approval.max()
        rates[attr] = dict(approval=approval.round(3).to_dict(), ratio=round(float(ratio), 3),
                           flag=bool(ratio < 0.80))
    audit["disparate_impact"] = rates
    # (b) proxy screen: correlation of every model feature with protected attrs
    g = (meta["gender"] == "M").astype(int)
    m = (meta["marital_status"] == "married").astype(int)
    shap_df = pd.DataFrame(sv, columns=Xte.columns)
    rows = []
    for f in Xte.columns:
        cg = float(np.corrcoef(Xte[f].values, g)[0, 1])
        cm = float(np.corrcoef(Xte[f].values, m)[0, 1])
        gap = float(shap_df[f][g == 1].mean() - shap_df[f][g == 0].mean())
        flag = (max(abs(cg), abs(cm)) > 0.15) or abs(gap) > 0.10
        rows.append(dict(feature=f, corr_gender=round(cg, 3), corr_married=round(cm, 3),
                         shap_gap_M_minus_F=round(gap, 3), importance=round(float(imp[f]), 3),
                         flag=bool(flag)))
    audit["proxy"] = pd.DataFrame(rows).sort_values("importance", ascending=False)
    audit["decline_by_gender"] = pd.Series(declined).groupby(meta["gender"]).mean().round(3).to_dict()
    return audit


# =============================================================================
# 7. HTML REPORT (pink / sage / cream theme)
# =============================================================================
def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


CSS = """
:root{--pink:#f7c6dc;--hot:#e8789f;--sage:#a9bf7a;--deep:#6f8240;--cream:#fffbe6;--ink:#5c6b36;--line:#8fa65c}
*{box-sizing:border-box}
body{margin:0;padding:18px;font-family:'DotGothic16','Nunito','Segoe UI',sans-serif;color:var(--ink);line-height:1.65;
background:var(--pink);background-image:radial-gradient(circle at 12px 12px,#fbd9e8 2px,transparent 3px);background-size:28px 28px}
.wrap{max-width:1100px;margin:0 auto;border:2px solid var(--line);border-radius:14px;padding:14px;background:#fbd3e4}
header{border:2px solid var(--line);border-radius:10px;padding:26px 30px;background:linear-gradient(120deg,#fde7f1,#f7d9a8 50%,#d9e6b0)}
header h1{margin:0;font-size:2.6rem;color:var(--deep);letter-spacing:1px;text-shadow:2px 2px 0 #fff}
header p{margin:6px 0 0;max-width:60ch}
.grid{display:grid;grid-template-columns:210px 1fr;gap:14px;margin-top:14px}
nav{border:2px solid var(--line);border-radius:10px;background:var(--cream);padding:12px;align-self:start;position:sticky;top:12px}
nav a{display:block;margin-bottom:8px;padding:6px 10px;border:2px solid var(--line);border-radius:6px;text-decoration:none;color:var(--ink);
background:linear-gradient(#e7f0cd,#bcd08a);font-size:.95rem}
nav a:hover,nav a:focus-visible{background:linear-gradient(#fde0ec,var(--hot));color:#fff;outline:none}
main section{border:2px solid var(--line);border-radius:10px;background:var(--cream);padding:18px 22px;margin-bottom:14px}
h2{margin:0 0 8px;color:var(--hot);font-size:1.5rem}
h3{color:var(--deep);margin:16px 0 6px}
.stats{display:flex;gap:10px;flex-wrap:wrap}
.stat{flex:1 1 130px;border:2px dashed var(--line);border-radius:8px;padding:8px 12px;background:#fff7d6}
.stat b{display:block;font-size:1.5rem;color:var(--hot)}
img{max-width:100%;border:2px solid var(--line);border-radius:8px;background:var(--cream)}
table{border-collapse:collapse;width:100%;font-size:.9rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:left}
th{background:#e1ecc2}
.flag{background:#fdd0e0;color:#b03060;font-weight:bold}
.ok{background:#e7f0cd}
pre{background:#fff;border:2px solid var(--line);border-radius:8px;padding:14px;overflow-x:auto;font-size:.82rem;line-height:1.45;color:#444}
footer{margin-top:14px;text-align:center;border:2px solid var(--line);border-radius:10px;background:#e1ecc2;padding:10px;font-size:.85rem}
@media(max-width:760px){.grid{grid-template-columns:1fr}nav{position:static}}
"""


def build_report(metrics, imp, pairs, figs, audit, local, notice_text, df):
    gfig, ifig, xfig = figs[0]
    forcefig, cmpfig = figs[1]
    pos, s3, l3, agree = local
    di = audit["disparate_impact"]
    di_rows = "".join(
        f"<tr><td>{a}</td><td>{v['approval']}</td><td>{v['ratio']}</td>"
        f"<td class='{'flag' if v['flag'] else 'ok'}'>{'REVIEW (below 0.80)' if v['flag'] else 'Pass (0.80 or above)'}</td></tr>"
        for a, v in di.items())
    px = audit["proxy"]
    px_rows = "".join(
        f"<tr><td>{r.feature}</td><td>{r.importance}</td><td>{r.corr_gender}</td><td>{r.corr_married}</td>"
        f"<td>{r.shap_gap_M_minus_F}</td><td class='{'flag' if r.flag else 'ok'}'>{'Possible proxy' if r.flag else 'Clear'}</td></tr>"
        for r in px.itertuples())
    pair_rows = "".join(f"<tr><td>{a} x {b}</td><td>{v:.3f}</td></tr>" for a, b, v in pairs)
    flagged = [r.feature for r in px.itertuples() if r.flag]
    top3 = ", ".join(list(imp.index[:3]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>InsureXplain Report</title>
<link href="https://fonts.googleapis.com/css2?family=DotGothic16&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div class="wrap">
<header><h1>InsureXplain</h1><p>An interpretable underwriting engine for motor insurance. Every decline comes with reasons a customer can read, and every model gets a fairness check.</p></header>
<div class="grid"><nav><a href="#model">Model</a><a href="#global">Global view</a><a href="#local">Local view</a><a href="#notice">Adverse notice</a><a href="#audit">Compliance audit</a><a href="#notes">Method notes</a></nav><main>
<section id="model"><h2>Model snapshot</h2><div class="stats">
<div class="stat"><b>{metrics['auc']:.3f}</b>ROC-AUC</div>
<div class="stat"><b>{metrics['accuracy']:.1%}</b>accuracy at cut-off</div>
<div class="stat"><b>{metrics['threshold']:.2f}</b>decline threshold</div>
<div class="stat"><b>{metrics['n_train']+metrics['n_test']:,}</b>synthetic applicants</div></div>
<p>A gradient-boosted tree model predicts whether an applicant is high risk. Gender and marital status are held out of training, so they cannot be used directly. The riskiest 20% of test applicants are declined.</p></section>
<section id="global"><h2>Global explanations (SHAP)</h2>
<p>Global plots show what drives the model across all applicants. The three strongest drivers are <b>{top3}</b>. Pink dots mean a high feature value; sage dots a low one. Dots to the right raise predicted risk.</p>
<img src="{b64(gfig)}" alt="SHAP summary plot"><h3>Importance</h3><img src="{b64(ifig)}" alt="Feature importance bars">
<h3>Feature interactions</h3><img src="{b64(xfig)}" alt="SHAP interaction heatmap"><table><tr><th>Pair</th><th>Interaction strength</th></tr>{pair_rows}</table></section>
<section id="local"><h2>Local explanation for one declined applicant</h2>
<p>The force plot starts from the average risk and shows how each feature pushes this applicant's score up (pink) or down (sage).</p>
<img src="{b64(forcefig)}" alt="SHAP force plot">
<p>Top 3 adverse factors by SHAP: <b>{', '.join(s3)}</b>. By LIME: <b>{', '.join(l3)}</b>. Across 25 declined applicants, SHAP and LIME agreed on {agree:.0%} of the top-3 reasons on average.</p>
<img src="{b64(cmpfig)}" alt="SHAP versus LIME for one applicant"></section>
<section id="notice"><h2>Adverse Action Notice (generated)</h2><p>Sample notice produced automatically from the SHAP reasons above. Notices never mention protected characteristics.</p><pre>{notice_text}</pre></section>
<section id="audit"><h2>Regulatory compliance report</h2>
<h3>1. Disparate impact on approvals</h3><table><tr><th>Protected attribute</th><th>Approval rate by group</th><th>Lowest / highest</th><th>Result</th></tr>{di_rows}</table>
<h3>2. Proxy and correlation screen</h3>
<p>Each model feature is checked for correlation with protected attributes and for a gap in average SHAP contribution between men and women. Features flagged: <b>{', '.join(flagged) if flagged else 'none'}</b>.</p>
<table><tr><th>Feature</th><th>Importance</th><th>Corr. with gender (M=1)</th><th>Corr. with married</th><th>SHAP gap (M minus F)</th><th>Status</th></tr>{px_rows}</table>
<h3>3. Findings and recommendations</h3>
<p>The model never sees protected attributes, but a feature that correlates with one can act as a proxy. Where a feature is flagged, underwriters should confirm it has a documented, actuarially justified link to claims, and consider capping its influence or testing the model without it. Re-run this audit after every retraining.</p></section>
<section id="notes"><h2>Method notes</h2><p>Data are synthetic and generated with a known risk formula, including a young-driver by high-mileage interaction and a sports-car proxy for gender, so the audit has something real to find. SHAP values are in log-odds from TreeExplainer. A positive SHAP value is treated as an adverse factor. This is a teaching prototype and not legal advice.</p></section>
</main></div><footer>InsureXplain microproject | Module 1: XAI foundations | built {datetime.date.today():%B %Y}</footer></div></body></html>"""


# =============================================================================
def main():
    df = make_data()
    df.to_csv(os.path.join(OUT, "synthetic_applicants.csv"), index=False)
    model, Xtr, Xte, ytr, yte, idx_te, proba, metrics = train_model(df)
    explainer = shap.TreeExplainer(model)
    sv, imp, pairs, gfigs = explain_global(explainer, Xte)
    declined_pos = list(np.where(proba >= metrics["threshold"])[0])
    local = explain_local(explainer, model, Xtr, Xte, sv, declined_pos)
    notice = generate_notices(Xte, sv, declined_pos, idx_te)
    audit = compliance_audit(df, Xte, idx_te, proba, metrics["threshold"], sv, imp)
    audit["proxy"].to_csv(os.path.join(OUT, "proxy_audit.csv"), index=False)
    html = build_report(metrics, imp, pairs, (gfigs, local[4]), audit, local[:4], notice, df)
    with open(os.path.join(OUT, "InsureXplain_report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"AUC {metrics['auc']:.3f} | acc {metrics['accuracy']:.3f} | thr {metrics['threshold']:.3f}")
    print("Disparate impact:", audit["disparate_impact"])
    print("Top features:", list(imp.index[:4]), "| interactions:", [(a, b) for a, b, _ in pairs[:3]])
    print("SHAP/LIME top-3 agreement:", round(local[3], 2))


if __name__ == "__main__":
    main()
