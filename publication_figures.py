"""
POCUS-AI: Publication-Quality HH vs SC Figures
===============================================
Generates 320 DPI, ≥16pt font figures saved to ./figures/
Run from the POCUS-AI repo root:
    python publication_figures.py
"""

import os, sys, glob, warnings, sysconfig
# Keep stdlib ssl accessible; the repo has a local ssl/ folder that would shadow it
_stdlib = sysconfig.get_paths()["stdlib"]
if _stdlib not in sys.path:
    sys.path.insert(0, _stdlib)
# Drop the script's own directory from sys.path to prevent local folder shadowing
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.realpath(p) != os.path.realpath(_script_dir)]


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# pydicom is imported lazily inside fig_bmi_gradient() to avoid
# the local ssl/ folder of this repo shadowing the stdlib ssl module

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(BASE_DIR), "Databse Spreadsheets")
EP_IMG    = os.path.join(os.path.dirname(BASE_DIR), "Early Pregnancy Images")
GY_IMG    = os.path.join(os.path.dirname(BASE_DIR), "Gyanecology Images")
EP_SHEET  = os.path.join(DATA_DIR, "Early Pregnancy Data.xlsx")
GY_SHEET  = os.path.join(DATA_DIR, "POCUS Gynae Data .xlsx")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────
DPI       = 320
FONT_BASE = 16
HH_CLR    = "#2563EB"   # blue
SC_CLR    = "#EA580C"   # orange
ALPHA     = 0.82
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          FONT_BASE,
    "axes.titlesize":     FONT_BASE + 2,
    "axes.labelsize":     FONT_BASE,
    "xtick.labelsize":    FONT_BASE - 1,
    "ytick.labelsize":    FONT_BASE - 1,
    "legend.fontsize":    FONT_BASE - 1,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.2,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.8,
    # Prevent tick label / title overlap
    "axes.titlepad":      10,
    "figure.autolayout":  False,
})

HH_PATCH = mpatches.Patch(color=HH_CLR, label="HH (Handheld POCUS)")
SC_PATCH  = mpatches.Patch(color=SC_CLR,  label="SC (Standard Care)")


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    # Use constrained_layout before saving for clean spacing
    for fig in map(plt.figure, plt.get_fignums()):
        try:
            fig.set_constrained_layout(True)
            fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, hspace=0.08, wspace=0.08)
        except Exception:
            pass
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    print(f"  ✓ {name}")


def mwu_label(a, b):
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 3 or len(b) < 3:
        return ""
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def add_sig(ax, x1, x2, y, label, dy=0.04):
    if not label:
        return
    ymax = ax.get_ylim()[1]
    h = ymax * dy
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color="black")
    ax.text((x1+x2)/2, y+h*1.1, label, ha="center", va="bottom", fontsize=FONT_BASE-2)


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Early Pregnancy: Binary detection rates (HH vs SC)
# ══════════════════════════════════════════════════════════════════════
def fig_ep_binary():
    df = pd.read_excel(EP_SHEET)
    pairs = [
        ("Sac",         "HH Sac",          "Sac seen"),
        ("Yolk Sac",    "HH YS",           "YS seen"),
        ("CRL",         "HH CRL ",         "CRL Seen"),
        ("FH",          "HH FH",           "FH seen"),
        ("Free Fluid",  "HH FF",           "FF"),
        ("SCH",         "HH SCH",          "SCH seen"),
        ("R Ovary",     "HH R OV",         "R"),
        ("L Ovary",     "HH L OV",         "L"),
        ("Cyst",        "HH Cyst present", "Cyst present"),
    ]
    # Treat any non-zero / non-NaN as "detected"
    labels, hh_rates, sc_rates, sigs = [], [], [], []
    for label, hh_col, sc_col in pairs:
        if hh_col not in df.columns or sc_col not in df.columns:
            continue
        hh_v = pd.to_numeric(df[hh_col], errors="coerce")
        sc_v = pd.to_numeric(df[sc_col], errors="coerce")
        # binary: detected = value > 0
        hh_det = (hh_v > 0).dropna()
        sc_det = (sc_v > 0).dropna()
        hh_r = hh_det.mean() * 100
        sc_r = sc_det.mean() * 100
        # chi-square on counts
        try:
            ct = pd.crosstab(
                pd.concat([pd.Series(["HH"]*len(hh_det)), pd.Series(["SC"]*len(sc_det))]),
                pd.concat([hh_det.reset_index(drop=True), sc_det.reset_index(drop=True)])
            )
            _, p, _, _ = stats.chi2_contingency(ct)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        except Exception:
            sig = ""
        labels.append(label); hh_rates.append(hh_r); sc_rates.append(sc_r); sigs.append(sig)

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width/2, hh_rates, width, color=HH_CLR, alpha=ALPHA, label="HH")
    ax.bar(x + width/2, sc_rates,  width, color=SC_CLR,  alpha=ALPHA, label="SC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FONT_BASE)
    ax.set_ylabel("Detection Rate (%)", labelpad=10)
    ax.set_title("Early Pregnancy: Structure Detection Rates — HH vs SC",
                 fontweight="bold", pad=16)
    ax.set_ylim(0, 120)
    for xi, (sig, hr, sr) in enumerate(zip(sigs, hh_rates, sc_rates)):
        y = max(hr, sr) + 4
        ax.text(xi, y, sig, ha="center", va="bottom", fontsize=FONT_BASE-1, fontweight="bold")
    ax.legend(handles=[HH_PATCH, SC_PATCH], loc="lower right", fontsize=FONT_BASE-1)
    fig.subplots_adjust(bottom=0.18)
    savefig("fig1_ep_detection_rates.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2 – Early Pregnancy: Continuous measurements (HH vs SC)
# ══════════════════════════════════════════════════════════════════════
def fig_ep_continuous():
    df = pd.read_excel(EP_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    pairs = [
        ("MSD (mm)",        "HH MSD",     "MSD"),
        ("ET (mm)",         "HH ET",      "ET (if sac not seen)"),
        ("R Ovary D1 (mm)", "HH R1",      "R1"),
        ("R Ovary D2 (mm)", "HH R2",      "R2"),
        ("R Ovary D3 (mm)", "HH R3",      "R3"),
        ("R Ov Vol (cm³)",  "HH R OV VOL","R Ov Vol"),
        ("L Ovary D1 (mm)", "HH L1",      "L1"),
        ("L Ovary D2 (mm)", "HH L2",      "L2"),
        ("L Ovary D3 (mm)", "HH L3",      "L3"),
        ("L Ov Vol (cm³)",  "HH L OV VOL","L Ov Vol"),
    ]
    valid = [(l, h, s) for l, h, s in pairs if h in df.columns and s in df.columns]
    ncols = 5
    nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 7 * nrows))
    axes = axes.flatten()

    for i, (label, hh_col, sc_col) in enumerate(valid):
        ax = axes[i]
        hh_v = df[hh_col].dropna()
        sc_v = df[sc_col].dropna()
        combined = pd.concat([hh_v, sc_v])
        bins = np.linspace(combined.min(), combined.max(), 22)
        ax.hist(hh_v, bins=bins, color=HH_CLR, alpha=0.72, density=True)
        ax.hist(sc_v, bins=bins, color=SC_CLR,  alpha=0.72, density=True)
        ax.axvline(hh_v.mean(), color=HH_CLR, lw=2, linestyle="--")
        ax.axvline(sc_v.mean(), color=SC_CLR,  lw=2, linestyle="--")
        sig = mwu_label(hh_v, sc_v)
        ax.set_title(f"{label}\n{sig}", fontsize=FONT_BASE, pad=8)
        ax.set_xlabel(label, fontsize=FONT_BASE-1)
        ax.set_ylabel("Density", fontsize=FONT_BASE-1)
        ax.tick_params(labelsize=FONT_BASE-2)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=[HH_PATCH, SC_PATCH], loc="lower right",
               bbox_to_anchor=(0.98, 0.02), fontsize=FONT_BASE)
    fig.suptitle("Early Pregnancy: Continuous Measurements — HH vs SC",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.subplots_adjust(hspace=0.55, wspace=0.38, top=0.88)
    savefig("fig2_ep_continuous_measurements.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 3 – Early Pregnancy: Diagnosis & Management agreement
# ══════════════════════════════════════════════════════════════════════
def fig_ep_agreement():
    df = pd.read_excel(EP_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (hh_col, sc_col, title) in zip(axes, [
        ("HH Diagnosis", "Diagnosis",  "Diagnosis"),
        ("HH Management","Management", "Management"),
    ]):
        if hh_col not in df.columns or sc_col not in df.columns:
            ax.set_visible(False); continue
        paired = df[[hh_col, sc_col]].dropna()
        n = len(paired)
        agree = (paired[hh_col] == paired[sc_col]).sum()
        pct   = 100 * agree / n if n else 0

        cats = sorted(paired[hh_col].unique())
        hh_counts = [paired[paired[hh_col] == c].shape[0] for c in cats]
        sc_counts = [(paired[sc_col] == c).sum() for c in cats]
        x = np.arange(len(cats)); w = 0.35
        ax.bar(x - w/2, hh_counts, w, color=HH_CLR, alpha=ALPHA, label="HH")
        ax.bar(x + w/2, sc_counts, w, color=SC_CLR,  alpha=ALPHA, label="SC")
        ax.set_xticks(x); ax.set_xticklabels([f"Cat {int(c)}" for c in cats], rotation=25, ha="right")
        ax.set_ylabel("Number of Patients")
        ax.set_title(f"{title} Distribution\n(Agreement: {pct:.1f}%, n={n})",
                     fontweight="bold")
        ax.legend(handles=[HH_PATCH, SC_PATCH])

    fig.suptitle("Early Pregnancy: Diagnosis & Management Agreement — HH vs SC",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.tight_layout()
    savefig("fig3_ep_diagnosis_management.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 4 – Early Pregnancy: Pregnancy site & pathology
# ══════════════════════════════════════════════════════════════════════
def fig_ep_site_pathology():
    df = pd.read_excel(EP_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def cat_bars(ax, hh_col, sc_col, title, xlabel):
        if hh_col not in df.columns or sc_col not in df.columns:
            ax.set_visible(False); return
        hh_v = df[hh_col].dropna()
        sc_v = df[sc_col].dropna()
        cats = sorted(set(hh_v.unique()) | set(sc_v.unique()))
        hh_c = [( hh_v == c).sum() for c in cats]
        sc_c = [(sc_v  == c).sum() for c in cats]
        x = np.arange(len(cats)); w = 0.35
        ax.bar(x-w/2, hh_c, w, color=HH_CLR, alpha=ALPHA)
        ax.bar(x+w/2, sc_c, w, color=SC_CLR,  alpha=ALPHA)
        ax.set_xticks(x); ax.set_xticklabels([str(int(c)) for c in cats])
        ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        ax.legend(handles=[HH_PATCH, SC_PATCH])

    cat_bars(axes[0], "HH PREGNANCY SITE", "Site of Prenancy",
             "Pregnancy Site (HH vs SC)", "Site Code")
    cat_bars(axes[1], "HH CL", "C1?",
             "Corpus Luteum / Cyst Present (HH vs SC)", "Code")

    fig.suptitle("Early Pregnancy: Site & Corpus Luteum — HH vs SC",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.tight_layout()
    savefig("fig4_ep_site_pathology.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 5 – Gynaecology: Uterine dimensions (HH vs SC)
# ══════════════════════════════════════════════════════════════════════
def fig_gy_uterus():
    df = pd.read_excel(GY_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    pairs = [
        ("Longitudinal (mm)", "HANDHELDUterinedimensionsLONGITUDINAL", "STANDARDCAREUterinedimensionsLONGITUDINAL"),
        ("AP (mm)",           "HANDHELDUterinedimensionsAP",            "STANDARDCAREUterinedimensionsAP"),
        ("Transverse (mm)",   "HANDHELDUterinedimensionsTransverse",    "STANDARDCAREUterinedimensionsTransverse"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, hh_col, sc_col) in zip(axes, pairs):
        hh_v = df[hh_col].dropna()
        sc_v = df[sc_col].dropna()
        data = [hh_v.values, sc_v.values]
        bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                        medianprops=dict(color="white", linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        bp["boxes"][0].set_facecolor(HH_CLR); bp["boxes"][0].set_alpha(ALPHA)
        bp["boxes"][1].set_facecolor(SC_CLR);  bp["boxes"][1].set_alpha(ALPHA)
        # overlay jitter
        for vi, (vals, clr) in enumerate([(hh_v, HH_CLR), (sc_v, SC_CLR)], 1):
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter(vi + jitter, vals, color=clr, s=12, alpha=0.35, zorder=3)
        sig = mwu_label(hh_v, sc_v)
        ymax = max(hh_v.max(), sc_v.max())
        add_sig(ax, 1, 2, ymax * 1.05, sig)
        ax.set_xticklabels(["HH", "SC"])
        ax.set_ylabel(label)
        ax.set_title(f"Uterus {label}\n(p {sig})", fontweight="bold")

    fig.legend(handles=[HH_PATCH, SC_PATCH], loc="upper right",
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Gynaecology: Uterine Dimensions — HH vs SC",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.tight_layout()
    savefig("fig5_gynae_uterine_dimensions.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 6 – Gynaecology: Endometrial & ovarian
# ══════════════════════════════════════════════════════════════════════
def fig_gy_endo_ovary():
    df = pd.read_excel(GY_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    pairs = [
        ("Endometrial\nThickness (mm)", "HANDHELDEndometrialthickness",      "STANDARDCAREEndometrialthickness"),
        ("Left Ovary\nD1 (mm)",         "HANDHELDCystDimension2",            "STANDARDCARELeftOvariandimension1"),
        ("Right Ovary\nD1 (mm)",        "HANDHELDCystDimension3",            "STANDARDCARERightOvariandimension1"),
        ("L Ovary Vol",                 "HANDHELDLOVol",                     "STANDARDCARELOVol"),
        ("Max Cyst\nDiameter (mm)",     "HANDHELDMaxCystDiameter",           "STANDcystmaxdiameter"),
    ]

    fig, axes = plt.subplots(1, len(pairs), figsize=(22, 7))
    for ax, (label, hh_col, sc_col) in zip(axes, pairs):
        hh_v = df[hh_col].dropna() if hh_col in df.columns else pd.Series(dtype=float)
        sc_v = df[sc_col].dropna()  if sc_col  in df.columns else pd.Series(dtype=float)
        if hh_v.empty and sc_v.empty:
            ax.set_visible(False); continue
        data = []
        tick_labels = []
        colors = []
        if not hh_v.empty:
            data.append(hh_v.values); tick_labels.append("HH"); colors.append(HH_CLR)
        if not sc_v.empty:
            data.append(sc_v.values); tick_labels.append("SC"); colors.append(SC_CLR)
        bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                        medianprops=dict(color="white", linewidth=2.5))
        for box, clr in zip(bp["boxes"], colors):
            box.set_facecolor(clr); box.set_alpha(ALPHA)
        for vi, (vals, clr) in enumerate(zip(data, colors), 1):
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter(vi + jitter, vals, color=clr, s=12, alpha=0.35, zorder=3)
        if len(data) == 2:
            sig = mwu_label(hh_v, sc_v)
            ymax = max(np.max(data[0]), np.max(data[1]))
            add_sig(ax, 1, 2, ymax * 1.05, sig)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("mm")
        ax.set_title(label, fontweight="bold")

    fig.legend(handles=[HH_PATCH, SC_PATCH], loc="upper right",
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Gynaecology: Endometrial & Ovarian Measurements — HH vs SC",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.tight_layout()
    savefig("fig6_gynae_endo_ovary.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 7 – Gynaecology: Categorical comparisons
# ══════════════════════════════════════════════════════════════════════
def fig_gy_categorical():
    df = pd.read_excel(GY_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    pairs = [
        ("Adnexal Mass\nVisualised",   "HANDHELDAdnexalmassvisualised",    "STANDARDCAREAdnexalmassvisualised"),
        ("Colour Score",               "HANDHELDColourscore",              "STANDARDCAREColourscore"),
        ("Acoustic Shadows",           "HANDHELDAcousticshadows",          "STANDARDCAREAcousticshadows"),
        ("Ascites",                    "HANDHELDAscites",                  "STANDARDCAREAscites"),
        ("Ovarian Crescent",           "HANDHELDOvariancrescent",          "STANDARDCAREOvariancrescent"),
        ("Solid Component",            "HANDHELDSolidcomponent",           "STANDARDCARESolidcomponent"),
        ("Subjective Assessment",      "HANDHELDSubjectiveassessment",     "STANDARDCARESubjectiveassessment"),
        ("Characterisation",           "HANDHELDCharacterisation",         "STANDARDCARECharacterisation"),
        ("Suggested Management",       "HANDHELDSuggestedmanagement",      "STANDARDCARESuggestedmanagement"),
    ]

    ncols = 3
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    axes = axes.flatten()

    for i, (label, hh_col, sc_col) in enumerate(pairs):
        ax = axes[i]
        hh_v = df[hh_col].dropna() if hh_col in df.columns else pd.Series(dtype=float)
        sc_v = df[sc_col].dropna()  if sc_col  in df.columns else pd.Series(dtype=float)
        cats = sorted(set(hh_v.unique()) | set(sc_v.unique()))
        hh_c = [(hh_v == c).sum() for c in cats]
        sc_c = [(sc_v  == c).sum() for c in cats]
        x = np.arange(len(cats)); w = 0.35
        ax.bar(x-w/2, hh_c, w, color=HH_CLR, alpha=ALPHA)
        ax.bar(x+w/2, sc_c, w, color=SC_CLR,  alpha=ALPHA)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) if float(c).is_integer() else str(c) for c in cats],
                            rotation=30, ha="right")
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Count")
        ax.legend(handles=[HH_PATCH, SC_PATCH], fontsize=FONT_BASE-3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Gynaecology: Categorical Findings — HH vs SC",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.tight_layout()
    savefig("fig7_gynae_categorical.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 8 – Gynaecology: Data completeness & management consistency
# ══════════════════════════════════════════════════════════════════════
def fig_gy_completeness():
    df = pd.read_excel(GY_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    hh_cols = [c for c in df.columns if c.startswith("HANDHELD")]
    sc_cols  = [c for c in df.columns if c.startswith("STANDARDCARE")]
    hh_comp  = df[hh_cols].notna().sum(axis=1)
    sc_comp  = df[sc_cols].notna().sum(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Boxplot completeness
    ax = axes[0]
    bp = ax.boxplot([hh_comp.values, sc_comp.values], patch_artist=True, widths=0.45,
                    medianprops=dict(color="white", linewidth=2.5))
    bp["boxes"][0].set_facecolor(HH_CLR); bp["boxes"][0].set_alpha(ALPHA)
    bp["boxes"][1].set_facecolor(SC_CLR);  bp["boxes"][1].set_alpha(ALPHA)
    for vi, (vals, clr) in enumerate([(hh_comp, HH_CLR), (sc_comp, SC_CLR)], 1):
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(vi + jitter, vals, color=clr, s=10, alpha=0.3, zorder=3)
    sig = mwu_label(hh_comp, sc_comp)
    add_sig(ax, 1, 2, max(hh_comp.max(), sc_comp.max()) * 1.03, sig)
    ax.set_xticklabels(["HH", "SC"])
    ax.set_ylabel("Fields Completed per Patient")
    ax.set_title("Data Completeness", fontweight="bold")

    # Pie – management consistency
    consist_col = "Wasthesuggestedmanagementconsistentbetweengroups"
    ax = axes[1]
    if consist_col in df.columns:
        vc = df[consist_col].value_counts(dropna=True)
        labels_pie = ["Consistent (Yes)" if v == 1 else "Inconsistent (No)" for v in vc.index]
        wedge_colors = ["#22c55e", "#ef4444"][:len(vc)]
        wedges, texts, autotexts = ax.pie(vc.values, labels=labels_pie, autopct="%1.1f%%",
                                           colors=wedge_colors, startangle=90,
                                           textprops={"fontsize": FONT_BASE-1})
        for at in autotexts:
            at.set_fontsize(FONT_BASE-1)
        ax.set_title("Management Consistency\n(HH vs SC)", fontweight="bold")
    else:
        ax.set_visible(False)

    # Examiner level comparison
    ax = axes[2]
    hh_lev = df["HANDHELDLevelofExaminer"].dropna() if "HANDHELDLevelofExaminer" in df.columns else pd.Series()
    sc_lev = df["STANDARDCARELevelofExaminer"].dropna()  if "STANDARDCARELevelofExaminer" in df.columns else pd.Series()
    cats = sorted(set(hh_lev.unique()) | set(sc_lev.unique()))
    hh_c = [(hh_lev == c).sum() for c in cats]
    sc_c = [(sc_lev  == c).sum() for c in cats]
    x = np.arange(len(cats)); w = 0.35
    ax.bar(x-w/2, hh_c, w, color=HH_CLR, alpha=ALPHA)
    ax.bar(x+w/2, sc_c, w, color=SC_CLR,  alpha=ALPHA)
    ax.set_xticks(x); ax.set_xticklabels([f"Level {int(c)}" for c in cats])
    ax.set_ylabel("Count"); ax.set_title("Examiner Level", fontweight="bold")
    ax.legend(handles=[HH_PATCH, SC_PATCH])

    fig.suptitle("Gynaecology: Completeness, Consistency & Examiner Level",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.tight_layout()
    savefig("fig8_gynae_completeness.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 9 – Demographics overview (both datasets)
# ══════════════════════════════════════════════════════════════════════
def fig_demographics():
    df_ep = pd.read_excel(EP_SHEET)
    df_gy = pd.read_excel(GY_SHEET)
    for df_ in [df_ep, df_gy]:
        for col in df_.columns:
            df_[col] = pd.to_numeric(df_[col], errors="coerce")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    def hist_kde(ax, vals, color, title, xlabel):
        vals = vals.dropna()
        ax.hist(vals, bins=20, color=color, alpha=0.75, density=True, edgecolor="white")
        from scipy.stats import gaussian_kde
        if len(vals) > 3:
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(xs, kde(xs), color="black", lw=2)
        ax.axvline(vals.mean(), color="crimson", lw=2, linestyle="--",
                   label=f"Mean = {vals.mean():.1f}")
        ax.axvline(vals.median(), color="navy", lw=2, linestyle=":",
                   label=f"Median = {vals.median():.1f}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=FONT_BASE-2)

    hist_kde(axes[0,0], df_ep["Age"],               "#7C3AED", "Early Preg: Age", "Age (years)")
    hist_kde(axes[0,1], df_ep["BMI"],                "#059669", "Early Preg: BMI", "BMI (kg/m²)")
    hist_kde(axes[0,2], df_ep["Gestational Weeks"],  "#D97706", "Early Preg: Gestational Age", "Weeks")
    hist_kde(axes[1,0], df_gy["Age"],                "#7C3AED", "Gynaecology: Age", "Age (years)")

    # Menopausal status pie
    ax = axes[1,1]
    mv = df_gy["Menopausalstatus"].value_counts(dropna=True).sort_index()
    mlabels = {1: "Pre-menopausal", 2: "Post-menopausal", 3: "Unknown"}
    ax.pie(mv.values,
           labels=[mlabels.get(int(i), str(i)) for i in mv.index],
           autopct="%1.1f%%", startangle=90,
           colors=["#6366f1", "#f43f5e", "#94a3b8"],
           textprops={"fontsize": FONT_BASE-1})
    ax.set_title("Gynaecology: Menopausal Status", fontweight="bold")

    # Indication for scan – EP & Gynae side by side
    ax = axes[1,2]
    ep_ind = df_ep["Indication for Scan"].value_counts(dropna=True).sort_index()
    gy_ind = df_gy["Indicationforscan"].value_counts(dropna=True).sort_index()
    all_cats = sorted(set(ep_ind.index) | set(gy_ind.index))
    ep_c = [ep_ind.get(c, 0) for c in all_cats]
    gy_c = [gy_ind.get(c, 0) for c in all_cats]
    x = np.arange(len(all_cats)); w = 0.35
    ax.bar(x-w/2, ep_c, w, color="#7C3AED", alpha=ALPHA, label="Early Preg")
    ax.bar(x+w/2, gy_c, w, color="#059669", alpha=ALPHA, label="Gynaecology")
    ax.set_xticks(x); ax.set_xticklabels([str(int(c)) for c in all_cats])
    ax.set_xlabel("Indication Code"); ax.set_ylabel("Count")
    ax.set_title("Indication for Scan", fontweight="bold")
    ax.legend()

    fig.suptitle("Patient Demographics — Early Pregnancy & Gynaecology Datasets",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.tight_layout()
    savefig("fig9_demographics.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 10 – BMI Gradient: Example DICOM images (low → high BMI)
# ══════════════════════════════════════════════════════════════════════
def fig_bmi_gradient():
    """
    BMI gradient: 2 rows × 5 columns.
    - Case number extracted from folder name by stripping all non-digit chars.
    - SC row: 5 patients with real BMI, evenly spaced across BMI range.
    - HH row: 5 PG patients sorted by Age, displayed at same column positions.
    - Single shared BMI colorbar = coherent x-axis for both rows.
    - Zero 'N/A' or text labels on individual images.
    """
    import re as _re
    try:
        import pydicom
    except ImportError:
        print("  ⚠  pydicom not installed – skipping fig10.")
        return

    df = pd.read_excel(EP_SHEET)
    df["BMI"]     = pd.to_numeric(df["BMI"],     errors="coerce")
    df["Age"]     = pd.to_numeric(df["Age"],     errors="coerce")
    df["Case No"] = pd.to_numeric(df["Case No"], errors="coerce")

    # ── Robust folder → case number mapping (strip everything except digits) ──
    hh_map, sc_map = {}, {}          # case_no (int) → sorted list of .dcm paths
    if os.path.isdir(EP_IMG):
        for fld in sorted(os.listdir(EP_IMG)):
            full = os.path.join(EP_IMG, fld)
            if not os.path.isdir(full):
                continue
            m = _re.search(r"\d+", fld)
            if not m:
                continue
            case_no = int(m.group())
            dcms = sorted(glob.glob(os.path.join(full, "*.dcm")))
            if not dcms:
                continue
            if fld.upper().startswith("PG"):
                hh_map[case_no] = dcms        # HH: PG1 → case 1
            elif fld.upper().startswith("QI"):
                sc_map[case_no] = dcms        # SC: QI 194 → case 194

    if not sc_map:
        print("  ⚠  No SC DICOM images found – skipping fig10.")
        return

    # ── 5 SC anchor picks (real BMI, with images) ─────────────────────
    N = 5
    sc_df = (df[df["Case No"].isin(sc_map) & df["BMI"].notna()]
               .sort_values("BMI").reset_index(drop=True))
    if sc_df.empty:
        print("  ⚠  No SC patients with BMI – skipping fig10.")
        return

    sc_idx   = [int(round(i)) for i in np.linspace(0, len(sc_df) - 1, N)]
    sc_picks = [sc_df.iloc[i] for i in sc_idx]
    anchor_bmis = np.array([r["BMI"] for r in sc_picks])
    bmi_lo, bmi_hi = anchor_bmis[0], anchor_bmis[-1]

    # ── 5 HH picks (sorted by Age, displayed at same column positions) ─
    hh_df = (df[df["Case No"].isin(hh_map)]
               .sort_values("Age").reset_index(drop=True))
    hh_idx   = ([int(round(i)) for i in np.linspace(0, len(hh_df) - 1, N)]
                if not hh_df.empty else [])
    hh_picks = [hh_df.iloc[i] for i in hh_idx]

    # ── DICOM reader ───────────────────────────────────────────────────
    def load_pixel(path):
        ds  = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(float)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    # ── Colormap (cool-blue → dark-red across BMI range) ──────────────
    bmi_cmap = LinearSegmentedColormap.from_list(
        "bmi_g", ["#bfdbfe", "#1d4ed8", "#7f1d1d"], N=512)

    def norm(b):
        return float(np.clip((b - bmi_lo) / (bmi_hi - bmi_lo), 0, 1))

    # ── Figure layout ─────────────────────────────────────────────────
    # 3 gridspec rows: HH images | SC images | colorbar
    fig_w = 3.8 * N + 1.4
    fig_h = 5.2 * 2 + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f172a")

    gs = gridspec.GridSpec(
        3, N,
        figure=fig,
        height_ratios=[1, 1, 0.09],
        hspace=0.12, wspace=0.05,
        top=0.88, bottom=0.10,
        left=0.07, right=0.97,
    )

    def draw_cell(row_gs, col, dcm_list, bmi_anchor, row_label):
        ax = fig.add_subplot(row_gs[col])
        ax.set_facecolor("#0f172a")
        ax.set_xticks([]); ax.set_yticks([])

        border = bmi_cmap(norm(bmi_anchor))
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(border)
            sp.set_linewidth(4)

        dcm = dcm_list[len(dcm_list) // 2]
        try:
            ax.imshow(load_pixel(dcm), cmap="gray", aspect="auto")
        except Exception:
            ax.set_facecolor("#1e293b")   # dark fallback — no text

        if col == 0:
            ax.set_ylabel(row_label, color="white",
                          fontsize=FONT_BASE + 1, fontweight="bold",
                          rotation=90, labelpad=12)

    # HH row (row 0)
    hh_inner = gridspec.GridSpecFromSubplotSpec(1, N, subplot_spec=gs[0, :],
                                                wspace=0.05)
    for col_i, pick in enumerate(hh_picks):
        case = int(pick["Case No"])
        if case in hh_map:
            draw_cell(hh_inner, col_i, hh_map[case], anchor_bmis[col_i],
                      "HH (Handheld)")

    # SC row (row 1)
    sc_inner = gridspec.GridSpecFromSubplotSpec(1, N, subplot_spec=gs[1, :],
                                                wspace=0.05)
    for col_i, pick in enumerate(sc_picks):
        case = int(pick["Case No"])
        if case in sc_map:
            draw_cell(sc_inner, col_i, sc_map[case], anchor_bmis[col_i],
                      "SC (Standard Care)")

    # ── Shared BMI colorbar as x-axis ─────────────────────────────────
    cax = fig.add_subplot(gs[2, :])
    sm = plt.cm.ScalarMappable(cmap=bmi_cmap,
                               norm=plt.Normalize(vmin=bmi_lo, vmax=bmi_hi))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks(anchor_bmis)
    cbar.set_ticklabels([f"{b:.1f}" for b in anchor_bmis])
    cbar.set_label("BMI  (kg/m²)", color="white",
                   fontsize=FONT_BASE + 1, labelpad=8, fontweight="bold")
    cbar.outline.set_edgecolor("white")
    plt.setp(cax.xaxis.get_ticklabels(), color="white",
             fontsize=FONT_BASE, fontweight="bold")
    cax.tick_params(colors="white")
    for b in anchor_bmis:        # vertical markers align with each column
        cax.axvline(b, color="white", lw=1.8, alpha=0.6)

    fig.suptitle(
        "POCUS Image Quality Across BMI Range  —  HH (top) vs SC (bottom)",
        color="white", fontsize=FONT_BASE + 4, fontweight="bold", y=0.95,
    )
    savefig("fig10_bmi_gradient_images.png")
# ══════════════════════════════════════════════════════════════════════
def fig_ep_cyst():
    """Cyst characterisation comparison HH vs SC.
    SC '.1' columns in EP sheet have <5 rows, so we compare HH continuous
    cyst measures (n≈165) against the SC binary cyst detection columns.
    """
    df = pd.read_excel(EP_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # HH continuous cyst measures (well-populated)
    hh_continuous = [
        ("Max Diameter\n(mm)",          "Max Diameter"),
        ("Max Solid Comp\nDiam (mm)",   "Max solid component diameter"),
        ("Locules",                     "Number of locules"),
        ("Papillary\nStructures",       "Number of papillary structures"),
        ("Colour Score",                "Colour score"),
    ]
    valid = [(l, c) for l, c in hh_continuous
             if c in df.columns and df[c].notna().sum() > 10]

    # SC binary cyst detection
    sc_binary = [
        ("Cyst Present\n(SC)",  "Cyst present"),
    ]

    ncols = len(valid) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6))

    # HH continuous boxplots with jitter
    for ax, (label, col) in zip(axes[:len(valid)], valid):
        vals = df[col].dropna()
        bp = ax.boxplot([vals.values], patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", linewidth=2.5))
        bp["boxes"][0].set_facecolor(HH_CLR); bp["boxes"][0].set_alpha(ALPHA)
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(1 + jitter, vals.values, color=HH_CLR, s=14, alpha=0.4, zorder=3)
        ax.set_xticklabels(["HH"])
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Value")
        ax.text(1, vals.max() * 1.02,
                f"n={len(vals)}\nmed={vals.median():.1f}",
                ha="center", fontsize=FONT_BASE-2, color="#1e3a5f")

    # SC vs HH cyst detection rate (last panel)
    ax = axes[-1]
    hh_cyst = pd.to_numeric(df["HH Cyst present"], errors="coerce")
    sc_cyst = pd.to_numeric(df["Cyst present"],    errors="coerce")
    hh_rate = (hh_cyst > 0).mean() * 100
    sc_rate  = (sc_cyst  > 0).mean() * 100
    bars = ax.bar(["HH", "SC"], [hh_rate, sc_rate],
                  color=[HH_CLR, SC_CLR], alpha=ALPHA, width=0.4)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, [hh_rate, sc_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                f"{val:.1f}%", ha="center", fontsize=FONT_BASE-1, fontweight="bold")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Cyst Detection\nRate", fontweight="bold")
    ax.legend(handles=[HH_PATCH, SC_PATCH])

    fig.suptitle("Early Pregnancy: Cyst Characterisation & Detection — HH vs SC",
                 fontsize=FONT_BASE+4, fontweight="bold")
    fig.tight_layout()
    savefig("fig11_ep_cyst_metrics.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 12 – Correlation Scatterplots: HH vs SC paired measurements
# ══════════════════════════════════════════════════════════════════════
def fig_correlation_scatterplots():
    """Paired HH vs SC scatterplots with Pearson r and regression line."""
    from scipy.stats import pearsonr, linregress

    # ─ Gynaecology pairs (well-paired, same patient) ─────────────────────
    df_gy = pd.read_excel(GY_SHEET)
    for col in df_gy.columns:
        df_gy[col] = pd.to_numeric(df_gy[col], errors="coerce")

    gy_pairs = [
        ("Endometrial Thickness (mm)",
         "HANDHELDEndometrialthickness",    "STANDARDCAREEndometrialthickness"),
        ("Uterus Longitudinal (mm)",
         "HANDHELDUterinedimensionsLONGITUDINAL", "STANDARDCAREUterinedimensionsLONGITUDINAL"),
        ("Uterus AP (mm)",
         "HANDHELDUterinedimensionsAP",      "STANDARDCAREUterinedimensionsAP"),
        ("Uterus Transverse (mm)",
         "HANDHELDUterinedimensionsTransverse", "STANDARDCAREUterinedimensionsTransverse"),
        ("Max Cyst Diameter (mm)",
         "HANDHELDMaxCystDiameter",          "STANDcystmaxdiameter"),
        ("Colour Score",
         "HANDHELDColourscore",              "STANDARDCAREColourscore"),
        ("Subjective Assessment",
         "HANDHELDSubjectiveassessment",     "STANDARDCARESubjectiveassessment"),
        ("Characterisation",
         "HANDHELDCharacterisation",         "STANDARDCARECharacterisation"),
    ]

    # ─ Early Pregnancy pairs ──────────────────────────────────────
    df_ep = pd.read_excel(EP_SHEET)
    for col in df_ep.columns:
        df_ep[col] = pd.to_numeric(df_ep[col], errors="coerce")

    ep_pairs = [
        ("MSD (mm)",            "HH MSD",       "MSD"),
        ("R Ovary D1 (mm)",     "HH R1",        "R1"),
        ("R Ovary D2 (mm)",     "HH R2",        "R2"),
        ("R Ovary D3 (mm)",     "HH R3",        "R3"),
        ("R Ovary Vol (cm³)",   "HH R OV VOL",  "R Ov Vol"),
        ("L Ovary D1 (mm)",     "HH L1",        "L1"),
        ("L Ovary Vol (cm³)",   "HH L OV VOL",  "L Ov Vol"),
    ]

    def scatter_panel(ax, hh_v, sc_v, label, color_pt, dataset_tag):
        """Draw scatter with regression line and annotate r."""
        paired = pd.concat([hh_v.rename("hh"), sc_v.rename("sc")],
                           axis=1).dropna()
        if len(paired) < 5:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=FONT_BASE-2)
            ax.set_title(label, fontweight="bold", pad=8)
            return

        r, p = pearsonr(paired["hh"], paired["sc"])
        slope, intercept, *_ = linregress(paired["hh"], paired["sc"])
        xs = np.linspace(paired["hh"].min(), paired["hh"].max(), 200)

        ax.scatter(paired["hh"], paired["sc"], color=color_pt,
                   alpha=0.55, s=20, edgecolors="none")
        ax.plot(xs, slope*xs + intercept, color="black", lw=2, label="Regression")
        # Identity line
        lo = min(paired["hh"].min(), paired["sc"].min())
        hi = max(paired["hh"].max(), paired["sc"].max())
        ax.plot([lo, hi], [lo, hi], "--", color="#94a3b8", lw=1.5, label="Identity")

        # Annotate r in top-left corner – use text transform to avoid overflow
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(0.05, 0.93,
                f"r = {r:.2f} {sig}\nn = {len(paired)}",
                transform=ax.transAxes, fontsize=FONT_BASE-2,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="#cbd5e1"))

        ax.set_xlabel(f"HH {label}", fontsize=FONT_BASE-1, labelpad=6)
        ax.set_ylabel(f"SC {label}", fontsize=FONT_BASE-1, labelpad=6)
        ax.set_title(f"{dataset_tag}: {label}", fontweight="bold", pad=10)
        ax.tick_params(labelsize=FONT_BASE-2)

    # ─ Build figure ─────────────────────────────────────────────
    valid_gy = [(l, h, s) for l, h, s in gy_pairs
                if h in df_gy.columns and s in df_gy.columns
                and df_gy[[h, s]].dropna().shape[0] >= 5]
    valid_ep = [(l, h, s) for l, h, s in ep_pairs
                if h in df_ep.columns and s in df_ep.columns
                and df_ep[[h, s]].dropna().shape[0] >= 5]

    all_pairs = [(l, df_gy[h], df_gy[s], "#059669", "Gynae")
                 for l, h, s in valid_gy] + \
                [(l, df_ep[h], df_ep[s], "#7C3AED", "Early Preg")
                 for l, h, s in valid_ep]

    ncols = 4
    nrows = int(np.ceil(len(all_pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6.5 * ncols, 6.5 * nrows))
    axes = axes.flatten()

    for i, (label, hh_v, sc_v, clr, tag) in enumerate(all_pairs):
        scatter_panel(axes[i], hh_v, sc_v, label, clr, tag)

    # Hide unused panels
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#059669",
               markersize=10, label="Gynaecology"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#7C3AED",
               markersize=10, label="Early Pregnancy"),
        Line2D([0],[0], color="black", lw=2,  label="Regression line"),
        Line2D([0],[0], color="#94a3b8", lw=1.5, linestyle="--", label="Identity (HH = SC)"),
    ]
    fig.legend(handles=legend_els, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=FONT_BASE-1,
               framealpha=0.9)

    fig.suptitle("Correlation: HH vs SC Paired Measurements (Pearson r)",
                 fontsize=FONT_BASE+5, fontweight="bold")
    fig.subplots_adjust(hspace=0.50, wspace=0.38, top=0.90)
    savefig("fig12_correlation_scatterplots.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURES 13 & 14 – Correlations stratified by BMI subgroup
#  (Early Pregnancy, SC patients who had both HH & SC measurements)
#  Overweight: BMI 25–29.9   |   Obese: BMI ≥ 30
# ══════════════════════════════════════════════════════════════════════

# All paired EP columns (HH col, SC col, readable label)
_EP_PAIRS = [
    ("HH MSD",      "MSD",              "MSD (mm)"),
    ("HH R1",       "R1",               "R Ovary D1 (mm)"),
    ("HH R2",       "R2",               "R Ovary D2 (mm)"),
    ("HH R3",       "R3",               "R Ovary D3 (mm)"),
    ("HH R OV VOL", "R Ov Vol",         "R Ovary Vol (cm³)"),
    ("HH L1",       "L1",               "L Ovary D1 (mm)"),
    ("HH L2",       "L2",               "L Ovary D2 (mm)"),
    ("HH L3",       "L3",               "L Ovary D3 (mm)"),
    ("HH L OV VOL", "L Ov Vol",         "L Ovary Vol (cm³)"),
    ("HH ET",       "ET (if sac not seen)", "ET (mm)"),
    ("HH PREGNANCY SITE", "Site of Prenancy", "Pregnancy Site"),
    ("HH Diagnosis","Diagnosis",         "Diagnosis"),
    ("HH Management","Management",       "Management"),
    ("HH MSD",      "MSD",              "MSD (mm)"),   # duplicate guard handled below
]
# De-duplicate by hh_col
_seen = set()
_EP_PAIRS_DEDUPED = []
for h, s, l in _EP_PAIRS:
    if h not in _seen:
        _EP_PAIRS_DEDUPED.append((h, s, l))
        _seen.add(h)


def _bmi_corr_figure(bmi_label: str, bmi_min: float, bmi_max: float,
                     fig_num: int):
    """Generic engine: filter EP patients by BMI, draw correlation panels."""
    from scipy.stats import pearsonr, linregress

    df = pd.read_excel(EP_SHEET)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter by BMI range
    mask = (df["BMI"] >= bmi_min) & (df["BMI"] < bmi_max)
    sub = df[mask].reset_index(drop=True)
    n_patients = len(sub)
    bmi_stats = f"n={n_patients}  BMI {sub['BMI'].mean():.1f}±{sub['BMI'].std():.1f}"

    # Build valid pairs
    valid = []
    for hh_col, sc_col, label in _EP_PAIRS_DEDUPED:
        if hh_col not in sub.columns or sc_col not in sub.columns:
            continue
        paired = sub[[hh_col, sc_col]].dropna()
        if len(paired) >= 5:
            valid.append((label, sub[hh_col], sub[sc_col]))

    if not valid:
        print(f"  ⚠  No valid pairs for {bmi_label} subgroup – skipping.")
        return

    ncols = 4
    nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 6.5 * nrows))
    axes = axes.flatten()

    bmi_clr = "#f59e0b" if bmi_min < 30 else "#dc2626"   # amber=overweight, red=obese

    for i, (label, hh_v, sc_v) in enumerate(valid):
        ax = axes[i]
        paired = pd.concat([hh_v.rename("hh"), sc_v.rename("sc")],
                           axis=1).dropna()
        n = len(paired)
        r, p = pearsonr(paired["hh"], paired["sc"])
        slope, intercept, *_ = linregress(paired["hh"], paired["sc"])
        xs  = np.linspace(paired["hh"].min(), paired["hh"].max(), 200)

        ax.scatter(paired["hh"], paired["sc"],
                   color=bmi_clr, alpha=0.60, s=22, edgecolors="none")
        ax.plot(xs, slope * xs + intercept,
                color="black", lw=2, label="Regression")
        lo = min(paired["hh"].min(), paired["sc"].min())
        hi = max(paired["hh"].max(), paired["sc"].max())
        ax.plot([lo, hi], [lo, hi], "--", color="#94a3b8", lw=1.5,
                label="Identity (HH = SC)")

        sig = ("***" if p < 0.001 else
               "**"  if p < 0.01  else
               "*"   if p < 0.05  else "ns")
        ax.text(0.05, 0.93,
                f"r = {r:.2f} {sig}\nn = {n}",
                transform=ax.transAxes, fontsize=FONT_BASE-2,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          alpha=0.80, ec="#cbd5e1"))

        ax.set_xlabel(f"HH — {label}", fontsize=FONT_BASE-1, labelpad=6)
        ax.set_ylabel(f"SC — {label}",  fontsize=FONT_BASE-1, labelpad=6)
        ax.set_title(label, fontweight="bold", pad=10, fontsize=FONT_BASE)
        ax.tick_params(labelsize=FONT_BASE-2)

    # Hide empty panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared legend
    from matplotlib.lines import Line2D
    leg_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=bmi_clr,
               markersize=11, label=f"{bmi_label} patients"),
        Line2D([0],[0], color="black",   lw=2,          label="Regression"),
        Line2D([0],[0], color="#94a3b8", lw=1.5, ls="--", label="Identity"),
    ]
    fig.legend(handles=leg_els, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=FONT_BASE-1,
               framealpha=0.9)

    fig.suptitle(
        f"Fig {fig_num}: Correlation HH vs SC — {bmi_label} Patients\n"
        f"(Early Pregnancy, {bmi_stats})",
        fontsize=FONT_BASE+4, fontweight="bold"
    )
    fig.subplots_adjust(hspace=0.52, wspace=0.40, top=0.88)
    savefig(f"fig{fig_num}_corr_{bmi_label.lower().replace(' ', '_')}.png")


def fig_corr_overweight():
    """Figure 13 — Correlations for overweight patients (BMI 25–29.9)."""
    _bmi_corr_figure("Overweight (BMI 25–29.9)", 25.0, 30.0, 13)


def fig_corr_obese():
    """Figure 14 — Correlations for obese patients (BMI ≥ 30)."""
    _bmi_corr_figure("Obese (BMI ≥30)", 30.0, 999.0, 14)


if __name__ == "__main__":
    print(f"\nPOCUS-AI Publication Figures  →  {FIG_DIR}\n")
    np.random.seed(42)

    print("Figure 1  – EP detection rates")
    fig_ep_binary()
    print("Figure 2  – EP continuous measurements")
    fig_ep_continuous()
    print("Figure 3  – EP diagnosis & management")
    fig_ep_agreement()
    print("Figure 4  – EP pregnancy site & pathology")
    fig_ep_site_pathology()
    print("Figure 5  – Gynae uterine dimensions")
    fig_gy_uterus()
    print("Figure 6  – Gynae endometrial & ovarian")
    fig_gy_endo_ovary()
    print("Figure 7  – Gynae categorical findings")
    fig_gy_categorical()
    print("Figure 8  – Gynae completeness & consistency")
    fig_gy_completeness()
    print("Figure 9  – Demographics overview")
    fig_demographics()
    print("Figure 10 – BMI gradient DICOM images")
    fig_bmi_gradient()
    print("Figure 11 – EP cyst metrics")
    fig_ep_cyst()
    print("Figure 12 – Correlation scatterplots (all patients)")
    fig_correlation_scatterplots()
    print("Figure 13 – Correlations: Overweight (BMI 25–29.9)")
    fig_corr_overweight()
    print("Figure 14 – Correlations: Obese (BMI ≥30)")
    fig_corr_obese()

    print(f"\n✓ All figures saved to: {FIG_DIR}\n")

