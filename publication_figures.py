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
})

HH_PATCH = mpatches.Patch(color=HH_CLR, label="HH (Handheld POCUS)")
SC_PATCH  = mpatches.Patch(color=SC_CLR,  label="SC (Standard Care)")


def savefig(name):
    path = os.path.join(FIG_DIR, name)
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
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - width/2, hh_rates, width, color=HH_CLR, alpha=ALPHA, label="HH")
    b2 = ax.bar(x + width/2, sc_rates,  width, color=SC_CLR,  alpha=ALPHA, label="SC")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Early Pregnancy: Structure Detection Rates — HH vs SC", fontweight="bold", pad=14)
    ax.set_ylim(0, 115)
    for xi, sig in zip(x, sigs):
        y = max(hh_rates[xi], sc_rates[xi]) + 3
        ax.text(xi, y, sig, ha="center", va="bottom", fontsize=FONT_BASE-2, fontweight="bold")
    ax.legend(handles=[HH_PATCH, SC_PATCH], loc="lower right")
    fig.tight_layout()
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5.5 * nrows))
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
        ax.set_title(f"{label}  {sig}", fontsize=FONT_BASE)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=[HH_PATCH, SC_PATCH], loc="lower right",
               bbox_to_anchor=(0.98, 0.02), fontsize=FONT_BASE)
    fig.suptitle("Early Pregnancy: Continuous Measurements — HH vs SC",
                 fontsize=FONT_BASE+4, fontweight="bold", y=1.01)
    fig.tight_layout()
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
    try:
        import pydicom
    except ImportError:
        print("  ⚠  pydicom not installed – skipping BMI image gradient figure.")
        print("     Install with: pip install pydicom")
        return

    df_ep = pd.read_excel(EP_SHEET)
    df_ep["BMI"] = pd.to_numeric(df_ep["BMI"], errors="coerce")
    df_ep["Case No"] = pd.to_numeric(df_ep["Case No"], errors="coerce")

    # Build map: case number → folder in EP_IMG
    # HH folders: PG1, PG2, … SC folders: QI 194, QI 195, …
    hh_map = {}  # case_no → folder path
    sc_map = {}
    if os.path.isdir(EP_IMG):
        for fld in sorted(os.listdir(EP_IMG)):
            full = os.path.join(EP_IMG, fld)
            if not os.path.isdir(full):
                continue
            dcms = sorted(glob.glob(os.path.join(full, "*.dcm")))
            if not dcms:
                continue
            if fld.startswith("PG"):
                try: case = int(fld[2:])
                except ValueError: continue
                hh_map[case] = dcms
            elif fld.startswith("QI"):
                try: case = int(fld.split()[1])
                except (ValueError, IndexError): continue
                sc_map[case] = dcms

    # Sort EP patients by BMI; pick 5 evenly spaced across BMI range
    valid_bmi = df_ep[df_ep["BMI"].notna()].sort_values("BMI")

    def pick_5(case_map):
        available = valid_bmi[valid_bmi["Case No"].isin(case_map.keys())]
        if available.empty:
            return []
        n = len(available)
        indices = [int(round(i)) for i in np.linspace(0, n-1, min(5, n))]
        return available.iloc[indices][["Case No", "BMI"]].values.tolist()

    hh_picks = pick_5(hh_map)
    sc_picks  = pick_5(sc_map)

    def load_pixel(dcm_path):
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(float)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        return arr

    n_cols = max(len(hh_picks), len(sc_picks))
    if n_cols == 0:
        print("  ⚠  No DICOM images found – skipping BMI gradient figure.")
        return

    # Define a subtle warm gradient colormap for BMI axis
    bmi_cmap = LinearSegmentedColormap.from_list(
        "bmi", ["#bfdbfe", "#1d4ed8", "#7f1d1d"], N=256)

    fig = plt.figure(figsize=(4 * n_cols + 1, 10))
    fig.patch.set_facecolor("#0f172a")
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.12)

    def fill_row(row_idx, picks, case_map, row_title):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer[row_idx], wspace=0.05)
        for col_i, (case, bmi) in enumerate(picks):
            ax = fig.add_subplot(inner[col_i])
            dcm_path = case_map[int(case)][len(case_map[int(case)])//2]
            try:
                img = load_pixel(dcm_path)
                bmi_norm = min((bmi - 16) / (50 - 16), 1.0)
                ax.imshow(img, cmap="gray", aspect="auto")
                for spine in ax.spines.values():
                    spine.set_edgecolor(bmi_cmap(bmi_norm))
                    spine.set_linewidth(4)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center",
                        color="white", fontsize=10)
                ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"BMI {bmi:.1f}", color="white", fontsize=FONT_BASE,
                          fontweight="bold", labelpad=6)
            if col_i == 0:
                ax.set_ylabel(row_title, color="white", fontsize=FONT_BASE+1,
                              fontweight="bold", rotation=90, labelpad=8)

    if hh_picks:
        fill_row(0, hh_picks, hh_map, "HH (Handheld)")
    if sc_picks:
        fill_row(1, sc_picks, sc_map,  "SC (Standard Care)")

    # Colorbar for BMI gradient
    sm = plt.cm.ScalarMappable(cmap=bmi_cmap, norm=plt.Normalize(vmin=16, vmax=50))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation="horizontal",
                        fraction=0.03, pad=0.06, shrink=0.6)
    cbar.set_label("BMI (kg/m²)", color="white", fontsize=FONT_BASE, labelpad=6)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white", fontsize=FONT_BASE-1)
    cbar.outline.set_edgecolor("white")

    fig.suptitle("POCUS Image Quality Across BMI Range — HH vs SC",
                 color="white", fontsize=FONT_BASE+5, fontweight="bold", y=1.01)
    savefig("fig10_bmi_gradient_images.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 11 – EP Cyst characterisation (HH) + categorical binary
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
#  MAIN
# ══════════════════════════════════════════════════════════════════════
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

    print(f"\n✓ All figures saved to: {FIG_DIR}\n")
