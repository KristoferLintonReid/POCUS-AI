"""
POCUS-AI: SC vs HH (Handheld POCUS) Statistical Analysis
==========================================================
Compares Standard Care (SC) ultrasound vs Handheld (HH) POCUS across:
  - Early Pregnancy dataset
  - Gynaecology dataset

HH = Handheld POCUS device (folders prefixed PG*)
SC = Standard Care ultrasound (folders prefixed QI*)

Usage:
    python stats_analysis.py

Outputs:
    - Console summary statistics
    - Plots saved to ./stats_output/
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  PATHS  (edit these if you move the data)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Databse Spreadsheets")
EP_IMG_DIR = os.path.join(os.path.dirname(BASE_DIR), "Early Pregnancy Images")
GY_IMG_DIR = os.path.join(os.path.dirname(BASE_DIR), "Gyanecology Images")
EP_SHEET = os.path.join(DATA_DIR, "Early Pregnancy Data.xlsx")
GY_SHEET = os.path.join(DATA_DIR, "POCUS Gynae Data .xlsx")
OUT_DIR = os.path.join(BASE_DIR, "stats_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
#  STYLE
# ──────────────────────────────────────────────
HH_COLOR = "#4C72B0"   # blue for HH
SC_COLOR  = "#DD8452"  # orange for SC
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def count_dcm(folder: str) -> int:
    """Return number of DICOM files recursively under folder."""
    if not os.path.isdir(folder):
        return 0
    return len(glob.glob(os.path.join(folder, "**", "*.dcm"), recursive=True))


def describe(series: pd.Series, label: str) -> dict:
    s = series.dropna()
    return {
        "label":  label,
        "n":      len(s),
        "mean":   s.mean(),
        "median": s.median(),
        "std":    s.std(),
        "min":    s.min(),
        "max":    s.max(),
        "q25":    s.quantile(0.25),
        "q75":    s.quantile(0.75),
    }


def mannwhitney(a: pd.Series, b: pd.Series) -> tuple:
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return u, p


def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def print_desc(d: dict):
    print(f"  {d['label']:30s} n={d['n']:4d}  "
          f"mean={d['mean']:.2f}  median={d['median']:.2f}  "
          f"std={d['std']:.2f}  [{d['min']:.1f}–{d['max']:.1f}]")


def save_fig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  → saved {path}")


# ══════════════════════════════════════════════
#  IMAGE COUNT ANALYSIS
# ══════════════════════════════════════════════

def image_count_analysis():
    print_section("IMAGE COUNT ANALYSIS  (DICOM files per patient folder)")

    # Early Pregnancy Images
    ep_records = []
    if os.path.isdir(EP_IMG_DIR):
        for folder in sorted(os.listdir(EP_IMG_DIR)):
            full = os.path.join(EP_IMG_DIR, folder)
            if not os.path.isdir(full):
                continue
            n = count_dcm(full)
            # HH = Handheld → folders starting with PG
            # SC = Standard Care → folders starting with QI
            if folder.startswith("PG"):
                scan_type = "HH"
            elif folder.startswith("QI"):
                scan_type = "SC"
            else:
                scan_type = "Unknown"
            ep_records.append({"patient": folder, "scan_type": scan_type, "n_images": n})
    ep_df = pd.DataFrame(ep_records)

    # Gynaecology Images
    gy_records = []
    if os.path.isdir(GY_IMG_DIR):
        for folder in sorted(os.listdir(GY_IMG_DIR)):
            full = os.path.join(GY_IMG_DIR, folder)
            if not os.path.isdir(full):
                continue
            n = count_dcm(full)
            gy_records.append({"patient": folder, "scan_type": "SC", "n_images": n})
    gy_df = pd.DataFrame(gy_records)

    # ── Summary: Early Pregnancy ──
    print("\n── Early Pregnancy ──")
    for stype in ["HH", "SC"]:
        sub = ep_df[ep_df["scan_type"] == stype]["n_images"]
        d = describe(sub, f"Early Preg {stype}")
        print_desc(d)

    if ep_df[ep_df["scan_type"] == "SC"].shape[0] > 0:
        u, p = mannwhitney(
            ep_df[ep_df["scan_type"] == "HH"]["n_images"],
            ep_df[ep_df["scan_type"] == "SC"]["n_images"]
        )
        print(f"  Mann-Whitney U={u:.1f}, p={p:.4f}")

    print(f"\n  Total EP images: {ep_df['n_images'].sum()}")
    print(f"  Patients (HH):  {(ep_df['scan_type']=='HH').sum()}")
    print(f"  Patients (SC):  {(ep_df['scan_type']=='SC').sum()}")

    # ── Gynaecology ──
    print("\n── Gynaecology ──")
    if not gy_df.empty:
        d = describe(gy_df["n_images"], "Gynae SC")
        print_desc(d)
        print(f"  Total images: {gy_df['n_images'].sum()}")
    else:
        print("  (no image folders found)")

    # ── Plot: EP image counts ──
    hh_imgs = ep_df[ep_df["scan_type"] == "HH"]["n_images"]
    sc_imgs = ep_df[ep_df["scan_type"] == "SC"]["n_images"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Early Pregnancy: Images per Patient Folder", fontsize=14, fontweight="bold")

    # Histogram
    ax = axes[0]
    all_vals = pd.concat([hh_imgs, sc_imgs])
    bins = range(0, int(all_vals.max()) + 3, 2)
    ax.hist(hh_imgs, bins=bins, alpha=0.7, color=HH_COLOR, label="HH (Handheld POCUS)", edgecolor="white")
    ax.hist(sc_imgs,  bins=bins, alpha=0.7, color=SC_COLOR,  label="SC (Standard Care)",  edgecolor="white")
    ax.set_xlabel("Images per Patient")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Image Counts")
    ax.legend()

    # Box plot
    ax = axes[1]
    data_to_plot = [hh_imgs.values, sc_imgs.values]
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(HH_COLOR)
    if len(bp["boxes"]) > 1:
        bp["boxes"][1].set_facecolor(SC_COLOR)
    ax.set_xticklabels(["HH (Handheld)", "SC (Standard Care)"])
    ax.set_ylabel("Images per Patient")
    ax.set_title("Boxplot: HH vs SC Image Counts")
    plt.tight_layout()
    save_fig("ep_image_counts_hh_vs_sc.png")

    return ep_df, gy_df


# ══════════════════════════════════════════════
#  EARLY PREGNANCY CLINICAL DATA
# ══════════════════════════════════════════════

def early_pregnancy_stats():
    print_section("EARLY PREGNANCY — CLINICAL DATA (HH vs SC)")

    df = pd.read_excel(EP_SHEET)
    total = len(df)
    print(f"\n  Total patients: {total}")
    print(f"  Total columns:  {len(df.columns)}")

    # Demographics
    print("\n── Demographics ──")
    for col in ["Age", "BMI", "Gestational Weeks", "Gestation Days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            d = describe(df[col], col)
            print_desc(d)

    # Categorical: Indication for Scan
    if "Indication for Scan" in df.columns:
        print("\n  Indication for Scan (value counts):")
        vc = df["Indication for Scan"].value_counts(dropna=False)
        for val, cnt in vc.items():
            print(f"    {str(val):40s} {cnt:4d}  ({100*cnt/total:.1f}%)")

    # HH vs SC: structures visualised
    hh_cols = [c for c in df.columns if str(c).startswith("HH")]
    sc_cols  = [c for c in df.columns if not str(c).startswith("HH")
                and c not in ("Case No","Age","BMI","Indication for Scan",
                              "Gestational Weeks","Gestation Days","IVF",
                              "Progesterone use","Standard care User Level",
                              "Pathology ","Factors","Site")]

    # Binary visualisation completeness: how often did each modality record data
    hh_complete = df[hh_cols].notna().sum(axis=1)
    sc_complete = df[sc_cols].notna().sum(axis=1)

    print("\n── Completed fields per patient ──")
    print_desc(describe(hh_complete, "HH fields filled"))
    print_desc(describe(sc_complete, "SC fields filled"))

    u, p = mannwhitney(hh_complete, sc_complete)
    print(f"  Mann-Whitney U={u:.1f}, p={p:.4f}")

    # Diagnosis agreement
    hh_diag_col = "HH Diagnosis"
    sc_diag_col  = "Diagnosis"
    if hh_diag_col in df.columns and sc_diag_col in df.columns:
        paired = df[[hh_diag_col, sc_diag_col]].dropna()
        if len(paired) > 0:
            agree = (paired[hh_diag_col] == paired[sc_diag_col]).sum()
            print(f"\n  Diagnosis agreement (HH vs SC): {agree}/{len(paired)} = {100*agree/len(paired):.1f}%")

    # ── Plot: demographics ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Early Pregnancy: Patient Demographics", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, ["Age", "BMI", "Gestational Weeks"]):
        if col in df.columns:
            vals = df[col].dropna()
            ax.hist(vals, bins=15, color="#5A9E6F", edgecolor="white", alpha=0.85)
            ax.axvline(vals.mean(), color="crimson", lw=1.5, linestyle="--", label=f"mean={vals.mean():.1f}")
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig("ep_demographics.png")

    # ── Plot: HH vs SC field completeness ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([hh_complete.dropna().values, sc_complete.dropna().values],
               patch_artist=True,
               medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(ax.patches, [HH_COLOR, SC_COLOR]):
        patch.set_facecolor(color)
    ax.set_xticklabels(["HH (Handheld POCUS)", "SC (Standard Care)"])
    ax.set_ylabel("Number of Fields Completed per Patient")
    ax.set_title("Early Pregnancy: Data Completeness — HH vs SC")
    save_fig("ep_data_completeness.png")

    return df


# ══════════════════════════════════════════════
#  GYNAECOLOGY CLINICAL DATA
# ══════════════════════════════════════════════

def gynae_stats():
    print_section("GYNAECOLOGY — CLINICAL DATA (HANDHELD vs STANDARD CARE)")

    df = pd.read_excel(GY_SHEET)
    total = len(df)
    print(f"\n  Total patients: {total}")
    print(f"  Total columns:  {len(df.columns)}")

    # Demographics
    print("\n── Demographics ──")
    for col in ["Age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            d = describe(df[col], col)
            print_desc(d)

    if "Menopausalstatus" in df.columns:
        print("\n  Menopausal Status (1=pre, 2=post):")
        vc = df["Menopausalstatus"].value_counts(dropna=False)
        for val, cnt in vc.items():
            print(f"    {str(val):20s} {cnt:4d}  ({100*cnt/total:.1f}%)")

    if "Indicationforscan" in df.columns:
        print("\n  Indication for Scan:")
        vc = df["Indicationforscan"].value_counts(dropna=False)
        for val, cnt in vc.items():
            print(f"    {str(val):40s} {cnt:4d}  ({100*cnt/total:.1f}%)")

    # Handheld vs Standard Care: numeric measurements
    hh_measurements = {
        "Endometrial Thickness": ("HANDHELDEndometrialthickness", "STANDARDCAREEndometrialthickness"),
        "Left Ovary Vol":        ("HANDHELDLOVol",                "STANDARDCARELOVol"),
        "Right Ovary Vol":       ("HANDHELDROVol",                "StandardcareROVol"),
        "Uterus LONG":           ("HANDHELDUterinedimensionsLONGITUDINAL", "STANDARDCAREUterinedimensionsLONGITUDINAL"),
        "Uterus AP":             ("HANDHELDUterinedimensionsAP",  "STANDARDCAREUterinedimensionsAP"),
        "Uterus Transverse":     ("HANDHELDUterinedimensionsTransverse", "STANDARDCAREUterinedimensionsTransverse"),
        "Max Cyst Diameter":     ("HANDHELDMaxCystDiameter",      "STANDcystmaxdiameter"),
    }

    print("\n── Measurement Comparison (HH vs SC) ──")
    print(f"  {'Measurement':28s} {'HH mean':>10} {'SC mean':>10} {'p-value':>10}")
    print(f"  {'-'*62}")

    results = []
    for label, (hh_col, sc_col) in hh_measurements.items():
        hh_vals = df[hh_col].dropna() if hh_col in df.columns else pd.Series(dtype=float)
        sc_vals = df[sc_col].dropna()  if sc_col  in df.columns else pd.Series(dtype=float)
        u, p = mannwhitney(hh_vals, sc_vals)
        hh_mean = hh_vals.mean() if len(hh_vals) > 0 else np.nan
        sc_mean = sc_vals.mean() if len(sc_vals) > 0 else np.nan
        sig = "***" if (p < 0.001) else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")) if not np.isnan(p) else "—"
        print(f"  {label:28s} {hh_mean:>10.2f} {sc_mean:>10.2f} {str(round(p,4)) if not np.isnan(p) else 'n/a':>10}  {sig}")
        results.append({"measurement": label, "hh_mean": hh_mean, "sc_mean": sc_mean, "p_value": p})

    # Data completeness
    hh_cols = [c for c in df.columns if c.startswith("HANDHELD")]
    sc_cols  = [c for c in df.columns if c.startswith("STANDARDCARE")]
    hh_complete = df[hh_cols].notna().sum(axis=1)
    sc_complete  = df[sc_cols].notna().sum(axis=1)

    print("\n── Data Completeness ──")
    print_desc(describe(hh_complete, "HH fields filled"))
    print_desc(describe(sc_complete, "SC fields filled"))
    u, p = mannwhitney(hh_complete, sc_complete)
    print(f"  Mann-Whitney U={u:.1f}, p={p:.4f}")

    # Agreement on suggested management
    hh_mgmt = "HANDHELDSuggestedmanagement"
    sc_mgmt  = "STANDARDCARESuggestedmanagement"
    if hh_mgmt in df.columns and sc_mgmt in df.columns:
        paired = df[[hh_mgmt, sc_mgmt]].dropna()
        if len(paired) > 0:
            agree = (paired[hh_mgmt] == paired[sc_mgmt]).sum()
            print(f"\n  Management agreement (HH vs SC): {agree}/{len(paired)} = {100*agree/len(paired):.1f}%")

    consist_col = "Wasthesuggestedmanagementconsistentbetweengroups"
    if consist_col in df.columns:
        vc = df[consist_col].value_counts(dropna=False)
        print(f"\n  Was management consistent between groups?")
        for val, cnt in vc.items():
            print(f"    {str(val):20s} {cnt:4d}  ({100*cnt/total:.1f}%)")

    # ── Plots ──
    # Measurement comparison bar chart
    res_df = pd.DataFrame(results).dropna(subset=["hh_mean", "sc_mean"])
    if not res_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(res_df))
        w = 0.35
        ax.bar(x - w/2, res_df["hh_mean"], w, color=HH_COLOR, label="HH (Handheld)", alpha=0.85)
        ax.bar(x + w/2, res_df["sc_mean"],  w, color=SC_COLOR,  label="SC (Standard Care)", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(res_df["measurement"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean Value")
        ax.set_title("Gynaecology: Clinical Measurements — HH vs SC")
        ax.legend()
        plt.tight_layout()
        save_fig("gynae_measurements_hh_vs_sc.png")

    # Completeness boxplot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([hh_complete.values, sc_complete.values],
               patch_artist=True,
               medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(ax.patches, [HH_COLOR, SC_COLOR]):
        patch.set_facecolor(color)
    ax.set_xticklabels(["HH (Handheld POCUS)", "SC (Standard Care)"])
    ax.set_ylabel("Fields Completed per Patient")
    ax.set_title("Gynaecology: Data Completeness — HH vs SC")
    save_fig("gynae_data_completeness.png")

    # Age distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    if "Age" in df.columns:
        ax.hist(df["Age"].dropna(), bins=15, color="#9B59B6", edgecolor="white", alpha=0.85)
        ax.axvline(df["Age"].mean(), color="crimson", lw=1.5, linestyle="--",
                   label=f"mean={df['Age'].mean():.1f}")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Count")
        ax.set_title("Gynaecology: Age Distribution")
        ax.legend()
    save_fig("gynae_age_distribution.png")

    return df


# ══════════════════════════════════════════════
#  SUMMARY DASHBOARD
# ══════════════════════════════════════════════

def summary_dashboard(ep_img_df, gy_img_df, ep_df, gy_df):
    print_section("OVERALL SUMMARY DASHBOARD")

    hh_ep = ep_img_df[ep_img_df["scan_type"] == "HH"]
    sc_ep  = ep_img_df[ep_img_df["scan_type"] == "SC"]

    summary = {
        "Dataset": ["Early Pregnancy", "Early Pregnancy", "Gynaecology"],
        "Scan Type": ["HH (Handheld)", "SC (Standard Care)", "SC (Standard Care)"],
        "Patients (folders)": [len(hh_ep), len(sc_ep), len(gy_img_df)],
        "Total DICOM images": [hh_ep["n_images"].sum(), sc_ep["n_images"].sum(), gy_img_df["n_images"].sum() if not gy_img_df.empty else 0],
        "Mean images/patient": [hh_ep["n_images"].mean().round(1), sc_ep["n_images"].mean().round(1),
                                  gy_img_df["n_images"].mean().round(1) if not gy_img_df.empty else "n/a"],
    }
    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))

    # Save summary CSV
    csv_path = os.path.join(OUT_DIR, "summary_stats.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  → Summary saved to {csv_path}")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("POCUS-AI: HH vs SC Statistical Analysis")
    print(f"Output directory: {OUT_DIR}\n")

    ep_img_df, gy_img_df = image_count_analysis()
    ep_df = early_pregnancy_stats()
    gy_df  = gynae_stats()
    summary_dashboard(ep_img_df, gy_img_df, ep_df, gy_df)

    print(f"\n{'═'*60}")
    print("  Done! All plots saved to:", OUT_DIR)
    print(f"{'═'*60}\n")
