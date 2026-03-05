import os, sys, glob, warnings, sysconfig
# Keep stdlib ssl accessible; the repo has a local ssl/ folder that would shadow it
_stdlib = sysconfig.get_paths()["stdlib"]
if _stdlib not in sys.path:
    sys.path.insert(0, _stdlib)
# Drop the script's own directory from sys.path to prevent local folder shadowing
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.realpath(p) != os.path.realpath(_script_dir)]

import re as _re
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from radiomics import firstorder
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(BASE_DIR), "Databse Spreadsheets")
EP_IMG    = os.path.join(os.path.dirname(BASE_DIR), "Early Pregnancy Images")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def load_image_as_sitk(path):
    """Loads an image (DICOM or PNG/JPG) and returns a SimpleITK image."""
    if path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(path).convert('L')
        arr = np.array(img).astype(float)
    else:
        ds  = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(float)
        
        # Handle different dimensions:
        # (H, W) -> 2D
        # (H, W, 3) -> RGB 2D
        # (F, H, W) -> Multi-frame Grayscale
        # (F, H, W, 3) -> Multi-frame RGB
        if arr.ndim == 3:
            # If 3rd dim is 3, it's likely RGB. If it's large, it's likely frames.
            # We want one grayscale channel of one frame.
            if arr.shape[-1] == 3:
                arr = arr[..., 0] # Take 1st component
            else:
                arr = arr[0, ...] # Take 1st frame
        elif arr.ndim == 4:
            # (Frames, H, W, 3)
            arr = arr[0, :, :, 0]
    
    # Robust normalization: clip top/bottom 1% to reduce UI/text interference
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p1, p99)
    # Simple normalization to 0-255 for radiomics
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255
    return sitk.GetImageFromArray(arr.astype(np.uint8))

def extract_first_order_features(sitk_img):
    """Extracts first-order features from the entire image area."""
    # Create a mask for the whole image
    mask_arr = np.ones(sitk_img.GetSize()[::-1], dtype=np.uint8)
    sitk_mask = sitk.GetImageFromArray(mask_arr)
    sitk_mask.CopyInformation(sitk_img)

    extractor = firstorder.RadiomicsFirstOrder(sitk_img, sitk_mask)
    # Enable all first-order features
    extractor.enableAllFeatures()
    return extractor.execute()

def fuzzy_normalize(s):
    """Normalize names for fuzzy matching."""
    s = s.lower()
    s = s.replace("sagital", "sagittal")
    s = s.replace("transverse", "trans")
    s = s.replace("and", "&")
    # remove all non-alphanumeric
    s = _re.sub(r'[^a-z0-9]', '', s)
    return s

def find_paired_images():
    """Identifies folders with paired HH and SC images by matching suffixes with robust fuzzy logic."""
    pairs = []
    unpaired = []
    if not os.path.isdir(EP_IMG):
        return []
    
    for fld in os.listdir(EP_IMG):
        full = os.path.join(EP_IMG, fld)
        if not os.path.isdir(full): continue
        m = _re.search(r"\d+", fld)
        if not m: continue
        case_no = int(m.group())
        
        files = os.listdir(full)
        
        # 1. Map HH images by their fuzzy suffix
        hh_map = {}
        for f in files:
            if f.upper().startswith("HH") and f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                suffix_raw = _re.sub(r"^HH\s*", "", f, flags=_re.IGNORECASE)
                suffix_raw = os.path.splitext(suffix_raw)[0].strip()
                fuzzy_sfx = fuzzy_normalize(suffix_raw)
                # Store (path, original_name, base_name_no_index)
                base_name = _re.sub(r"\d+$", "", fuzzy_sfx)
                hh_map[fuzzy_sfx] = (os.path.join(full, f), suffix_raw, base_name)
        
        # 2. Map SC images by their fuzzy suffix
        sc_map = {}
        for f in files:
            if f.upper().startswith("SC") and f.lower().endswith('.dcm'):
                suffix_raw = _re.sub(r"^SC\s*", "", f, flags=_re.IGNORECASE)
                suffix_raw = os.path.splitext(suffix_raw)[0].strip()
                fuzzy_sfx = fuzzy_normalize(suffix_raw)
                base_name = _re.sub(r"\d+$", "", fuzzy_sfx)
                sc_map[fuzzy_sfx] = (os.path.join(full, f), suffix_raw, base_name)
        
        # 3. Pair them up
        matched_hh = set()
        matched_sc = set()
        
        # Exact fuzzy match first
        for sfx_h, (hh_path, hh_orig, base_h) in hh_map.items():
            if sfx_h in sc_map:
                sc_path, sc_orig, base_s = sc_map[sfx_h]
                pairs.append({"Case No": case_no, "Suffix": hh_orig, "hh_path": hh_path, "sc_path": sc_path})
                matched_hh.add(sfx_h); matched_sc.add(sfx_h)
        
        # Fuzzy match where one side lacks an index (e.g. 'Embryo' <-> 'Embryo 1')
        for sfx_h, (hh_path, hh_orig, base_h) in hh_map.items():
            if sfx_h in matched_hh: continue
            for sfx_s, (sc_path, sc_orig, base_s) in sc_map.items():
                if sfx_s in matched_sc: continue
                if base_h == base_s:
                    # One is a base of the other or both share base and one is indexed
                    pairs.append({"Case No": case_no, "Suffix": hh_orig, "hh_path": hh_path, "sc_path": sc_path})
                    matched_hh.add(sfx_h); matched_sc.add(sfx_s)
                    break
                    
        # Log unpaired for debugging
        for sfx_h, (hh_path, hh_orig, base_h) in hh_map.items():
            if sfx_h not in matched_hh:
                unpaired.append(f"Case {case_no}: HH {hh_orig} (unpaired)")
        for sfx_s, (sc_path, sc_orig, base_s) in sc_map.items():
            if sfx_s not in matched_sc:
                unpaired.append(f"Case {case_no}: SC {sc_orig} (unpaired)")
                
    if unpaired:
        print(f"\nReport: {len(unpaired)} files remained unpaired. (Top 10 below)")
        for u in unpaired[:10]: print(f"  {u}")
        
    return pairs

def main():
    print("POCUS Radiomics Analysis (Whole Image, Paired HH vs SC)")
    print("-------------------------------------------------------")
    
    pairs = find_paired_images()
    print(f"Found {len(pairs)} folders with paired images.")
    
    results = []
    for pair in pairs:
        case = pair["Case No"]
        print(f"Processing Case {case}...")
        try:
            print(f"  Loading HH: {os.path.basename(pair['hh_path'])}")
            sitk_hh = load_image_as_sitk(pair["hh_path"])
            print(f"  Loading SC: {os.path.basename(pair['sc_path'])}")
            sitk_sc = load_image_as_sitk(pair["sc_path"])
            
            print(f"  Extracting HH features...")
            feat_hh = extract_first_order_features(sitk_hh)
            print(f"  Extracting SC features...")
            feat_sc = extract_first_order_features(sitk_sc)
            
            row = {"Case No": case, "Suffix": pair["Suffix"]}
            for k, v in feat_hh.items():
                # cast to float scalar to avoid numpy array issues in pandas
                row[f"HH_{k}"] = float(v)
            for k, v in feat_sc.items():
                row[f"SC_{k}"] = float(v)
            results.append(row)
        except Exception as e:
            print(f"\nError processing Case {case}: {e}")
            continue
            
    if not results:
        print("\nNo results to analyze.")
        return

    df = pd.DataFrame(results)
    
    # Clean column names (remove 'original_firstorder_')
    rename_cols = {}
    for col in df.columns:
        if "original_firstorder_" in col:
            new_col = col.replace("original_firstorder_", "")
            rename_cols[col] = new_col
    df = df.rename(columns=rename_cols)
    
    # Save raw results
    csv_path = os.path.join(BASE_DIR, "radiomics_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")
    
    # --- ANALYSIS PART ────────────────────────────────────────────────────────
    
    def compute_correlations(target_df, label):
        cols = [c.replace("HH_", "") for c in target_df.columns if c.startswith("HH_")]
        corr_dict = {}
        for feat in cols:
            h = f"HH_{feat}"; s = f"SC_{feat}"
            if h in target_df.columns and s in target_df.columns:
                if target_df[h].nunique() > 1 and target_df[s].nunique() > 1:
                    sub = target_df[[h, s]].dropna()
                    if len(sub) > 1:
                        r = sub.corr().iloc[0, 1]
                        corr_dict[feat] = r if not np.isnan(r) else 0.0
        return pd.Series(corr_dict).sort_values(ascending=False)

    print("\nRunning correlations...")
    img_corr = compute_correlations(df, "Image-Level")
    
    # Patient-Level Averaging
    feat_cols = [c for c in df.columns if c.startswith("HH_") or c.startswith("SC_")]
    patient_df = df.groupby("Case No")[feat_cols].mean().reset_index()
    pat_corr = compute_correlations(patient_df, "Patient-Level")
    
    print(f"Image-Level N: {len(df)}")
    print(f"Patient-Level N: {len(patient_df)}")

    # 1. Visual Comparison: Bar Plot
    # Combine results for comparison
    comp_df = pd.DataFrame({
        "Image-Level (n=37)": img_corr,
        "Patient-Level (n=14)": pat_corr
    }).sort_values("Patient-Level (n=14)", ascending=False)

    plt.figure(figsize=(12, 10))
    comp_df.plot(kind="barh", figsize=(12, 10), color=["#2ecc71", "#3498db"], alpha=0.8)
    plt.axvline(0, color='black', lw=1)
    plt.title("Radiomics Correlation: HH vs SC (Image-Level vs Patient-Level Average)", fontweight="bold")
    plt.xlabel("Pearson r")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "radiomics_correlation_comparison.png"), dpi=300)
    
    # 2. Patient-Level Heatmap — build a clean float64 matrix for seaborn
    hh_f_cols = [f"HH_{f}" for f in pat_corr.index]
    sc_f_cols  = [f"SC_{f}"  for f in pat_corr.index]
    feat_names = list(pat_corr.index)
    n = len(feat_names)
    
    # Compute pairwise correlations into a fresh numpy matrix
    mat = np.full((n, n), np.nan)
    for i, hf in enumerate(hh_f_cols):
        for j, sf in enumerate(sc_f_cols):
            sub = patient_df[[hf, sf]].dropna()
            if len(sub) > 1:
                mat[i, j] = float(sub.corr().iloc[0, 1])
    
    # Build clean DataFrame — use integer positional index first, then rename
    cross_corr_p = pd.DataFrame(mat, dtype=float)
    cross_corr_p.index   = feat_names
    cross_corr_p.columns = feat_names
    
    # Build annotation labels array (same shape as mat)
    annot_labels = np.array([[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in mat])
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cross_corr_p,              # data for colour mapping
        annot=annot_labels,        # explicit string annotations –– bypasses seaborn dtype bug
        fmt="",                    # must be empty string when annot is strings
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        annot_kws={"size": 9},
        cbar_kws={'label': 'Pearson r'},
        ax=ax
    )
    ax.set_title("Patient-Level Radiomics Heatmap\n(HH vs SC Averaged Features per Case)",
                 fontweight="bold", pad=20)
    ax.set_xlabel("Standard Care (SC) Features", labelpad=10)
    ax.set_ylabel("Handheld (HH) Features", labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "radiomics_heatmap_patient_level.png"), dpi=300)
    plt.close()
    
    print("\nSummary of Patient-Level Correlations (Diagonal):")
    print(pat_corr.to_string())
    print("\n✓ Analysis complete.")

if __name__ == "__main__":
    main()
