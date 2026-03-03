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

def find_paired_images():
    """Identifies folders with paired HH and SC images by matching suffixes."""
    pairs = []
    if not os.path.isdir(EP_IMG):
        return []
    
    for fld in os.listdir(EP_IMG):
        full = os.path.join(EP_IMG, fld)
        if not os.path.isdir(full): continue
        m = _re.search(r"\d+", fld)
        if not m: continue
        case_no = int(m.group())
        
        files = os.listdir(full)
        
        # 1. Map HH images by their suffix (text after "HH ")
        hh_map = {}
        for f in files:
            if f.upper().startswith("HH") and f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                suffix = _re.sub(r"^HH\s*", "", f, flags=_re.IGNORECASE)
                suffix = os.path.splitext(suffix)[0].strip().lower()
                hh_map[suffix] = os.path.join(full, f)
        
        # 2. Map SC images by their suffix (text after "SC ")
        sc_map = {}
        for f in files:
            if f.upper().startswith("SC") and f.lower().endswith('.dcm'):
                suffix = _re.sub(r"^SC\s*", "", f, flags=_re.IGNORECASE)
                suffix = os.path.splitext(suffix)[0].strip().lower()
                sc_map[suffix] = os.path.join(full, f)
        
        # 3. Pair them up where suffixes match
        for sfx, hh_path in hh_map.items():
            if sfx in sc_map:
                pairs.append({
                    "Case No": case_no,
                    "Suffix": sfx,
                    "hh_path": hh_path,
                    "sc_path": sc_map[sfx]
                })
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
    
    # Compute correlation matrix between corresponding features
    features = [c.replace("HH_", "") for c in df.columns if c.startswith("HH_")]
    corr_data = {}
    for feat in features:
        hh_col = f"HH_{feat}"
        sc_col = f"SC_{feat}"
        if hh_col in df.columns and sc_col in df.columns:
            # Filter non-constant signals
            if df[hh_col].nunique() > 1 and df[sc_col].nunique() > 1:
                sub = df[[hh_col, sc_col]].dropna()
                if len(sub) > 1:
                    r = sub.corr().iloc[0, 1]
                    # Map NaN result to 0 for plotting
                    corr_data[feat] = r if not np.isnan(r) else 0.0
    
    if not corr_data:
        print("\nNo varying features for correlation analysis.")
        return

    # 1. Bar plot of diagonal correlations (paired corresponding features)
    corr_series = pd.Series(corr_data).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=corr_series.values, y=corr_series.index, palette="mako")
    plt.axvline(0, color='black', lw=1)
    plt.title("Correlation: HH vs SC Corresponding Features (Whole Image)", fontweight="bold")
    plt.xlabel("Pearson r")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "radiomics_correlation_summary.png"), dpi=300)
    
    # 2. Full Cross-Correlation Heatmap (HH features vs SC features)
    # Extract just the feature columns for HH and SC
    hh_cols = [c for c in df.columns if c.startswith("HH_") and c.replace("HH_", "") in corr_series.index]
    sc_cols = [c for c in df.columns if c.startswith("SC_") and c.replace("SC_", "") in corr_series.index]
    
    # Compute the cross-correlation matrix (HH vs SC)
    cross_corr = pd.DataFrame(index=[c.replace("HH_", "") for c in hh_cols], 
                              columns=[c.replace("SC_", "") for c in sc_cols])
    
    for h_col in hh_cols:
        for s_col in sc_cols:
            r = df[[h_col, s_col]].dropna().corr().iloc[0, 1]
            cross_corr.loc[h_col.replace("HH_", ""), s_col.replace("SC_", "")] = r
            
    cross_corr = cross_corr.astype(float)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cross_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                cbar_kws={'label': 'Pearson r'})
    plt.title("Cross-Correlation Heatmap: HH Features vs SC Features", fontweight="bold")
    plt.xlabel("Standard Care (SC) Features")
    plt.ylabel("Handheld (HH) Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "radiomics_heatmap.png"), dpi=300)
    
    print("\nSummary of correlations:")
    print(corr_series.to_string())
    print(f"\n✓ Analysis complete. Heatmap saved to {os.path.join(FIG_DIR, 'radiomics_heatmap.png')}")

if __name__ == "__main__":
    main()
