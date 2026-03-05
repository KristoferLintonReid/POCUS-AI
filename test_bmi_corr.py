import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# Load data
df_ep = pd.read_excel('/Users/kl2418/Downloads/POCUS G/Databse Spreadsheets/Early Pregnancy Data.xlsx')
df_gy = pd.read_excel('/Users/kl2418/Downloads/POCUS G/Databse Spreadsheets/POCUS Gynae Data .xlsx')

print("--- EP BMI Correlations ---")
ep_pairs = [
    ("MSD",            "HH MSD",       "MSD"),
    ("R Ovary D1",     "HH R1",        "R1"),
    ("L Ovary D1",     "HH L1",        "L1"),
]

if "BMI" in df_ep.columns:
    df_ep["BMI"] = pd.to_numeric(df_ep["BMI"], errors="coerce")
    for label, hh, sc in ep_pairs:
        df_ep[hh] = pd.to_numeric(df_ep[hh], errors="coerce")
        df_ep[sc] = pd.to_numeric(df_ep[sc], errors="coerce")
        
        paired = df_ep[["BMI", hh, sc]].dropna()
        if len(paired) > 2:
            diff = abs(paired[hh] - paired[sc])
            r, p = pearsonr(paired["BMI"], diff)
            print(f"{label}: r={r:.3f}, p={p:.3f}, n={len(paired)}")

print("\n--- GYNAE BMI Correlations ---")
gy_pairs = [
    ("Endometrial Thickness", "HANDHELDEndometrialthickness", "STANDARDCAREEndometrialthickness"),
    ("Uterus Length", "HANDHELDUterinedimensionsLONGITUDINAL", "STANDARDCAREUterinedimensionsLONGITUDINAL"),
]

bmi_col = 'BMI' if 'BMI' in df_gy.columns else ('BMI (SC)' if 'BMI (SC)' in df_gy.columns else None)
if bmi_col:
    df_gy[bmi_col] = pd.to_numeric(df_gy[bmi_col], errors="coerce")
    for label, hh, sc in gy_pairs:
        df_gy[hh] = pd.to_numeric(df_gy[hh], errors="coerce")
        df_gy[sc] = pd.to_numeric(df_gy[sc], errors="coerce")
        
        paired = df_gy[[bmi_col, hh, sc]].dropna()
        if len(paired) > 2:
            diff = abs(paired[hh] - paired[sc])
            r, p = pearsonr(paired[bmi_col], diff)
            print(f"{label}: r={r:.3f}, p={p:.3f}, n={len(paired)}")
