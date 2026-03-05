import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np

# Load data
df_ep = pd.read_excel('/Users/kl2418/Downloads/POCUS G/Databse Spreadsheets/Early Pregnancy Data.xlsx')
df_gyn = pd.read_excel('/Users/kl2418/Downloads/POCUS G/Databse Spreadsheets/POCUS Gynae Data .xlsx')

# Clean columns
def clean_cols(df):
    df.columns = df.columns.astype(str).str.strip()
    return df

df_ep = clean_cols(df_ep)
df_gyn = clean_cols(df_gyn)

# EP
ep_metrics = [
    ('CRL (mm) (HH)', 'CRL (mm) (SC)'),
    ('Mean GS (mm) (HH)', 'Mean GS (mm) (SC)'),
    ('Endometrial Thickness (mm) (HH)', 'Endometrial Thickness (mm) (SC)')
]
print("--- EARLY PREGNANCY STATS ---")
if 'BMI (SC)' in df_ep.columns:
    df_ep['BMI (SC)'] = pd.to_numeric(df_ep['BMI (SC)'], errors='coerce')
    for hh, sc in ep_metrics:
        if hh in df_ep.columns and sc in df_ep.columns:
            df_ep[hh] = pd.to_numeric(df_ep[hh], errors='coerce')
            df_ep[sc] = pd.to_numeric(df_ep[sc], errors='coerce')
            df_valid = df_ep.dropna(subset=['BMI (SC)', hh, sc])
            if len(df_valid) > 2:
                diff = abs(df_valid[hh] - df_valid[sc])
                corr, p = pearsonr(df_valid['BMI (SC)'], diff)
                print(f"EP {hh.replace(' (mm) (HH)', '')}: r={corr:.3f}, p={p:.3f}, n={len(df_valid)}")

# Gynae
gyn_metrics = [
    ('Endometrial Thickness (HH)', 'Endometrial Thickness (SC)'),
    ('Uterus LONG (HH)', 'Uterus LONG (SC)'),
    ('Right Ovary Vol (HH)', 'Right Ovary Vol (SC)')
]
print("\n--- GYNAECOLOGY STATS ---")
bmi_col = 'BMI' if 'BMI' in df_gyn.columns else 'BMI (SC)'
if bmi_col in df_gyn.columns:
    df_gyn[bmi_col] = pd.to_numeric(df_gyn[bmi_col], errors='coerce')
    for hh, sc in gyn_metrics:
        if hh in df_gyn.columns and sc in df_gyn.columns:
            df_gyn[hh] = pd.to_numeric(df_gyn[hh], errors='coerce')
            df_gyn[sc] = pd.to_numeric(df_gyn[sc], errors='coerce')
            df_valid = df_gyn.dropna(subset=[bmi_col, hh, sc])
            if len(df_valid) > 2:
                diff = abs(df_valid[hh] - df_valid[sc])
                corr, p = pearsonr(df_valid[bmi_col], diff)
                print(f"Gyn {hh.replace(' (HH)', '')}: r={corr:.3f}, p={p:.3f}, n={len(df_valid)}")
