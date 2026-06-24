"""
Data loading and splitting utilities.
Import these in your notebooks instead of repeating the same code.
"""

import pandas as pd
import numpy as np
import pickle

PATIENT_COL  = "PATNO"
LABEL_COL    = "label"
LATENT_COLS  = [f"latent_{i}" for i in range(256)]
SBR_COLS     = ["DATSCAN_CAUDATE_R", "DATSCAN_CAUDATE_L",
                "DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L",
                "DATSCAN_PUTAMEN_R_ANT", "DATSCAN_PUTAMEN_L_ANT"]

def patient_stratified_split(df, train_ratio=0.8, label_col=LABEL_COL,
                             patient_col=PATIENT_COL, random_state=42):
#  Split df into train and val ensuring all visits of one patientstay in the same split.
#  Stratified by diagnosis label. 
# Returns (df_train, df_val). 

    np.random.seed(random_state)
    patients = df.drop_duplicates(patient_col)[[patient_col, label_col]]
    train_patients = set()
    for label_val in patients[label_col].unique():
        group = patients[patients[label_col] == label_val][patient_col].tolist()
        np.random.shuffle(group)
        n_train = int(train_ratio * len(group))
        train_patients.update(group[:n_train])
    df_train = df[df[patient_col].isin(train_patients)].copy()
    df_val   = df[~df[patient_col].isin(train_patients)].copy()
    overlap  = set(df_train[patient_col]) & set(df_val[patient_col])
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} patients in both splits"
    return df_train, df_val

def load_fitted_objects(scaler_path, pca_path):
    # Load pre-fitted scaler and PCA objects saved during training.
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    with open(pca_path,   "rb") as f: pca    = pickle.load(f)
    return scaler, pca

def get_active_dims(df, latent_cols=LATENT_COLS, variance_threshold=0.1):
    # Return list of latent dims with std above threshold (not collapsed).
    return [c for c in latent_cols if df[c].std() > variance_threshold]
