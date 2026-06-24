"""
Statistical analysis utilities.
Correlation tests, Bonferroni correction, ANCOVA.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def bonferroni_threshold(n_tests, alpha=0.05):
    """Return Bonferroni-corrected significance threshold."""
    return alpha / n_tests

def pearson_r2(series_a, series_b):
    """Return (r, r_squared, p_value) for two series, dropping NaNs."""
    valid = pd.concat([series_a, series_b], axis=1).dropna()
    if len(valid) < 5:
        return np.nan, np.nan, np.nan
    r, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
    return r, r**2, p

def eta_squared(df, dim_col, group_col):
    """Return (eta_squared, p_value) for one-way ANOVA of dim_col by group_col."""
    groups     = [df[df[group_col] == g][dim_col].dropna().values
                  for g in df[group_col].unique()]
    _, p       = stats.f_oneway(*groups)
    grand_mean = df[dim_col].mean()
    ss_total   = ((df[dim_col] - grand_mean)**2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    eta2       = ss_between / ss_total if ss_total > 0 else 0
    return eta2, p

def run_ancova(df, dim_col, categorical_col, continuous_col, alpha=0.05):
    """
    Fit: dim ~ C(categorical) + continuous
    Returns dict with partial eta2 and significance for each effect.
    """
    data  = df[[dim_col, categorical_col, continuous_col]].dropna()
    model = ols(f"{dim_col} ~ C({categorical_col}) + {continuous_col}", data=data).fit()
    tbl   = sm.stats.anova_lm(model, typ=2)

    ss_cat  = tbl.loc[f"C({categorical_col})", "sum_sq"]
    ss_cont = tbl.loc[continuous_col,          "sum_sq"]
    ss_res  = tbl.loc["Residual",              "sum_sq"]
    p_cat   = tbl.loc[f"C({categorical_col})", "PR(>F)"]
    p_cont  = tbl.loc[continuous_col,          "PR(>F)"]

    return {
        "peta2_categorical":  ss_cat  / (ss_cat  + ss_res),
        "peta2_continuous":   ss_cont / (ss_cont + ss_res),
        "p_categorical":      p_cat,
        "p_continuous":       p_cont,
        "sig_categorical":    p_cat  < alpha,
        "sig_continuous":     p_cont < alpha,
    }
