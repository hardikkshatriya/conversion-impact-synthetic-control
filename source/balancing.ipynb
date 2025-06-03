import numpy as np

def calculate_smd(X, treatment, weights=None):
    """
    Calculates Standardized Mean Difference (SMD) for covariate balance checking.
    If weights are provided, computes weighted means and variances.
    """
    smd_results = {}
    for col in X.columns:
        treated = X[treatment == 1][col]
        control = X[treatment == 0][col]
        if weights is not None:
            wt_treated = weights[treatment == 1]
            wt_control = weights[treatment == 0]
            mean_t = np.average(treated, weights=wt_treated)
            mean_c = np.average(control, weights=wt_control)
            var_t = np.average((treated - mean_t) ** 2, weights=wt_treated)
            var_c = np.average((control - mean_c) ** 2, weights=wt_control)
        else:
            mean_t = treated.mean()
            mean_c = control.mean()
            var_t = treated.var()
            var_c = control.var()
        smd = (mean_t - mean_c) / np.sqrt((var_t + var_c) / 2)
        smd_results[col] = smd
    return smd_results
