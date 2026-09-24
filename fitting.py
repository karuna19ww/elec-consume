# Copyright (c) 2026 Anurak
"""Curve fitting models for monthly consumption (t = 1, 2, 3, ... months)."""
import numpy as np
from scipy.optimize import curve_fit

MODELS = ["Linear", "Parabola (2nd degree)", "Exponential"]


def fit_model(name, t, y):
    """Fit one model by least squares; return a predict(t) function."""
    if name == "Linear":                      # y = a*t + b
        c = np.polyfit(t, y, 1)
        return lambda x: np.polyval(c, x)
    if name == "Parabola (2nd degree)":       # y = a*t^2 + b*t + c
        c = np.polyfit(t, y, 2)
        return lambda x: np.polyval(c, x)
    if name == "Exponential":                 # y = a*e^(b*t)
        f = lambda x, a, b: a * np.exp(b * x)
        p, _ = curve_fit(f, t, y, p0=[max(y.mean(), 1e-3), 0.0], maxfev=10000)
        return lambda x: f(np.asarray(x, float), *p)
    raise ValueError(name)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def r2(y, yhat):
    y = np.asarray(y)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot else float("nan")


def compare_models(y, n_test=None):
    """Fit every model; score on the last n_test months held out. Best first."""
    y = np.asarray(y, float)
    if len(y) < 6:
        raise ValueError("Need at least 6 monthly readings.")
    t = np.arange(1, len(y) + 1)
    n_test = min(3, len(y) // 4) if n_test is None else n_test
    cut = len(y) - n_test
    results = []
    for name in MODELS:
        try:
            full = fit_model(name, t, y)
            test_rmse = (rmse(y[cut:], fit_model(name, t[:cut], y[:cut])(t[cut:]))
                         if n_test else float("nan"))
            results.append(dict(model=name, r2=r2(y, full(t)), rmse=rmse(y, full(t)),
                                test_rmse=test_rmse, predict=full))
        except Exception:
            continue   # a model that fails to converge is simply skipped
    key = "test_rmse" if n_test else "rmse"
    return sorted(results, key=lambda r: r[key])


def forecast(predict, n_known, horizon):
    """Predicted units for the next `horizon` months (never below zero)."""
    t = np.arange(n_known + 1, n_known + horizon + 1)
    return t, np.clip(predict(t), 0, None)