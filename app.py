# Copyright (c) 2026 Anurak
"""Run with:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fitting import compare_models, forecast
from tariff import bill, DEFAULT_SLABS

st.set_page_config(page_title="Energy forecaster", layout="wide")
st.title("Electricity consumption forecaster")
st.caption("Fits curves to your monthly readings, forecasts future units and estimates the bill.")

# ---- inputs -------------------------------------------------------------
up = st.sidebar.file_uploader("Monthly readings (CSV with a 'units' column)", type="csv")
df = pd.read_csv(up) if up else pd.read_csv("sample_data.csv")
if "units" not in df.columns:
    st.error("Your CSV needs a column named 'units' (kWh per month).")
    st.stop()
if not up:
    st.info("Showing sample data. Upload your own bills in the sidebar.")
y = df["units"].astype(float).to_numpy()

horizon = st.sidebar.slider("Months to forecast", 1, 12, 3)
fixed = st.sidebar.number_input("Fixed monthly charge", min_value=0.0, value=0.0)
st.sidebar.subheader("Tariff slabs")
st.sidebar.caption("Leave 'slab_units' empty for the last slab. Defaults are placeholders.")
slab_df = st.sidebar.data_editor(pd.DataFrame(DEFAULT_SLABS, columns=["slab_units", "rate"]),
                                 num_rows="dynamic", hide_index=True)
slabs = [(None if pd.isna(s) else float(s), float(r))
         for s, r in zip(slab_df["slab_units"], slab_df["rate"]) if not pd.isna(r)]

# ---- fitting ------------------------------------------------------------
try:
    results = compare_models(y)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.subheader("Model comparison")
st.caption("Ranked by error on the last months held out from fitting (test_rmse). Lower is better.")
st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != "predict"} for r in results]).round(2),
             hide_index=True)

choice = st.selectbox("Model to use", [r["model"] for r in results])
best = next(r for r in results if r["model"] == choice)
t = np.arange(1, len(y) + 1)
ft, fy = forecast(best["predict"], len(y), horizon)

# ---- plot ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(t, y, label="Readings", color="black")
tt = np.linspace(1, ft[-1], 300)
ax.plot(tt, best["predict"](tt), label=choice)
ax.scatter(ft, fy, marker="D", label="Forecast", color="tab:red")
ax.set_xlabel("Month")
ax.set_ylabel("Units (kWh)")
ax.legend()
st.pyplot(fig)

# ---- forecast + bill ----------------------------------------------------
st.subheader("Forecast and estimated bill")
cut = st.slider("What if I reduce usage by (%)", 0, 50, 0)
out = pd.DataFrame({"month": ft, "units": fy.round(1)})
out["bill"] = [bill(u, slabs, fixed) for u in fy]
out["bill_after_cut"] = [bill(u * (1 - cut / 100), slabs, fixed) for u in fy]
st.dataframe(out, hide_index=True)
st.metric("Estimated saving over the forecast period", round(out["bill"].sum() - out["bill_after_cut"].sum(), 2))