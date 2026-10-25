import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rossmann Sales Forecast", page_icon="📈", layout="wide")
st.title("📈 Rossmann Store Sales Forecast")
st.write("Backtest demo: pick a store and see how the model predicted its daily sales "
         "for 6 weeks (20 Jun to 31 Jul 2015), and why.")


@st.cache_resource
def load_all():
    art = joblib.load("models/lgbm_v2.joblib")
    data = pd.read_csv("app_data.csv", parse_dates=["Date"])
    model, feats = art["model"], art["features"]
    explainer = shap.TreeExplainer(model)
    return model, feats, data, explainer


model, feats, data, explainer = load_all()

# ---------- Store chuno ----------
store = st.sidebar.selectbox("Store", sorted(data["Store"].unique()))
sd = data[data["Store"] == store].sort_values("Date").copy()
sd["Predicted"] = np.expm1(model.predict(sd[feats]))
sd["Error %"] = (sd["Predicted"] - sd["Sales"]) / sd["Sales"] * 100

c1, c2, c3 = st.columns(3)
c1.metric("Store", f"#{store}")
c2.metric("Average daily sales", f"{sd['Sales'].mean():,.0f}")
c3.metric("Average error (MAPE)", f"{sd['Error %'].abs().mean():.1f}%")

st.subheader("Actual vs predicted")
chart = sd.set_index("Date")[["Sales", "Predicted"]].rename(columns={"Sales": "Actual"})
st.line_chart(chart)

# ---------- Ek din chuno ----------
st.subheader("Explain one day")
days = sd["Date"].dt.date.tolist()
day = st.selectbox("Date (only days the store was open)", days)
row = sd[sd["Date"].dt.date == day].iloc[[0]]
x = row[feats].copy()

orig_promo = int(row["Promo"].iloc[0])
promo = st.radio("What-if: promo on this day?", [0, 1], index=orig_promo, horizontal=True,
                 format_func=lambda v: "Promo ON" if v == 1 else "Promo OFF")
x["Promo"] = promo

pred = float(np.expm1(model.predict(x)[0]))
actual = float(row["Sales"].iloc[0])
orig_pred = float(row["Predicted"].iloc[0])

m1, m2, m3 = st.columns(3)
m1.metric("Predicted", f"{pred:,.0f}",
          delta=f"{pred - orig_pred:+,.0f} vs original" if promo != orig_promo else None)
m2.metric("Actual", f"{actual:,.0f}")
m3.metric("Error", f"{(pred - actual) / actual * 100:+.1f}%")

if promo != orig_promo:
    st.info("This is a what-if: actual sales are from the real promo setting, "
            "so the error is not meaningful here.")

# ---------- SHAP ----------
st.subheader("Why this prediction?")
sv = explainer.shap_values(x)[0]
base = float(np.ravel(explainer.expected_value)[0])
exp = shap.Explanation(values=sv, base_values=base,
                       data=x.iloc[0].values, feature_names=feats)
plt.figure()
shap.plots.waterfall(exp, show=False)
st.pyplot(plt.gcf())
st.caption("Values are in log scale. A SHAP value of +0.2 means this feature pushed the prediction "
           "up by about 22% (e^0.2 = 1.22); -0.2 means about 18% down.")

with st.expander("About this demo and its limits"):
    st.write("""
- LightGBM trained on 2013 to mid-June 2015 data, target is log(1 + Sales).
- Lag features (sales 364 days ago, 28-day average) are built with a gap of at least 42 days,
  matching a 6-week forecast horizon, so there is no leakage.
- This is a backtest on a held-out window, not a live forecast.
- `Customers` was deliberately not used: it is unknown at prediction time.
""")