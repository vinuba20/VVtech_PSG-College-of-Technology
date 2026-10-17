import streamlit as st, pandas as pd, numpy as np, joblib
import plotly.graph_objects as go
import xgboost as xgb

st.set_page_config(page_title="Diabetes Digital Twin", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("ehr.csv"), pd.read_csv("cgm.csv", parse_dates=["time"])

@st.cache_resource
def load_model():
    b = joblib.load("model.joblib")
    return b["model"], b["features"]

def add_features(df):
    df = df.sort_values("time").reset_index(drop=True)
    g = df["glucose"]
    df["g_lag3"], df["g_lag6"], df["g_lag12"] = g.shift(3), g.shift(6), g.shift(12)
    df["roc_15"], df["roc_30"] = g - df["g_lag3"], g - df["g_lag6"]
    df["mean_1h"] = g.rolling(12).mean()
    df["std_1h"] = g.rolling(12).std()
    df["max_1h"] = g.rolling(12).max()
    hr = df["time"].dt.hour + df["time"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    return df

ehr, cgm = load_data()
model, FEATURES = load_model()

st.title("🩺 Diabetes Digital Twin: Doctor Dashboard")
st.caption("Prototype using synthetic data only. Not for clinical use.")

# ---- Sidebar: choose patient ----
pid = st.sidebar.selectbox("Select virtual patient", ehr["patient_id"].tolist())
p = ehr[ehr["patient_id"] == pid].iloc[0]
st.sidebar.subheader("Patient record (EHR)")
st.sidebar.write(f"**Age:** {p['age']}")
st.sidebar.write(f"**BMI:** {p['bmi']}")
st.sidebar.write(f"**HbA1c:** {p['hba1c']}%")
st.sidebar.write(f"**Years with diabetes:** {p['diabetes_years']}")
st.sidebar.write(f"**On insulin:** {'Yes' if p['on_insulin'] else 'No'}")

df = add_features(cgm[cgm["patient_id"] == pid].copy().merge(ehr, on="patient_id"))
valid = df.dropna(subset=FEATURES).reset_index(drop=True)

# ---- Time selection ----
t = st.slider("Move through time (simulated live data)",
              min_value=valid["time"].iloc[0].to_pydatetime(),
              max_value=valid["time"].iloc[-1].to_pydatetime(),
              value=valid["time"].iloc[500].to_pydatetime(),
              step=pd.Timedelta(minutes=5).to_pytimedelta(),
              format="MMM DD, HH:mm")
t = pd.Timestamp(t)
row = valid.iloc[(valid["time"] - t).abs().argmin()]
X = pd.DataFrame([row[FEATURES].astype(float)])

# ---- Prediction ----
risk = float(model.predict_proba(X)[0, 1])
level = "HIGH" if risk > 0.6 else "MODERATE" if risk > 0.3 else "LOW"

c1, c2, c3 = st.columns(3)
c1.metric("Current glucose", f"{row['glucose']:.0f} mg/dL")
c2.metric("Risk of glucose > 180 in next 2 h", f"{risk*100:.0f}%")
c3.metric("Risk level", level)

if level == "HIGH":
    st.error("⚠️ Alert: high chance of a glucose spike within 2 hours. "
             "Consider reviewing diet, activity or insulin timing.")
elif level == "MODERATE":
    st.warning("Moderate risk. Keep monitoring.")
else:
    st.success("Low risk. Glucose expected to stay in range.")

# ---- Glucose chart ----
win = df[(df["time"] >= t - pd.Timedelta(hours=6)) &
         (df["time"] <= t + pd.Timedelta(hours=2))]
fig = go.Figure()
fig.add_trace(go.Scatter(x=win["time"], y=win["glucose"], mode="lines",
                         name="Glucose"))
fig.add_hline(y=180, line_dash="dash", line_color="red",
              annotation_text="180 mg/dL limit")
fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper",
              line=dict(color="green", width=2))
fig.add_vrect(x0=t, x1=t + pd.Timedelta(hours=2), fillcolor="orange",
              opacity=0.12, line_width=0)
fig.update_layout(title="Glucose: past 6 h and the 2 h prediction window (orange)",
                  yaxis_title="mg/dL", height=400)
st.plotly_chart(fig, use_container_width=True)
st.caption("The orange area shows what actually happened next in the simulation, "
           "so you can compare it with the prediction.")

# ---- Why this prediction? ----
contrib = model.get_booster().predict(
    xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)[0][:-1]
cdf = (pd.DataFrame({"feature": FEATURES, "impact": contrib})
       .assign(abs_=lambda d: d["impact"].abs())
       .sort_values("abs_", ascending=False).head(8).iloc[::-1])
fig2 = go.Figure(go.Bar(
    x=cdf["impact"], y=cdf["feature"], orientation="h",
    marker_color=["crimson" if v > 0 else "steelblue" for v in cdf["impact"]]))
fig2.update_layout(title="Why this prediction? (red = raises risk, blue = lowers risk)",
                   height=350)
st.plotly_chart(fig2, use_container_width=True)