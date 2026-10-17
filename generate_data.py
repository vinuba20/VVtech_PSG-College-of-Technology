import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N, DAYS, STEP = 100, 14, 5          # 100 patients, 14 days, reading every 5 min
n = DAYS * 24 * 60 // STEP
ehr, cgm = [], []

for pid in range(N):
    hba1c = round(float(rng.normal(7.8, 1.2)), 1)
    ehr.append({
        "patient_id": pid,
        "age": int(rng.integers(30, 75)),
        "bmi": round(float(rng.normal(27, 4)), 1),
        "hba1c": hba1c,
        "diabetes_years": int(rng.integers(0, 20)),
        "on_insulin": int(rng.random() < 0.3),
    })
    g = 90 + (hba1c - 5.5) * 12 + rng.normal(0, 5, n)   # baseline + noise
    for d in range(DAYS):
        for h in (8, 13, 20):                            # breakfast, lunch, dinner
            start = (d * 24 + h) * 60 // STEP + int(rng.integers(-6, 6))
            amp = rng.normal(60, 15) + (hba1c - 6) * 8
            k = np.arange(36)                            # 3-hour meal spike
            idx = start + k
            ok = idx < n
            g[idx[ok]] += amp * np.exp(-((k[ok] - 12) ** 2) / 72)
    t = pd.date_range("2026-01-01", periods=n, freq="5min")
    cgm.append(pd.DataFrame({"patient_id": pid, "time": t, "glucose": g.round(1)}))

pd.DataFrame(ehr).to_csv("ehr.csv", index=False)
pd.concat(cgm).to_csv("cgm.csv", index=False)
print("Done: ehr.csv and cgm.csv created")