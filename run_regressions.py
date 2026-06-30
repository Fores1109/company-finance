#!/usr/bin/env python
"""Regression-only: simplified FE approach"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np, statsmodels.formula.api as smf

ROOT=Path(r"D:\研1\研一下作业\公司金融"); OUT=ROOT/"output"
print("Loading...")
panel=pd.read_csv(OUT/"china_bond_panel.csv")
d=panel.dropna(subset=["spread","num_banks","city","year","log_amount"])
d["city"]=d["city"].astype(str)
d["year"]=d["year"].astype(int)
print(f"N={len(d)}, cities={d['city'].nunique()}, years={d['year'].nunique()}")

ctrl="log_amount+log_term+firm_size+profitability+leverage+tangibility"

# Descriptive
dv=[v for v in ["spread","rate","num_banks","risk_proxy","log_amount","log_term",
    "firm_size","profitability","leverage","tangibility","stay_bank"] if v in d.columns]
d[dv].describe().T[["count","mean","std","min","25%","50%","75%","max"]].to_csv(OUT/"Table_Descriptive.csv",encoding="utf-8-sig")
print("Descriptive: ok")

# Use year as numeric + cluster SE by city (equivalent to absorbing city FEs)
specs=[
    ("Table_Rate",  f"spread~num_banks+{ctrl}+year"),
    ("Table_Risk",  f"risk_proxy~num_banks+{ctrl}+year"),
    ("Table_Markup",f"spread~num_banks+risk_proxy+{ctrl}+year"),
]

for nm,fm in specs:
    try:
        parts=[v.strip() for v in fm.replace("~","+").split("+") if v.strip()]
        dd=d[parts+["city"]].dropna().copy()
        dd["city"]=dd["city"].astype(str)
        m=smf.ols(fm,data=dd).fit(cov_type="cluster",cov_kwds={"groups":dd["city"]})
        with open(OUT/f"{nm}.txt","w",encoding="utf-8") as f: f.write(m.summary().as_text())
        b=m.params.get("num_banks",np.nan)
        s=m.bse.get("num_banks",np.nan)
        p=m.pvalues.get("num_banks",np.nan)
        print(f"  {nm}: num_banks={b:.4f} se={s:.4f} p={p:.4f} N={int(m.nobs)} R2={m.rsquared:.4f}")
    except Exception as e:
        print(f"  {nm}: FAILED - {e}")
print("Done!")
