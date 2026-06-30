#!/usr/bin/env python
"""Optimized: use openpyxl read_only + only necessary columns"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

ROOT = Path(r"D:\研1\研一下作业\公司金融")
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

BENCHMARK_RATES = [
    ("2014-11-22","2015-02-28",5.60,6.15),("2015-03-01","2015-05-10",5.35,5.90),
    ("2015-05-11","2015-06-27",5.10,5.65),("2015-06-28","2015-08-25",4.85,5.40),
    ("2015-08-26","2015-10-23",4.60,5.15),("2015-10-24","2019-08-19",4.35,4.90),
]
RATING_MAP = {"AAA":1,"AA+":2,"AA":3,"AA-":4,"A+":5,"A":6,"A-":7,
    "BBB+":8,"BBB":9,"BBB-":10,"BB+":11,"BB":12,"BB-":13,
    "B+":14,"B":15,"B-":16,"CCC":17,"CC":18,"C":19}
def winsorize(s, lo=0.01, hi=0.99):
    qlo, qhi = s.quantile(lo), s.quantile(hi)
    return s.clip(qlo, qhi)

def load_lpr():
    files = sorted(DATA.glob("lpr_*.xlsx"))
    dfs = [pd.read_excel(f) for f in files]
    lpr = pd.concat(dfs, ignore_index=True)
    lpr["date"] = pd.to_datetime(lpr["日期"], errors="coerce")
    lpr["lpr_1y"] = pd.to_numeric(lpr["1Y"], errors="coerce")
    lpr["lpr_5y"] = pd.to_numeric(lpr["5Y"], errors="coerce")
    lpr = lpr[["date","lpr_1y","lpr_5y"]].dropna().sort_values("date").drop_duplicates("date",keep="last")
    daily = pd.DataFrame({"date": pd.date_range("2015-01-01","2024-12-31",freq="D")})
    for s, e, r1, r5 in BENCHMARK_RATES:
        m = (daily["date"]>=s)&(daily["date"]<=e)
        daily.loc[m,"benchmark_1y"]=r1; daily.loc[m,"benchmark_5y"]=r5
    for _,r in lpr.iterrows():
        daily.loc[daily["date"]>=r["date"],"benchmark_1y"]=r["lpr_1y"]
        daily.loc[daily["date"]>=r["date"],"benchmark_5y"]=r["lpr_5y"]
    daily["benchmark_1y"]=daily["benchmark_1y"].ffill()
    daily["benchmark_5y"]=daily["benchmark_5y"].ffill()
    daily["ym"]=daily["date"].dt.to_period("M").astype(str)
    return daily.groupby("ym")[["benchmark_1y","benchmark_5y"]].last().reset_index()

def read_cols(folder, fn, cols):
    return pd.read_excel(DATA/folder/fn, usecols=cols)

def main():
    print("="*60)
    print("Step 1: Loading...")
    lpr = load_lpr()
    print(" LPR: ok")

    # Bonds
    cols=["Liscd","Orgid","Conme","Bndtype","Intrrate","Acisuquty","Term","Ipodt","Matdt","Crdrate"]
    print(" Reading bond_issuance...")
    bonds=read_cols("bond_issuance","bond_issuance.xlsx",cols)
    bonds["Ipodt_dt"]=pd.to_datetime(bonds["Ipodt"],errors="coerce")
    bonds=bonds.dropna(subset=["Ipodt_dt"])
    bonds["year"]=bonds["Ipodt_dt"].dt.year
    bonds["ym"]=bonds["Ipodt_dt"].dt.to_period("M").astype(str)
    bonds=bonds[(bonds["year"]>=2015)&(bonds["year"]<=2024)]
    bonds["rate"]=pd.to_numeric(bonds["Intrrate"],errors="coerce")
    bonds=bonds.dropna(subset=["rate"]); bonds=bonds[(bonds["rate"]>0)&(bonds["rate"]<20)]
    bonds["amount"]=pd.to_numeric(bonds["Acisuquty"],errors="coerce")
    bonds=bonds.dropna(subset=["amount"]); bonds=bonds[bonds["amount"]>0]
    bonds["term"]=pd.to_numeric(bonds["Term"],errors="coerce")
    bonds["Liscd"]=bonds["Liscd"].astype(str).str.strip()
    print(f" Bonds: {len(bonds)}")

    # Issuers
    print(" Reading issuer_info...")
    iss=read_cols("issuer_info","issuer_info.xlsx",["InstitutionID","Province"])
    iss=iss.rename(columns={"InstitutionID":"Orgid"})
    bonds["Orgid"]=bonds["Orgid"].astype(str).str.strip()
    iss["Orgid"]=iss["Orgid"].astype(str).str.strip()
    bonds=bonds.merge(iss[["Orgid","Province"]],on="Orgid",how="left")
    bonds["city"]=bonds["Province"]; bonds=bonds.dropna(subset=["city"])
    print(f" Issuers merge: {len(bonds)}")

    # Underwriters
    print(" Reading underwriters...")
    uws=[]
    for fn in ["bond_underwriters1.xlsx","bond_underwriters2.xlsx"]:
        uws.append(read_cols("bond_underwriters",fn,["InterBankCode","SHHCode","SHZCode","BSECode","AgencyName","IssueDate"]))
    uw=pd.concat(uws,ignore_index=True)
    uwl=[]
    for cc in ["InterBankCode","SHHCode","SHZCode","BSECode"]:
        sub=uw[[cc,"AgencyName","IssueDate"]].dropna(subset=[cc]).copy()
        sub["bond_code"]=sub[cc].astype(str).str.strip()
        sub["uw_name"]=sub["AgencyName"].astype(str).str.strip()
        uwl.append(sub[["bond_code","uw_name","IssueDate"]])
    uwl=pd.concat(uwl,ignore_index=True)
    uwl["uw_year"]=pd.to_datetime(uwl["IssueDate"],errors="coerce").dt.year
    bonds=bonds.merge(uwl[["bond_code","uw_name","uw_year"]],left_on="Liscd",right_on="bond_code",how="left")
    print(f" UW merge: {len(bonds)}")

    # Ratings
    print(" Reading ratings...")
    rt=read_cols("bond_ratings","bond_ratings.xlsx",["Liscd","RatingDate","Btcr"])
    rt["Liscd"]=rt["Liscd"].astype(str).str.strip()
    rt["RatingDate_dt"]=pd.to_datetime(rt["RatingDate"],errors="coerce")
    rt["rating_year"]=rt["RatingDate_dt"].dt.year
    rt=rt.sort_values("RatingDate_dt").drop_duplicates(["Liscd","rating_year"],keep="last")
    rt["issuer_rating"]=rt["Btcr"].astype(str).str.strip()
    bonds=bonds.merge(rt[["Liscd","rating_year","issuer_rating"]],left_on=["Liscd","year"],right_on=["Liscd","rating_year"],how="left")
    bonds["risk_proxy"]=bonds["issuer_rating"].map(RATING_MAP)
    print(f" Ratings merge: {len(bonds)}")

    # Balance sheet
    print(" Reading balance sheet...")
    bs=read_cols("firm_balance_sheet","firm_balance_sheet.xlsx",["Liscd","Accper","A001000000","A001212000","A001218000","A001220000","A002000000"])
    bs["Liscd"]=bs["Liscd"].astype(str).str.strip()
    bs["Accper_dt"]=pd.to_datetime(bs["Accper"],errors="coerce")
    bs["firm_year"]=bs["Accper_dt"].dt.year
    bonds=bonds.merge(bs[["Liscd","firm_year","A001000000","A001212000","A001218000","A001220000","A002000000"]],left_on=["Liscd","year"],right_on=["Liscd","firm_year"],how="left")
    print(f" BS merge: {len(bonds)}")

    # Income
    print(" Reading income...")
    inc=read_cols("firm_income","firm_income.xlsx",["Liscd","Accper","B001101000","B001300000","B001000000","B002000000","B001211100"])
    inc["Liscd"]=inc["Liscd"].astype(str).str.strip()
    inc["Accper_dt"]=pd.to_datetime(inc["Accper"],errors="coerce")
    inc["firm_year"]=inc["Accper_dt"].dt.year
    bonds=bonds.merge(inc[["Liscd","firm_year","B001101000","B001300000","B001000000","B002000000","B001211100"]],left_on=["Liscd","year"],right_on=["Liscd","firm_year"],how="left",suffixes=("","_inc"))
    print(f" Inc merge: {len(bonds)}")

    # Defaults
    print(" Reading defaults...")
    dfd=read_cols("bond_defaults","bond_defaults.xlsx",["BondID"])
    dfd["BondID"]=dfd["BondID"].astype(str).str.strip()
    dfd["default_flag"]=1
    bonds=bonds.merge(dfd[["BondID","default_flag"]].drop_duplicates("BondID"),left_on="Liscd",right_on="BondID",how="left")
    bonds["default_flag"]=bonds["default_flag"].fillna(0).astype(int)

    # Variables
    print("="*60)
    print("Step 2: Variables...")
    panel=bonds.merge(lpr,on="ym",how="left")
    panel["spread"]=panel["rate"]-panel["benchmark_1y"]
    panel["firm_size"]=np.log(panel["A001000000"].fillna(0)+1)
    panel["ebitda"]=panel["B001300000"].fillna(panel["B001000000"])
    panel["profitability"]=panel["ebitda"]/panel["A001000000"].replace(0,np.nan)
    panel["leverage"]=panel["A002000000"]/panel["A001000000"].replace(0,np.nan)
    panel["intangibles"]=panel["A001218000"].fillna(0)+panel["A001220000"].fillna(0)
    panel["tangibility"]=(panel["A001000000"]-panel["intangibles"])/panel["A001000000"].replace(0,np.nan)
    panel["log_amount"]=np.log(panel["amount"]+1)
    panel["log_term"]=np.log(panel["term"].fillna(1)+1)

    mkt=panel.groupby(["city","year"])["uw_name"].nunique().reset_index(name="num_banks")
    panel=panel.merge(mkt,on=["city","year"],how="left")
    vol=panel.groupby(["city","year"])["amount"].sum().reset_index(name="city_volume")
    panel=panel.merge(vol,on=["city","year"],how="left")
    panel["log_volume"]=np.log(panel["city_volume"]+1)
    panel=panel.sort_values(["Orgid","Ipodt_dt"])
    panel["prev_uw"]=panel.groupby("Orgid")["uw_name"].shift(1)
    panel["stay_bank"]=(panel["uw_name"]==panel["prev_uw"]).astype(int)

    for c in ["rate","spread","profitability","leverage","tangibility"]:
        if c in panel.columns: panel[c]=winsorize(panel[c])

    key=["spread","num_banks","city","year","log_amount"]
    rs=panel.dropna(subset=key)
    print(f" Sample: {len(rs)} obs")

    sc=[c for c in ["Liscd","Orgid","Conme","year","city","rate","spread","amount","log_amount","term","log_term","risk_proxy","issuer_rating","num_banks","city_volume","log_volume","stay_bank","uw_name","firm_size","profitability","leverage","tangibility","benchmark_1y","default_flag"] if c in rs.columns]
    rs[sc].to_csv(OUT/"china_bond_panel.csv",index=False,encoding="utf-8-sig")
    print(f" Saved panel")

    # Regressions
    print("="*60)
    print("Step 3: Regressions...")
    d=rs.copy()
    ctrl="log_amount+log_term+firm_size+profitability+leverage+tangibility"

    dv=[v for v in ["spread","rate","num_banks","risk_proxy","log_amount","log_term","firm_size","profitability","leverage","tangibility","stay_bank"] if v in d.columns]
    desc=d[dv].describe().T
    desc[["count","mean","std","min","25%","50%","75%","max"]].to_csv(OUT/"Table_Descriptive.csv",encoding="utf-8-sig")
    print(" Descriptive: ok")

    specs={
        "Table_Rate_vs_Banks":f"spread~num_banks+{ctrl}+ C(year)",
        "Table_Risk_vs_Banks":f"risk_proxy~num_banks+{ctrl}+ C(year)",
        "Table_Markup":f"spread~num_banks+risk_proxy+{ctrl}+ C(year)",
    }
    for nm,fm in specs.items():
        try:
            m=smf.ols(fm,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["city"]})
            with open(OUT/f"{nm}.txt","w",encoding="utf-8") as f: f.write(m.summary().as_text())
            b=m.params.get("num_banks",np.nan)
            s=m.bse.get("num_banks",np.nan)
            p=m.pvalues.get("num_banks",np.nan)
            print(f" {nm}: num_banks={b:.4f} se={s:.4f} p={p:.4f} N={int(m.nobs)} R2={m.rsquared:.4f}")
        except Exception as e:
            print(f" {nm}: FAILED - {e}")
    print("="*60)
    print("Done!")

if __name__=="__main__":
    main()


