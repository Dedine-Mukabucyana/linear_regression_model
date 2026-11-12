import re
import sys
import time

import pandas as pd
import requests

BASE = "https://api.dhsprogram.com/rest/dhs"
session = requests.Session()

# Physical and sexual violence by husband/partner in the last 12 months
TARGET_ID = "DV_SPV1_W_PAS"

# (new column name, words that must be in the label, words to avoid)
NATIONAL_FEATURES = [
    ("women_literate_pct", ["literate"], ["not literate", "men", "number"]),
    ("women_secondary_edu_pct", ["women", "secondary"], ["men", "number", "median"]),
    ("women_employed_pct", ["women", "employed"], ["men", "number", "unemployed", "not employed"]),
    ("women_mobile_phone_pct", ["mobile"], ["men", "number"]),
    ("women_bank_account_pct", ["bank account"], ["men", "number"]),
    ("women_accept_wife_beating_pct", ["wife beating", "at least one"], ["men", "number"]),
    ("women_median_age_first_marriage", ["median age at first marriage"], ["men"]),
    ("total_fertility_rate", ["total fertility rate"], []),
]


def fetch_all(endpoint, **params):
    rows, page = [], 1
    while True:
        p = dict(params, f="json", perpage=5000, page=page)
        for attempt in range(3):
            try:
                r = session.get(f"{BASE}/{endpoint}", params=p, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    sys.exit(f"API request failed: {e}")
                time.sleep(2)
        js = r.json()
        rows += js.get("Data", [])
        if page >= js.get("TotalPages", 1):
            break
        page += 1
    return pd.DataFrame(rows)


def find_indicator(ind, must, avoid=()):
    labels = ind["Label"].str.lower()
    mask = pd.Series(True, index=ind.index)
    for w in must:
        mask &= labels.str.contains(w.lower(), regex=False)
    for w in avoid:
        # whole-word match, so "men" does not match "women"
        mask &= ~labels.str.contains(r"\b" + re.escape(w.lower()) + r"\b", regex=True)
    return ind[mask]


print("Downloading indicator list ...")
ind = fetch_all("indicators")
print("Using target:", TARGET_ID)
print(ind[ind["IndicatorId"] == TARGET_ID][["IndicatorId", "Label"]].to_string(index=False))

df = fetch_all("data", indicatorIds=TARGET_ID, breakdown="all")
print("Raw target rows:", len(df))

keep = ["SurveyId", "CountryName", "SurveyYear", "CharacteristicCategory",
        "CharacteristicLabel", "DenominatorWeighted", "Value"]
df = df[[c for c in keep if c in df.columns]].copy()
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df["DenominatorWeighted"] = pd.to_numeric(df["DenominatorWeighted"], errors="coerce")
df["SurveyYear"] = pd.to_numeric(df["SurveyYear"], errors="coerce")
df = df.dropna(subset=["Value", "SurveyYear"])
df = df[~df["CharacteristicCategory"].isin(["Total", "Region"])]
df = df[df["CharacteristicLabel"].notna()]

for name, must, avoid in NATIONAL_FEATURES:
    m = find_indicator(ind, must, avoid)
    if m.empty:
        print(f"[skip] {name}: no indicator found")
        continue
    print(f"[ok]   {name}: candidates ->")
    print(m[["IndicatorId", "Label"]].head(3).to_string(index=False))
    iid = m.iloc[0]["IndicatorId"]
    nat = fetch_all("data", indicatorIds=iid, breakdown="national")
    if nat.empty:
        continue
    nat["Value"] = pd.to_numeric(nat["Value"], errors="coerce")
    nat = nat.groupby("SurveyId", as_index=False)["Value"].mean().rename(columns={"Value": name})
    df = df.merge(nat, on="SurveyId", how="left")

df = df.drop(columns=["SurveyId"]).rename(columns={"Value": "violence_pct"})
df = df.dropna(axis=1, thresh=int(0.5 * len(df)))

X = df.drop(columns="violence_pct")
n_features = X.shape[1]
print("\nShape:", df.shape)
print("Feature columns:", n_features)
print("Numeric features:", X.select_dtypes(include="number").shape[1])
print("Categorical features:", X.select_dtypes(exclude="number").shape[1])
print("Rows >= 1000:", len(df) >= 1000, "| Features >= 8:", n_features >= 8)

df.to_csv("violence_dataset.csv", index=False)
print("\nSaved violence_dataset.csv")
print(df.head())