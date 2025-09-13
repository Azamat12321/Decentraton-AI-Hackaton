
import pandas as pd
import numpy as np
from collections import defaultdict
from ..utils import to_kzt

ALL_CATEGORIES = [
 "Одежда и обувь","Продукты питания","Кафе и рестораны","Медицина","Авто","Спорт","Развлечения","АЗС","Кино","Питомцы","Книги","Цветы",
 "Едим дома","Смотрим дома","Играем дома","Косметика и Парфюмерия","Подарки","Ремонт дома","Мебель","Спа и массаж","Ювелирные украшения",
 "Такси","Отели","Путешествия"
]

def month_agg(df_tx):
    if df_tx.empty:
        return {"months":0,"tot_per_month_mean":0.0,"tot_per_month_std":0.0}
    df = df_tx.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ym"] = df["date"].dt.to_period("M")
    g = df.groupby("ym")["amount_kzt"].sum()
    return {"months": int(g.shape[0]),
            "tot_per_month_mean": float(g.mean() if len(g)>0 else 0.0),
            "tot_per_month_std": float(g.std() if len(g)>1 else 0.0)}

def build_features(profile, df_tx, rates):
    feat = {}

    # base profile
    feat["age"] = float(profile.get("age", 0) or 0.0)
    feat["avg_balance"] = float(profile.get("avg_monthly_balance_KZT", 0) or 0.0)

    # tx in KZT
    dfx = df_tx.copy()
    if not dfx.empty:
        dfx["amount_kzt"] = dfx.apply(lambda r: to_kzt(r["amount"], r["currency"], rates), axis=1)
    else:
        dfx["amount_kzt"] = []

    # totals
    tot = float(dfx["amount_kzt"].sum() if not dfx.empty else 0.0)
    feat["total_spend"] = tot
    feat["fx_share"] = float(dfx.loc[dfx["currency"].str.upper().ne("KZT"), "amount_kzt"].sum() / tot) if tot>0 else 0.0

    # category spends + shares
    if not dfx.empty:
        s = dfx.groupby("category")["amount_kzt"].sum()
        cat2sum = {k: float(v) for k,v in s.items()}
    else:
        cat2sum = {}

    for cat in ALL_CATEGORIES:
        v = cat2sum.get(cat, 0.0)
        feat[f"sum_{cat}"] = v
        feat[f"share_{cat}"] = (v / tot) if tot>0 else 0.0

    # online bundle
    online = ["Едим дома","Смотрим дома","Играем дома"]
    feat["sum_online"] = sum(cat2sum.get(c,0.0) for c in online)
    feat["share_online"] = feat["sum_online"]/tot if tot>0 else 0.0

    # travel bundle
    travel = ["Путешествия","Отели","Такси"]
    feat["sum_travel"] = sum(cat2sum.get(c,0.0) for c in travel)
    feat["share_travel"] = feat["sum_travel"]/tot if tot>0 else 0.0

    # month stats
    feat.update(month_agg(dfx))

    # one-hot status (simple)
    status = str(profile.get("status","")).strip()
    for st in ["Студент","Зарплатный клиент","Премиальный клиент","Стандартный клиент"]:
        feat[f"status_{st}"] = 1.0 if status==st else 0.0

    # drop NaNs
    for k,v in list(feat.items()):
        if v is None or (isinstance(v,float) and (np.isnan(v) or np.isinf(v))):
            feat[k] = 0.0

    return feat
