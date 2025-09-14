from pathlib import Path
import pandas as pd
import numpy as np
import glob
from collections import defaultdict

BASE = Path(".")
DATA = Path("data") if (Path("data") / "clients.csv").exists() else BASE
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

clients = pd.read_csv(DATA / "clients.csv", encoding="utf-8-sig")
clients.columns = [c.strip() for c in clients.columns]
clients["client_code"] = clients["client_code"].astype(str)

def read_many(patterns):
    frames = []
    for pat in patterns:
        for p in glob.glob(str(pat), recursive=True):
            df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
            df.columns = [c.strip() for c in df.columns]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

tx_raw = read_many([DATA/"client_*_transactions_3m.csv", DATA/"**/client_*_transactions_3m.csv"])
tr_raw = read_many([DATA/"client_*_transfers_3m.csv",   DATA/"**/client_*_transfers_3m.csv"])

def extract_labels(df, src):
    if {"client_code","product"}.issubset(df.columns):
        out = df[["client_code","product"]].dropna().drop_duplicates().copy()
        out["client_code"] = out["client_code"].astype(str)
        out["source"] = src
        return out
    return pd.DataFrame(columns=["client_code","product","source"])

labels = pd.concat([extract_labels(tx_raw, "transactions"),
                    extract_labels(tr_raw, "transfers")], ignore_index=True).drop_duplicates()


tx = pd.DataFrame(columns=["date","category","amount","currency","client_code"])
tr = pd.DataFrame(columns=["date","type","direction","amount","currency","client_code"])
if not tx_raw.empty:
    keep = [c for c in ["date","category","amount","currency","client_code"] if c in tx_raw.columns]
    tx = tx_raw[keep].copy()
if not tr_raw.empty:
    keep = [c for c in ["date","type","direction","amount","currency","client_code"] if c in tr_raw.columns]
    tr = tr_raw[keep].copy()


def normalize(df):
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns: df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "client_code" in df.columns: df["client_code"] = df["client_code"].astype(str)
    for c in ("category","type","direction","currency"):
        if c in df.columns: df[c] = df[c].fillna("").astype(str)
    return df

tx = normalize(tx)
tr = normalize(tr)

RATES = {"KZT": 1.0, "USD": 500.0, "EUR": 540.0, "RUB": 5.0}
tx["amount_kzt"] = tx.apply(lambda r: float(r["amount"]) * RATES.get(r["currency"].upper(), 1.0), axis=1)
tr["amount_kzt"] = tr.apply(lambda r: float(r["amount"]) * RATES.get(r["currency"].upper(), 1.0), axis=1)

TRAVEL_CATS   = {"Путешествия","Такси","Отели"}
PREMIUM_BONUS = {"Ювелирные украшения","Косметика и Парфюмерия","Кафе и рестораны"}
ONLINE_CATS   = {"Едим дома","Смотрим дома","Играем дома"}

P = {
    "travel_cb": 0.04,
    "premium_tiers": [(6_000_000, 0.04), (1_000_000, 0.03), (0, 0.02)],
    "premium_bonus_cb": 0.04,
    "premium_cap": 100_000.0,
    "atm_fee_rate": 0.005,
    "atm_free_limit": 3_000_000.0,
    "cc_fav": 0.10,
    "cc_online": 0.10,
    "fx_saved": 0.003,
    "dep_rates": {"savings":0.165, "accum":0.155, "multi":0.145},

    "MIN_TRAVEL_MONTHLY": 40_000,
    "MIN_CC_SPEND": 150_000,
    "MIN_CC_TOP3_SHARE": 0.40,
    "MIN_FX_VOL": 80_000,
    "MIN_FREE_CASH_DEP": 120_000,
    "MAX_ATM_FOR_SAVINGS": 100_000,
    "MIN_INV_FREE": 20_000,
    "MAX_INV_FREE": 250_000,

    "PRIORITY": {
        "Карта для путешествий":   1.15,
        "Премиальная карта":       0.50,
        "Кредитная карта":         0.25,  
        "Обмен валют":             1.05,
        "Депозит Сберегательный":  1.00,
        "Депозит Накопительный":   1.05,
        "Депозит Мультивалютный":  1.10,
        "Инвестиции":              1.00,
        "Кредит наличными":        1.20, 
    },

    "CC_ALREADY_PENALTY": 0.5,   
    "CREDIT_STRESS_PENALTY": 0.6 
}

MONTH_PREP = {1:"январе",2:"феврале",3:"марте",4:"апреле",5:"мае",6:"июне",
              7:"июле",8:"августе",9:"сентябре",10:"октябре",11:"ноябре",12:"декабре"}

def last_month_label(series):
    if series is None or series.empty: return "последние месяцы"
    dt = pd.to_datetime(series.max())
    return MONTH_PREP.get(int(dt.month), "последние месяцы")

def premium_tier(balance: float) -> float:
    for th, rate in P["premium_tiers"]:
        if balance >= th:
            return rate
    return 0.02

def fmt_money(x: float) -> str:
    return f"{float(x):,.0f}".replace(",", " ") + " ₸"

def fmt_approx(x: float) -> str:
    return "≈" + f"{float(x):,.0f}".replace(",", " ") + " ₸"

def build_features(cid: str) -> dict:
    prof = clients.loc[clients["client_code"] == cid].iloc[0].to_dict()
    tx_c = tx[tx["client_code"] == cid].copy()
    tr_c = tr[tr["client_code"] == cid].copy()

    spend_3m = tx_c["amount_kzt"].sum() if not tx_c.empty else 0.0
    spend_mo = spend_3m / 3.0

    cat_agg = (tx_c.groupby("category")["amount_kzt"].sum().sort_values(ascending=False)
               if not tx_c.empty else pd.Series(dtype=float))
    cat_mo = (cat_agg / 3.0).to_dict()
    top3 = list(cat_agg.head(3).index) if not tx_c.empty else []
    top3_sum = sum(cat_mo.get(c, 0.0) for c in top3[:3])
    top3_share = (top3_sum / spend_mo) if spend_mo > 0 else 0.0

    travel_mo = sum(cat_mo.get(c, 0.0) for c in TRAVEL_CATS)
    online_mo = sum(cat_mo.get(c, 0.0) for c in ONLINE_CATS)

    last_lbl = last_month_label(tx_c["date"]) if not tx_c.empty else "последние месяцы"
    if not tx_c.empty:
        last_ym = tx_c["date"].max().to_period("M")
        taxi_last = tx_c[(tx_c["category"] == "Такси") & (tx_c["date"].dt.to_period("M") == last_ym)]
        taxi_cnt_last = len(taxi_last)
        taxi_sum_last = taxi_last["amount_kzt"].sum()
    else:
        taxi_cnt_last = 0
        taxi_sum_last = 0.0

    non_kzt_spend_mo = (tx_c.loc[tx_c["currency"].str.upper().ne("KZT"), "amount_kzt"].sum() / 3.0) if not tx_c.empty else 0.0

    tr_types = defaultdict(float)
    inflow_last = outflow_last = 0.0
    if not tr_c.empty:
        for t, s in tr_c.groupby("type")["amount_kzt"].sum().items():
            tr_types[t] += s / 3.0
        last_m = tr_c["date"].max().to_period("M")
        tr_last = tr_c[tr_c["date"].dt.to_period("M") == last_m]
        inflow_last = tr_last.loc[tr_last["direction"] == "in", "amount_kzt"].sum()
        outflow_last = tr_last.loc[tr_last["direction"] == "out", "amount_kzt"].sum()

    bal = float(prof.get("avg_monthly_balance_KZT", 0.0) or 0.0)
    buffer_ratio = (bal / spend_mo) if spend_mo > 0 else (10.0 if bal > 0 else 0.0)  
    free_cash = max(0.0, bal - 1.2 * spend_mo) 

    has_cc = (tr_types.get("cc_repayment_out", 0) > 0) or (tr_types.get("installment_payment_out", 0) > 0)
    credit_stress = (tr_types.get("loan_payment_out", 0) > 0)

    fx_vol_mo = tr_types.get("fx_buy", 0.0) + tr_types.get("fx_sell", 0.0) + non_kzt_spend_mo
    atm_mo = tr_types.get("atm_withdrawal", 0.0)
    invest_vol_mo = tr_types.get("invest_in", 0.0) + tr_types.get("invest_out", 0.0)

    return {
        "profile": prof,
        "spend_mo": spend_mo,
        "cat_mo": cat_mo,
        "top3": top3,
        "top3_sum": top3_sum,
        "top3_share": top3_share,
        "travel_mo": travel_mo,
        "online_mo": online_mo,
        "taxi_cnt_last": taxi_cnt_last,
        "taxi_sum_last": taxi_sum_last,
        "non_kzt_spend_mo": non_kzt_spend_mo,
        "fx_vol_mo": fx_vol_mo,
        "atm_mo": atm_mo,
        "invest_vol_mo": invest_vol_mo,
        "inflow_last": inflow_last,
        "outflow_last": outflow_last,
        "balance": bal,
        "buffer_ratio": buffer_ratio,
        "free_cash": free_cash,
        "has_cc": has_cc,
        "credit_stress": credit_stress,
        "last_month_label": last_lbl,
    }

def fit_travel(F):
    amt = F["travel_mo"]; rides = F["taxi_cnt_last"]
    f_amt = min(1.0, amt / max(1.0, P["MIN_TRAVEL_MONTHLY"] * 2.5))
    f_cnt = min(1.0, rides / 20.0) if rides > 0 else 0.0
    return max(f_amt, f_cnt)

def benefit_travel(F):
    if F["travel_mo"] < P["MIN_TRAVEL_MONTHLY"] and F["taxi_cnt_last"] < 5:
        return 0.0, 0.0
    cb = P["travel_cb"] * F["travel_mo"]
    return cb, fit_travel(F)

def benefit_premium(F):
    bal = F["balance"]
    tier = premium_tier(bal)
    gate = (bal >= 1_000_000) or (F["atm_mo"] > 200_000) or (F["profile"].get("status") == "Премиальный клиент")
    if not gate:
        return 0.0, 0.0
    base_cb = tier * F["spend_mo"]
    bonus = P["premium_bonus_cb"] * sum(F["cat_mo"].get(c, 0.0) for c in PREMIUM_BONUS)
    cb = min(base_cb + bonus, P["premium_cap"])
    saved_atm = P["atm_fee_rate"] * min(F["atm_mo"], P["atm_free_limit"])
    gain = cb + saved_atm
    bonus_spend = sum(F["cat_mo"].get(c, 0.0) for c in PREMIUM_BONUS)
    fit = min(1.0, (bal / 6_000_000) * 0.6 + (bonus_spend / 200_000) * 0.4)
    return gain, fit

def benefit_credit(F):
    if F["spend_mo"] < P["MIN_CC_SPEND"] or F["top3_share"] < P["MIN_CC_TOP3_SHARE"]:
        return 0.0, 0.0
    cb = P["cc_fav"] * F["top3_sum"] + P["cc_online"] * F["online_mo"]
    if F["has_cc"]:
        cb *= P["CC_ALREADY_PENALTY"]
    if F["credit_stress"]:
        cb *= P["CREDIT_STRESS_PENALTY"]
    fit = min(1.0, 0.6 * F["top3_share"] + 0.4 * min(1.0, F["online_mo"] / 80_000))
    return cb, fit

def benefit_fx(F):
    if F["fx_vol_mo"] < P["MIN_FX_VOL"]:
        return 0.0, 0.0
    gain = P["fx_saved"] * F["fx_vol_mo"]
    fit = min(1.0, F["fx_vol_mo"] / 300_000)
    return gain, fit

def benefit_deposits(F):
    free_cash = F["free_cash"]
    if free_cash < P["MIN_FREE_CASH_DEP"]:
        return {"savings": (0.0, 0.0), "accum": (0.0, 0.0), "multi": (0.0, 0.0)}
    rates = P["dep_rates"]
    gains = {
        "savings": rates["savings"] / 12.0 * free_cash,
        "accum":   rates["accum"]   / 12.0 * free_cash,
        "multi":   rates["multi"]   / 12.0 * free_cash,
    }
    if F["fx_vol_mo"] > 0:
        gains["multi"] *= 1.10
    if F["atm_mo"] > P["MAX_ATM_FOR_SAVINGS"]:
        gains["savings"] *= 0.6
    fit_sav = min(1.0, F["buffer_ratio"] / 3.0) * (0.7 if F["atm_mo"] <= P["MAX_ATM_FOR_SAVINGS"] else 0.4)
    fit_acc = min(1.0, F["buffer_ratio"] / 2.0)
    fit_mul = min(1.0, (F["fx_vol_mo"] / 200_000) + 0.3)
    return {"savings": (gains["savings"], fit_sav),
            "accum":   (gains["accum"],   fit_acc),
            "multi":   (gains["multi"],   fit_mul)}

def benefit_invest(F):
    free_cash = F["free_cash"]
    if not (P["MIN_INV_FREE"] <= free_cash <= P["MAX_INV_FREE"]) and F["invest_vol_mo"] == 0:
        return 0.0, 0.0
    gain = 0.001 * F["invest_vol_mo"] + 0.0005 * free_cash
    fit = min(1.0, (free_cash / 200_000) + (F["invest_vol_mo"] / 200_000))
    return gain, fit

def need_cash_loan(F) -> bool:
    gap = F["outflow_last"] - F["inflow_last"]
    return (gap > 250_000) and (F["balance"] < 150_000)

# Скоринг
PRODUCTS = [
    "Карта для путешествий", "Премиальная карта", "Кредитная карта", "Обмен валют",
    "Депозит Сберегательный", "Депозит Накопительный", "Депозит Мультивалютный", "Инвестиции"
]

def score_all(F):
    gains_fits = {}

    g, f = benefit_travel(F);  gains_fits["Карта для путешествий"] = (g, f)
    g, f = benefit_premium(F); gains_fits["Премиальная карта"]     = (g, f)
    g, f = benefit_credit(F);  gains_fits["Кредитная карта"]       = (g, f)
    g, f = benefit_fx(F);      gains_fits["Обмен валют"]           = (g, f)

    deps = benefit_deposits(F)
    gains_fits["Депозит Сберегательный"] = deps["savings"]
    gains_fits["Депозит Накопительный"]  = deps["accum"]
    gains_fits["Депозит Мультивалютный"] = deps["multi"]

    g, f = benefit_invest(F);  gains_fits["Инвестиции"] = (g, f)

    scores = {}
    for prod, (gain, fit) in gains_fits.items():
        pr = P["PRIORITY"].get(prod, 1.0)
        scores[prod] = gain * fit * pr

    if (max(scores.values()) if scores else 0.0) < 1.0 and need_cash_loan(F):
        return "Кредит наличными", 0.0, scores, gains_fits

    best_prod = max(scores.items(), key=lambda kv: kv[1])[0] if scores else "Инвестиции"
    return best_prod, scores.get(best_prod, 0.0), scores, gains_fits

# (TOV)
def push_text(prod, F, score_est):
    name = F["profile"].get("name", "Клиент")
    if prod == "Карта для путешествий":
        return (f"{name}, в {F['last_month_label']} у вас много поездок и такси на {fmt_money(F['travel_mo'])} в месяц. "
                f"Тревел-карта вернёт до {fmt_approx(P['travel_cb']*F['travel_mo'])}. Оформить карту.")[:220]
    if prod == "Премиальная карта":
        bal = F["balance"]; tier_txt = f"{int(premium_tier(bal)*100)}%"
        return (f"{name}, стабильный остаток {fmt_money(bal)} и активные траты. "
                f"Премиальная карта: до {tier_txt} кешбэка и бесплатные снятия — выгода {fmt_approx(score_est)}. "
                f"Оформить сейчас.")[:220]
    if prod == "Кредитная карта":
        cats = (F["top3"] + ["покупки","покупки","покупки"])[:3]
        if F.get("has_cc", False):
            return (f"{name}, у вас уже есть кредитная карта. Настройте 3 любимые категории "
                    f"({cats[0]}, {cats[1]}, {cats[2]}) и онлайн-сервисы — кешбэк до 10%. Настроить категории.")[:220]
        return (f"{name}, ваши топ-категории — {cats[0]}, {cats[1]}, {cats[2]}. "
                f"Кредитная карта даст до 10% в любимых категориях и на онлайн-сервисы. Оформить карту.")[:220]
    if prod == "Обмен валют":
        return (f"{name}, часто оплачиваете/меняете валюту. В приложении выгодный курс и авто-покупка по целевому — "
                f"экономия {fmt_approx(P['fx_saved']*F['fx_vol_mo'])} в месяц. Настроить обмен.")[:220]
    if prod == "Депозит Сберегательный":
        return (f"{name}, свободные средства лучше работают. Сберегательный вклад даст "
                f"{fmt_approx(P['dep_rates']['savings']/12*F['free_cash'])} в месяц. Открыть вклад.")[:220]
    if prod == "Депозит Накопительный":
        return (f"{name}, копите без снятий — с пополнением. Накопительный вклад: "
                f"{fmt_approx(P['dep_rates']['accum']/12*F['free_cash'])} в месяц. Открыть вклад.")[:220]
    if prod == "Депозит Мультивалютный":
        return (f"{name}, храните часть средств в валютах: мульти-вклад принесёт "
                f"{fmt_approx(P['dep_rates']['multi']/12*F['free_cash'])} в месяц. Открыть вклад.")[:220]
    if prod == "Инвестиции":
        return (f"{name}, начните с малого: без комиссий на старт и с авто-пополнением. "
                f"Подойдёт под свободные {fmt_money(F['free_cash'])}. Открыть счёт.")[:220]
    if prod == "Кредит наличными":
        return (f"{name}, если нужен запас на крупные траты — оформите кредит наличными с гибкими выплатами. "
                f"Узнать доступный лимит.")[:220]
    return "—"

results = []
debug_rows = []
top4_rows = []

for cid in clients["client_code"].astype(str):
    F = build_features(cid)
    best_prod, best_score, scores, gains_fits = score_all(F)
    push = push_text(best_prod, F, best_score)

    results.append({"client_code": cid, "product": best_prod, "push_notification": push})

    row = {"client_code": cid}
    for prod, (gain, fit) in gains_fits.items():
        row[f"{prod}__gain"] = gain
        row[f"{prod}__fit"] = fit
        row[f"{prod}__score"] = scores.get(prod, 0.0)
    row["chosen"] = best_prod
    row["score_chosen"] = best_score
    debug_rows.append(row)

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
    trow = {"client_code": cid}
    for i, (p, s) in enumerate(top, start=1):
        trow[f"top{i}_product"] = p
        trow[f"top{i}_score"] = s
    top4_rows.append(trow)

res_df = pd.DataFrame(results)
dbg_df = pd.DataFrame(debug_rows)
t4_df  = pd.DataFrame(top4_rows)

(res_df).to_csv(OUTPUTS/"results.csv", index=False, encoding="utf-8-sig")
(dbg_df).to_csv(OUTPUTS/"debug_scores.csv", index=False, encoding="utf-8-sig")
(t4_df).to_csv(OUTPUTS/"top4.csv", index=False, encoding="utf-8-sig")

print("Save:")
print(" -", (OUTPUTS/"push_results.csv").as_posix())
print(" -", (OUTPUTS/"push_debug_scores.csv").as_posix())
print(" -", (OUTPUTS/"push_top4.csv").as_posix())
