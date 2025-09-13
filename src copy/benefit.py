
from collections import defaultdict
import pandas as pd
import numpy as np
from .utils import to_kzt, top_n_categories, clamp

TRAVEL_CATS = {"Путешествия", "Отели", "Такси"}
PREMIUM_4_CATS = {"Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"}
ONLINE_CATS = {"Едим дома", "Смотрим дома", "Играем дома"}

def monthly_spend(df_tx_kzt):
    if df_tx_kzt.empty:
        return {}
    df = df_tx_kzt.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ym"] = df["date"].dt.to_period("M")
    s = df.groupby("ym")["amount_kzt"].sum()
    return {str(k): float(v) for k,v in s.items()}

def spend_by_category(df_tx_kzt):
    if df_tx_kzt.empty:
        return {}
    s = df_tx_kzt.groupby("category")["amount_kzt"].sum()
    return {k: float(v) for k,v in s.items()}

def fx_share(df_tx):
    if df_tx.empty:
        return 0.0
    total = df_tx["amount_kzt"].sum()
    if total <= 0:
        return 0.0
    fx = df_tx.loc[df_tx["currency"].str.upper().ne("KZT"), "amount_kzt"].sum()
    return float(fx) / float(total)

def estimate_travel_card(spend_cat):
    base = sum(spend_cat.get(c,0.0) for c in TRAVEL_CATS)
    return 0.04 * base, {"travel_spend": base}

def premium_tier(balance):
    if balance >= 6_000_000:
        return 0.04
    elif balance >= 1_000_000:
        return 0.03
    else:
        return 0.02

def estimate_premium_card(spend_cat, total_spend, avg_balance):
    tier = premium_tier(avg_balance)
    base_cashback = tier * total_spend
    prem_extra_base = sum(spend_cat.get(c,0.0) for c in PREMIUM_4_CATS)
    incremental = max(0.0, 0.04 - tier) * prem_extra_base
    cap_per_month = 100_000.0
    months = 3
    cashback = min(base_cashback + incremental, cap_per_month * months)
    return cashback, {"tier": tier, "prem_extra_base": prem_extra_base, "total_spend": total_spend}

def estimate_credit_card(spend_cat):
    # 10% on top-3 categories + 10% on online services
    top3 = top_n_categories(spend_cat, 3)
    base_top3 = sum(spend_cat.get(c,0.0) for c in top3)
    online_extra = sum(spend_cat.get(c,0.0) for c in ONLINE_CATS)
    # avoid double counting if online cats are already in top3
    overlap = sum(spend_cat.get(c,0.0) for c in set(top3).intersection(ONLINE_CATS))
    benefit = 0.10 * (base_top3 + online_extra - overlap)
    return benefit, {"top3": top3, "online_spend": online_extra}

def estimate_fx(df_tx, fx_share_value):
    # if FX share significant, assume 1% spread saving on FX spend
    total = df_tx["amount_kzt"].sum()
    fx_spend = fx_share_value * total
    if fx_spend <= 0:
        return 0.0, {"fx_spend": 0.0}
    benefit = 0.01 * fx_spend
    return benefit, {"fx_spend": fx_spend}

def estimate_deposits(avg_balance, monthly_totals, fx_flag=False):
    # free balance heuristic: avg_balance - median monthly spend (>=0)
    if monthly_totals:
        med_spend = float(np.median(list(monthly_totals.values())))
    else:
        med_spend = 0.0
    free_bal = max(0.0, float(avg_balance) - med_spend)
    # annual rates -> 3 months interest
    months = 3.0 / 12.0
    sber = 0.165 * free_bal * months    # max rate, no access
    nako = 0.155 * free_bal * months    # top-up yes, withdraw no
    multi = (0.145 * free_bal * months) if fx_flag else 0.0
    return {"Депозит Сберегательный": sber,
            "Депозит Накопительный": nako,
            "Депозит Мультивалютный": multi}, {"free_balance": free_bal, "med_spend": med_spend}

def estimate_investments(avg_balance):
    # small nudge if there is free cash (proxy: avg_balance>0)
    return 0.001 * max(0.0, float(avg_balance)), {}

def estimate_cash_loan(avg_balance):
    # trigger only if balance is very low
    if avg_balance < 50_000:
        return 5_000.0, {}
    return 0.0, {}

def compute_benefits(profile_row, df_tx_client, rates):
    # prepare tx in KZT
    df_tx = df_tx_client.copy()
    if df_tx.empty:
        df_tx["amount_kzt"] = []
    else:
        df_tx["amount_kzt"] = df_tx.apply(lambda r: to_kzt(r["amount"], r["currency"], rates), axis=1)

    spend_cat = spend_by_category(df_tx)
    monthly_totals = monthly_spend(df_tx)
    total_spend = sum(spend_cat.values())
    avg_balance = float(profile_row.get("avg_monthly_balance_KZT", 0.0) or 0.0)

    fx_share_value = fx_share(df_tx)
    fx_flag = fx_share_value >= 0.10

    benefits = {}
    details = {}

    # Travel card
    b, d = estimate_travel_card(spend_cat)
    benefits["Карта для путешествий"] = b; details["Карта для путешествий"] = d

    # Premium card
    b, d = estimate_premium_card(spend_cat, total_spend, avg_balance)
    benefits["Премиальная карта"] = b; details["Премиальная карта"] = d

    # Credit card
    b, d = estimate_credit_card(spend_cat)
    benefits["Кредитная карта"] = b; details["Кредитная карта"] = d

    # FX
    b, d = estimate_fx(df_tx, fx_share_value)
    benefits["Обмен валют"] = b; details["Обмен валют"] = d

    # Deposits
    dep_b, dep_d = estimate_deposits(avg_balance, monthly_totals, fx_flag=fx_flag)
    for k,v in dep_b.items():
        benefits[k] = v
        details[k] = dep_d

    # Investments
    b, d = estimate_investments(avg_balance)
    benefits["Инвестиции"] = b; details["Инвестиции"] = d

    # Gold (tiny by default)
    benefits["Золотые слитки"] = 0.0; details["Золотые слитки"] = {}

    # Cash loan (only if need)
    b, d = estimate_cash_loan(avg_balance)
    benefits["Кредит наличными"] = b; details["Кредит наличными"] = d

    # rank
    top = sorted(benefits.items(), key=lambda kv: kv[1], reverse=True)
    return top, details, spend_cat, monthly_totals, fx_share_value
