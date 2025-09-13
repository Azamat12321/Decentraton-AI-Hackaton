
import os, joblib
import numpy as np
import pandas as pd
from .features import build_features
from ..benefit import compute_benefits
from ..utils import load_rates

PRODUCTS = [
 "Карта для путешествий","Премиальная карта","Кредитная карта","Обмен валют",
 "Депозит Сберегательный","Депозит Накопительный","Депозит Мультивалютный",
 "Инвестиции","Золотые слитки","Кредит наличными"
]

def load_models(models_dir):
    try:
        meta = joblib.load(os.path.join(models_dir, "meta.joblib"))
        regs = joblib.load(os.path.join(models_dir, "benefit_regs.joblib"))
        clf  = joblib.load(os.path.join(models_dir, "best_product_clf.joblib"))
        return meta, regs, clf
    except Exception:
        return None, None, None

def predict_for_client(profile_row, df_tx_client, data_root, models_dir):
    rates = load_rates(os.path.join(data_root, "exchange_rates.json"))
    meta, regs, clf = load_models(models_dir)

    # If no model — fallback to rule-based
    if meta is None or regs is None or clf is None:
        top, details, spend_cat, monthly_totals, fx_share_value = compute_benefits(profile_row, df_tx_client, rates)
        return top, {"fallback":"rules"}, spend_cat, monthly_totals, fx_share_value

    feat = build_features(profile_row, df_tx_client, rates)
    import pandas as pd
    X = pd.DataFrame([feat])
    # align columns
    for c in meta["feature_columns"]:
        if c not in X.columns:
            X[c] = 0.0
    X = X[meta["feature_columns"]]

    # regressors → benefits
    benefits = {}
    for p, reg in regs.items():
        if reg is None:
            benefits[p] = 0.0
        else:
            benefits[p] = float(reg.predict(X)[0])

    # rank
    top = sorted(benefits.items(), key=lambda kv: kv[1], reverse=True)
    # we can compute details via rules for push text context
    from ..benefit import compute_benefits as rules_compute
    top_rules, details, spend_cat, monthly_totals, fx_share_value = rules_compute(profile_row, df_tx_client, rates)
    return top, details, spend_cat, monthly_totals, fx_share_value
