
import os, joblib, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from .features import build_features
from ..data_loader import load_clients, load_transactions, load_transfers
from ..benefit import compute_benefits
from ..utils import load_rates

PRODUCTS = [
 "Карта для путешествий","Премиальная карта","Кредитная карта","Обмен валют",
 "Депозит Сберегательный","Депозит Накопительный","Депозит Мультивалютный",
 "Инвестиции","Золотые слитки","Кредит наличными"
]

def train_all(data_dir, tx_dir, tr_dir, models_dir):
    rates = load_rates(os.path.join(os.path.dirname(os.path.dirname(data_dir)), "exchange_rates.json"))
    clients = load_clients(data_dir)
    tx = load_transactions(tx_dir)
    tx_groups = dict(tuple(tx.groupby("client_code"))) if not tx.empty else {}

    X_rows, y_teacher = [], {p: [] for p in PRODUCTS}
    y_best_index = []

    for _, row in clients.iterrows():
        code = row.get("client_code")
        df_tx_client = tx_groups.get(code, pd.DataFrame(columns=tx.columns))
        top, details, spend_cat, monthly_totals, fx_share_value = compute_benefits(row, df_tx_client, rates)

        benefit_map = {p:0.0 for p in PRODUCTS}
        for p, val in top:
            benefit_map[p] = float(val)

        feat = build_features(row, df_tx_client, rates)
        X_rows.append(feat)
        for p in PRODUCTS:
            y_teacher[p].append(benefit_map[p])

        best = max(benefit_map.items(), key=lambda kv: kv[1])[0]
        y_best_index.append(PRODUCTS.index(best))

    X = pd.DataFrame(X_rows).fillna(0.0)

    regs = {}
    cv = KFold(n_splits=min(5, max(2, X.shape[0]//10)), shuffle=True, random_state=42)
    for p in PRODUCTS:
        y = np.array(y_teacher[p], dtype=float)
        if np.allclose(y, y[0]):
            regs[p] = None
            continue
        model = RandomForestRegressor(n_estimators=400, random_state=42)
        try:
            _ = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        except Exception:
            pass
        model.fit(X, y)
        regs[p] = model

    y_cls = np.array(y_best_index, dtype=int)
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
    try:
        _ = cross_val_score(clf, X, y_cls, cv=cv, scoring="accuracy")
    except Exception:
        pass
    clf.fit(X, y_cls)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump({"feature_columns": list(X.columns)}, os.path.join(models_dir, "meta.joblib"))
    joblib.dump(regs, os.path.join(models_dir, "benefit_regs.joblib"))
    joblib.dump(clf, os.path.join(models_dir, "best_product_clf.joblib"))
    return X.columns.tolist()
