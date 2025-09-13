
import os
import pandas as pd
from .data_loader import load_clients, load_transactions
from .ml.infer import predict_for_client
from .utils import load_rates
from .push_gen import gen_push

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TX_DIR = os.path.join(DATA_DIR, "transactions")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def run():
    clients = load_clients(DATA_DIR)
    tx = load_transactions(TX_DIR)

    tx_groups = dict(tuple(tx.groupby("client_code"))) if not tx.empty else {}
    out_rows = []

    for _, row in clients.iterrows():
        code = row.get("client_code")
        df_tx_client = tx_groups.get(code, pd.DataFrame(columns=tx.columns))

        top, details, spend_cat, monthly_totals, fx_share_value = predict_for_client(row, df_tx_client, BASE_DIR, MODELS_DIR)
        product = top[0][0] if top else "Инвестиции"
        push = gen_push(product, row, df_tx_client, details, spend_cat, fx_share_value)

        out_rows.append({"client_code": code, "product": product, "push_notification": push})

    out = pd.DataFrame(out_rows)
    out_path = os.path.join(BASE_DIR, "outputs", "result_ml.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(out)} rows ->", out_path)

if __name__ == "__main__":
    run()
