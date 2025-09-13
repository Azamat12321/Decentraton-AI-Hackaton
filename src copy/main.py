
import os
import pandas as pd
from .data_loader import load_clients, load_transactions, load_transfers
from .benefit import compute_benefits
from .utils import load_rates
from .push_gen import gen_push

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TX_DIR = os.path.join(DATA_DIR, "transactions")
TR_DIR = os.path.join(DATA_DIR, "transfers")

def run():
    rates = load_rates(os.path.join(os.path.dirname(os.path.dirname(__file__)), "exchange_rates.json"))
    clients = load_clients(DATA_DIR)
    tx = load_transactions(TX_DIR)
    tr = load_transfers(TR_DIR)  # not used in baseline, but left for future

    # normalize
    for col in ["currency","category"]:
        if col in tx.columns:
            tx[col] = tx[col].fillna("").astype(str)

    out_rows = []
    # index tx by client for speed
    tx_groups = dict(tuple(tx.groupby("client_code"))) if not tx.empty else {}

    for _, row in clients.iterrows():
        code = row.get("client_code")
        df_tx_client = tx_groups.get(code, pd.DataFrame(columns=tx.columns))

        top, details, spend_cat, monthly_totals, fx_share_value = compute_benefits(row, df_tx_client, rates)
        # choose best product (top[0])
        product = top[0][0] if top else "Инвестиции"
        push = gen_push(product, row, df_tx_client, details, spend_cat, fx_share_value)

        out_rows.append({"client_code": code, "product": product, "push_notification": push})

    out = pd.DataFrame(out_rows)
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "result.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(out)} rows ->", out_path)

if __name__ == "__main__":
    run()
