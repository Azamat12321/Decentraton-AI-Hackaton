
import os
import glob
import pandas as pd

def load_clients(data_dir):
    path = os.path.join(data_dir, "clients.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Не найден data/clients.csv")
    df = pd.read_csv(path)
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    return df

def load_transactions(tx_dir):
    # Merge multiple sources: single consolidated + per-client files
    frames = []
    # unified file
    uni = os.path.join(tx_dir, "transactions_3m.csv")
    if os.path.exists(uni):
        frames.append(pd.read_csv(uni))
    # per-client files
    for p in glob.glob(os.path.join(tx_dir, "client_*_transactions_3m.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["date","category","amount","currency","client_code"])
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_transfers(tr_dir):
    frames = []
    for p in glob.glob(os.path.join(tr_dir, "*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["date","type","direction","amount","currency","client_code"])
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return df
