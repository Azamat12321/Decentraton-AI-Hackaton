
import os
from .ml.trainer import train_all

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TX_DIR = os.path.join(DATA_DIR, "transactions")
TR_DIR = os.path.join(DATA_DIR, "transfers")
MODELS_DIR = os.path.join(BASE_DIR, "models")

if __name__ == "__main__":
    cols = train_all(DATA_DIR, TX_DIR, TR_DIR, MODELS_DIR)
    print("Trained. Features:", len(cols))
