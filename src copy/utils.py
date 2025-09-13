
import math
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os

MONTHS_GEN = {
    1: "январе", 2: "феврале", 3: "марте", 4: "апреле",
    5: "мае", 6: "июне", 7: "июле", 8: "августе",
    9: "сентябре", 10: "октябре", 11: "ноябре", 12: "декабре"
}
MONTHS_NOM = {
    1: "январь", 2: "февраль", 3: "март", 4: "апрель",
    5: "май", 6: "июнь", 7: "июль", 8: "август",
    9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь"
}

def load_rates(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"KZT":1.0}

def to_kzt(amount, currency, rates):
    r = rates.get(str(currency).upper(), 1.0)
    return float(amount) * float(r)

def kzt(amount, decimals=0):
    # space as thousands sep, comma as decimal
    if decimals == 0:
        s = f"{int(round(amount)):,}".replace(",", " ")
        return f"{s} ₸"
    else:
        s = f"{amount:,.{decimals}f}".replace(",", " ").replace(".", ",")
        return f"{s} ₸"

def month_name_gen(dt):
    if isinstance(dt, str):
        try:
            dt = pd.to_datetime(dt)
        except Exception:
            dt = datetime.today()
    return MONTHS_GEN.get(dt.month, "этом месяце")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def top_n_categories(spend_by_cat, n=3):
    items = sorted(spend_by_cat.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k,_ in items[:n]]
