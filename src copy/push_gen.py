
from .utils import kzt, month_name_gen

def pick_best_month_for_category(df_tx, target_cats):
    # Return month name (prepositional) with highest spend in target_cats
    import pandas as pd
    if df_tx.empty:
        return "этом месяце"
    df = df_tx.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ym"] = df["date"].dt.to_period("M")
    df["is_target"] = df["category"].isin(list(target_cats))
    s = df.groupby(["ym", "is_target"])["amount_kzt"].sum().unstack(fill_value=0)
    if s.empty:
        return "этом месяце"
    best_ym = s[True].idxmax()
    return month_name_gen(pd.Period(best_ym).to_timestamp())

def gen_push(product, profile, df_tx, details, spend_cat, fx_share_value):
    name = profile.get("name", "Клиент")
    balance = float(profile.get("avg_monthly_balance_KZT", 0) or 0)
    city = profile.get("city", "")
    status = str(profile.get("status", "")).strip()

    if product == "Карта для путешествий":
        travel_spend = details.get(product, {}).get("travel_spend", 0.0)
        m = pick_best_month_for_category(df_tx, {"Путешествия","Отели","Такси"})
        cashback = 0.04 * travel_spend
        msg = (f"{name}, в {m} у вас много поездок и такси. "
               f"С картой для путешествий вернули бы ≈{kzt(cashback)}. "
               f"Откройте карту.")
        return msg

    if product == "Премиальная карта":
        tier = details.get(product, {}).get("tier", 0.02)
        tier_pct = int(round(tier*100))
        msg = (f"{name}, у вас стабильный остаток — это ваш ресурс. "
               f"Премиальная карта даст до 4% кешбэка и бесплатные снятия/переводы. "
               f"Оформить сейчас.")
        return msg

    if product == "Кредитная карта":
        top3 = details.get(product, {}).get("top3", [])[:3]
        cats = ", ".join(top3) if top3 else "ваши любимые категории"
        msg = (f"{name}, ваши топ-категории — {cats}. "
               f"Кредитная карта даст до 10% кешбэка и 10% на онлайн-сервисы. "
               f"Оформить карту.")
        return msg

    if product == "Обмен валют":
        fx_spend = details.get(product, {}).get("fx_spend", 0.0)
        fx_curr = "валюте"  # без точности к USD/EUR
        msg = (f"{name}, часто платите в {fx_curr}. "
               f"В приложении — выгодный обмен и авто-покупка по целевому курсу. "
               f"Настроить обмен.")
        return msg

    if product in {"Депозит Сберегательный","Депозит Накопительный","Депозит Мультивалютный"}:
        msg = (f"{name}, у вас остаются свободные средства. "
               f"Разместите их на вкладе — удобно копить и получать вознаграждение. "
               f"Открыть вклад.")
        return msg

    if product == "Инвестиции":
        msg = (f"{name}, попробуйте инвестиции с низким порогом входа и без комиссий на старт. "
               f"Открыть счёт.")
        return msg

    if product == "Кредит наличными":
        msg = (f"{name}, если нужен запас на крупные траты — можно оформить кредит наличными "
               f"с гибкими выплатами. Узнать доступный лимит.")
        return msg

    # fallback
    return (f"{name}, у вас есть возможность получить выгоду с нашим продуктом. "
            f"Посмотрите подробности в приложении. Открыть.")
