import gc
import os
import pandas as pd


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def _load(folder, filename):
    return pd.read_csv(os.path.join(folder, filename))


def load_all(folder):
    app = _load(folder, 'application_features.csv')
    df = app.copy()
    del app

    bureau = _load(folder, 'bureau_features.csv')
    df = df.merge(bureau, how='left', on=['SK_ID_CURR'])
    del bureau
    gc.collect()

    prev_app = _load(folder, 'previous_application_features.csv')
    df = df.merge(prev_app, how='left', on=['SK_ID_CURR'])
    del prev_app
    gc.collect()

    installments = _load(folder, 'installments_payments_features.csv')
    df = df.merge(installments, how='left', on=['SK_ID_CURR'])
    del installments
    gc.collect()

    pos_cash = _load(folder, 'pos_cash_features.csv')
    df = df.merge(pos_cash, how='left', on=['SK_ID_CURR'])
    del pos_cash
    gc.collect()

    credit_card = _load(folder, 'credit_card_features.csv')
    df = df.merge(credit_card, how='left', on=['SK_ID_CURR'])
    del credit_card
    gc.collect()

    return df
