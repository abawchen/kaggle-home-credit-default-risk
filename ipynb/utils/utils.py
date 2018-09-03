import datetime
import json
import os
import time
from colorama import Fore, Style
from contextlib import contextmanager


def highlight_print(hightlight, message):
    print(hightlight + message + Style.RESET_ALL)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    diffs = time.time() - t0
    minutes = int(diffs/60)
    seconds = int(diffs)%60
    highlight_print(
        Fore.LIGHTGREEN_EX,
        "[Done] {} at {}".format(title, datetime.datetime.now())
    )


def calculate_feature_importance(df, model_folder):
    df = df[["feature", "importance"]] \
        .groupby("feature") \
        .mean() \
        .sort_values(by="importance", ascending=False)
    with open(os.path.join(model_folder, 'feats_importance.json'), 'w') as f:
        d = dict((i, r) for i, r in df['importance'].iteritems())
        json.dump(d, f, indent=2)
    return df


def submit(name, test_df, preds):
    submission_filename = os.path.join('..', 'data', 'submission', '{}.csv'.format(name))
    print(submission_filename)
    test_df['TARGET'] = preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_filename, index=False)


def load_feats(folder, name):
    filename = os.path.join(folder, name)
    try:
        with open(os.path.join(folder, name), 'r') as f:
            return json.load(f)
    except:
        pass
    return None

def diff_feats(cats_folder, dogs_folder):
    cats = set(load_feats(cat_folder))
    dogs = set(load_feats(dogs_folder))
    return (cats - dog), (dogs - cats)
