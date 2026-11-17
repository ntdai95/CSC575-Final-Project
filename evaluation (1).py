"""Evaluation module (Maki). Scores multi-label tag predictions.

Input convention (agree this with Tan):
    Y_true  : (n_songs, n_tags) 0/1 array, columns in the order of `tags`
    Y_pred  : (n_songs, n_tags) 0/1 array of hard predictions
    Y_score : (n_songs, n_tags) float array (decision_function or predict_proba), optional
    tags    : list of tag names, e.g. shared_vocabulary_50.csv["tag"]
    groups  : (n_songs,) artist label per song, only for the clustered bootstrap
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def per_tag_report(Y_true, Y_pred, tags, Y_score=None):
    """One row per tag (precision, recall, f1, support, label_accuracy, ap),
    then 'micro' and 'macro' rows. Tags with no test positives get NaN ap and
    are left out of the macro mean of ap (they cannot be scored)."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    p, r, f, s = precision_recall_fscore_support(
        Y_true, Y_pred, average=None, zero_division=0)
    acc = (Y_true == Y_pred).mean(axis=0)
    df = pd.DataFrame({"tag": list(tags), "precision": p, "recall": r, "f1": f,
                       "support": s.astype(int), "label_accuracy": acc})
    if Y_score is not None:
        Y_score = np.asarray(Y_score)
        df["ap"] = [average_precision_score(Y_true[:, j], Y_score[:, j])
                    if Y_true[:, j].any() else np.nan for j in range(len(tags))]
    mi = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0)[:3]
    ma = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0)[:3]
    tot = [dict(tag="micro", precision=mi[0], recall=mi[1], f1=mi[2],
                support=int(Y_true.sum()), label_accuracy=acc.mean()),
           dict(tag="macro", precision=ma[0], recall=ma[1], f1=ma[2],
                support=int(Y_true.sum()), label_accuracy=acc.mean())]
    if Y_score is not None:
        tot[0]["ap"] = average_precision_score(Y_true, Y_score, average="micro")
        tot[1]["ap"] = np.nanmean(df["ap"])
    return pd.concat([df, pd.DataFrame(tot)], ignore_index=True)


def save_report(report, path):
    report.to_csv(path, index=False)


def _macro_f1(Y_true, Y_pred):
    return precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0)[2]


def bootstrap_compare(Y_true, pred_a, pred_b, n_boot=1000, groups=None,
                      metric=_macro_f1, seed=0, alpha=0.05):
    """95% bootstrap CI for metric(A) - metric(B) on the SAME resamples.
    groups=None resamples songs; groups=artist array resamples whole artists
    (clustered bootstrap). Returns dict(diff, lo, hi, excludes_zero)."""
    Y_true, pred_a, pred_b = map(np.asarray, (Y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(Y_true)
    if groups is not None:
        codes, uniq = pd.factorize(np.asarray(groups))
        members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[k] for k in pick])
        diffs[b] = metric(Y_true[idx], pred_a[idx]) - metric(Y_true[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(diff=metric(Y_true, pred_a) - metric(Y_true, pred_b),
                lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0))


def ci_widening(Y_true, pred_a, pred_b, groups, **kw):
    """Advanced item: how much wider is the artist-clustered interval?"""
    s = bootstrap_compare(Y_true, pred_a, pred_b, **kw)
    c = bootstrap_compare(Y_true, pred_a, pred_b, groups=groups, **kw)
    return dict(song_width=s["hi"] - s["lo"], artist_width=c["hi"] - c["lo"],
                ratio=(c["hi"] - c["lo"]) / (s["hi"] - s["lo"]))
