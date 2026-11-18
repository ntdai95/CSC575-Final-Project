"""Sanity checks on fake data. Run: python test_evaluation.py"""
import numpy as np
from sklearn.metrics import f1_score
from evaluation import per_tag_report, bootstrap_compare, ci_widening

rng = np.random.default_rng(1)
n, k = 2000, 5
tags = [f"t{i}" for i in range(k)]
prev = np.array([0.4, 0.2, 0.1, 0.03, 0.01])
Y = (rng.random((n, k)) < prev).astype(int)
good = np.clip(Y + rng.normal(0, 0.4, Y.shape), 0, 1)      # informative scores
bad = rng.random((n, k))                                    # random scores
pred_good, pred_bad = (good > 0.5).astype(int), (bad > 0.5).astype(int)

rep = per_tag_report(Y, pred_good, tags, good)
assert abs(rep.loc[rep.tag == "macro", "f1"].item()
           - f1_score(Y, pred_good, average="macro")) < 1e-12
assert (rep.loc[:k - 1, "support"].values == Y.sum(0)).all()
# rare tag: always predicting 0 gives high accuracy but zero F1
zero = np.zeros_like(Y)
r0 = per_tag_report(Y, zero, tags)
assert r0.label_accuracy[k - 1] > 0.98 and r0.f1[k - 1] == 0
# AP of random scores is near prevalence, informative scores far above
assert rep.ap[0] > 0.9 and abs(per_tag_report(Y, pred_bad, tags, bad).ap[0] - 0.4) < 0.05

# bootstrap: clear winner excludes zero; identical classifiers do not
assert bootstrap_compare(Y, pred_good, pred_bad, 300)["excludes_zero"]
assert not bootstrap_compare(Y, pred_good, pred_good, 300)["excludes_zero"]
# interval narrows as the test set grows
w = []
for m in (250, 1000, 2000):
    r = bootstrap_compare(Y[:m], pred_good[:m], pred_bad[:m], 300)
    w.append(r["hi"] - r["lo"])
assert w[0] > w[1] > w[2], w
# clustered interval is wider when songs within an artist are correlated
art = rng.integers(0, 100, n)
Yc = Y.copy(); noise = rng.random((100, k)) < 0.3
Yc = np.where(noise[art], 1 - Yc, Yc)   # whole-artist label flips
cw = ci_widening(Yc, pred_good, pred_bad, art, n_boot=300)
print("widths", w, "clustered ratio", round(cw["ratio"], 2))
assert cw["ratio"] > 1
print("all checks passed")
