"""Linear probes for "candidate name is in scope" on the residual stream.

Two probe designs (a third, cheap variant is kept for comparison):

  use      Feature = residual at the use position `pred_pos` (the token containing the `%`,
           whose next-token logits emit the name). Label = the name that appears *next* is in
           scope. The candidate enters implicitly: it is the name the program emits there.
           Positives are use sites of valid programs; negatives are the same sites in a
           renamed copy where the used name's definition (and its earlier uses) got a fresh
           name, so the identical upcoming token is now out of scope (see
           `positions.corrupt_use`), plus natural violations in model generations. A probe that
           separates these reads the scope set at the emission point.

  def_diff Feature = residual at the use's `%` token minus the residual at the `%` token of the
           candidate's own definition line (a function parameter or the LHS of a statement).
           Label = the candidate is in scope at the use. Negatives are candidates the same
           function defines *after* the use, so positives and negatives come from the same
           program and the candidate enters through its own definition site. This is the harder
           probe: it must relate two positions rather than read a summary at one. Caveat: under
           the validator's flat scope "in scope" is equivalent to "defined before the use", and a
           difference of two residuals carries position-like signal (e.g. attention-to-BOS decay)
           from the first block on, so this design can read order rather than scope; compare its
           layer profile against `use`, which is at chance at the embedding layer by construction.

  name     (comparison) Feature = residual at the first token of the name, i.e. after the model
           has read the candidate. Label as for `use`. Measures decodability once the name is
           visible, the setting of variable-role probes for code models.

Training: L2-regularised logistic regression in numpy (standardised features, full-batch
Adam), grouped 5-fold cross-validation (examples from one program stay in one fold), a
shuffled-label control run with the same pipeline, and a bootstrap CI on the out-of-fold
accuracies via `sci.eval.stats.bootstrap_ci`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sci.eval.stats import bootstrap_ci

DESIGNS = ("use", "def_diff", "name")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass
class LogisticProbe:
    """Standardised logistic regression with L2 penalty, fitted by full-batch Adam."""
    l2: float = 1e-2
    steps: int = 300
    lr: float = 0.05
    w: np.ndarray | None = None
    b: float = 0.0
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def _z(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticProbe":
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = self._z(X)
        n, d = Z.shape
        w = np.zeros(d, np.float32); b = 0.0
        mw = np.zeros(d, np.float32); vw = np.zeros(d, np.float32); mb = vb = 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.steps + 1):
            p = _sigmoid(Z @ w + b)
            g = p - y
            gw = Z.T @ g / n + self.l2 * w
            gb = float(g.mean())
            mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
            c1, c2 = 1 - b1 ** t, 1 - b2 ** t
            w -= self.lr * (mw / c1) / (np.sqrt(vw / c2) + eps)
            b -= self.lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        self.w, self.b = w, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self._z(np.asarray(X, dtype=np.float32)) @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def group_folds(groups: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """Indices of k folds such that every group lands in exactly one fold."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    assign = {g: i % k for i, g in enumerate(uniq)}
    fold_of = np.array([assign[g] for g in groups])
    return [np.flatnonzero(fold_of == i) for i in range(k)]


def cv_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, k: int = 5, seed: int = 0,
                l2: float = 1e-2, steps: int = 300) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold correctness indicator (n,) and per-fold accuracies."""
    y = np.asarray(y, int)
    n = len(y)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    correct = np.zeros(n, int)
    accs = []
    for te in group_folds(groups, k, seed):
        if len(te) == 0:
            continue
        tr = np.setdiff1d(np.arange(n), te)
        if len(np.unique(y[tr])) < 2:
            pred = np.full(len(te), int(round(y[tr].mean())))
        else:
            pred = LogisticProbe(l2=l2, steps=steps).fit(X[tr], y[tr]).predict(X[te])
        correct[te] = (pred == y[te]).astype(int)
        accs.append(float(correct[te].mean()))
    return correct, accs


def evaluate_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, k: int = 5, seed: int = 0,
                   l2: float = 1e-2, steps: int = 300, n_boot: int = 2000) -> dict:
    """CV accuracy with bootstrap CI, the shuffled-label control, and the majority-class rate."""
    y = np.asarray(y, int)
    correct, accs = cv_accuracy(X, y, groups, k, seed, l2, steps)
    rng = np.random.default_rng(seed + 1)
    y_shuf = y.copy(); rng.shuffle(y_shuf)
    c_ctrl, accs_ctrl = cv_accuracy(X, y_shuf, groups, k, seed, l2, steps)
    ci = bootstrap_ci(correct, n_resamples=n_boot, seed=seed)
    ci_c = bootstrap_ci(c_ctrl, n_resamples=n_boot, seed=seed)
    return {
        "n": int(len(y)), "pos_rate": float(y.mean()), "majority": float(max(y.mean(), 1 - y.mean())),
        "acc": ci.point, "ci_low": ci.ci_low, "ci_high": ci.ci_high, "fold_acc": accs,
        "control_acc": ci_c.point, "control_ci_low": ci_c.ci_low, "control_ci_high": ci_c.ci_high,
    }


def probe_by_layer(feats: dict[int, np.ndarray], y: np.ndarray, groups: np.ndarray | None = None, **kw) -> dict[int, dict]:
    """`evaluate_probe` for every layer in `feats` ({layer: (n, d)})."""
    return {int(l): evaluate_probe(X, y, groups, **kw) for l, X in sorted(feats.items())}
