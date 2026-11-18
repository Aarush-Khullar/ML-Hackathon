"""Model definitions (initial / untuned configurations) for the 8-model comparison."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb

SEED = 42


class TorchlessMLP(BaseEstimator, ClassifierMixin):
    """sklearn MLP trained with explicit partial_fit epochs so that epoch-wise train/val loss can be recorded."""
    def __init__(self, hidden=(64, 32), epochs=15, batch=1024, lr=1e-3, alpha=1e-4, seed=SEED):
        self.hidden, self.epochs, self.batch, self.lr, self.alpha, self.seed = hidden, epochs, batch, lr, alpha, seed

    def fit(self, X, y, X_val=None, y_val=None):
        self.sc_ = StandardScaler().fit(X); Xs = self.sc_.transform(X)
        self.m_ = MLPClassifier(hidden_layer_sizes=self.hidden, learning_rate_init=self.lr, alpha=self.alpha,
                                batch_size=self.batch, random_state=self.seed)
        rng = np.random.RandomState(self.seed); self.classes_ = np.array([0, 1]); self.history_ = []
        Xv = self.sc_.transform(X_val) if X_val is not None else None
        for ep in range(self.epochs):
            perm = rng.permutation(len(Xs))
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]; self.m_.partial_fit(Xs[b], y[b], classes=self.classes_)
            if Xv is not None:
                tl = log_loss(y, np.clip(self.m_.predict_proba(Xs[:200000])[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1]) if False else None
                trp = self.m_.predict_proba(Xs[:100000])[:, 1]
                self.history_.append(dict(epoch=ep + 1, train_loss=log_loss(y[:100000], np.clip(trp, 1e-7, 1 - 1e-7)),
                                          val_loss=log_loss(y_val, np.clip(self.m_.predict_proba(Xv)[:, 1], 1e-7, 1 - 1e-7))))
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(self.sc_.transform(X))


def zoo():
    return {
        "M1_LogReg": ("Linear/probabilistic", "C=1.0, lbfgs, max_iter=200, standardised inputs",
                      lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=200))),
        "M2_GaussianNB": ("Linear/probabilistic", "var_smoothing=1e-9",
                          lambda: GaussianNB()),
        "M3_DecisionTree": ("Tree-based", "max_depth=12, min_samples_leaf=20",
                            lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED)),
        "M4_RandomForest": ("Ensemble (bagging)", "n_estimators=60, max_depth=14, min_samples_leaf=10, max_samples=0.5",
                            lambda: RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=10, max_samples=0.5,
                                                           n_jobs=8, random_state=SEED)),
        "M5_HistGB": ("Boosting (sklearn)", "max_iter=200, lr=0.1, max_leaf_nodes=31",
                      lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)),
        "M6_LightGBM": ("Boosting (LightGBM)", "n_estimators=300, lr=0.05, num_leaves=63, min_child_samples=20",
                        lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                                                   random_state=SEED, n_jobs=8, verbose=-1)),
        "M7_XGBoost": ("Boosting (XGBoost)", "n_estimators=300, lr=0.1, max_depth=6, hist",
                       lambda: xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                                                 random_state=SEED, n_jobs=8, eval_metric="logloss")),
        "M8_MLP": ("Neural network", "hidden=(64,32), adam lr=1e-3, alpha=1e-4, batch=1024, 15 epochs",
                   lambda: TorchlessMLP()),
    }
