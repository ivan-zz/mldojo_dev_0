
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Data Loading ──
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# ── Target and Feature Setup ──
target_cols = ['formation_energy_ev_natom', 'bandgap_energy_ev']
feature_cols = [c for c in train.columns if c not in target_cols and c != 'id']
X = train[feature_cols].values
Y = train[target_cols].values
X_test = test[feature_cols].values

# ── User Code ──
print(1)

# ── Evaluation (fallback) ──
# If the user code defines predict_model or train_and_predict but did NOT
# print a score, run cross-validation evaluation using those functions.
# Self-contained code that prints its own score does not need this block.
_has_api_functions = 'predict_model' in dir() or 'train_and_predict' in dir()

if _has_api_functions:
    from sklearn.model_selection import KFold

    def rmsle(y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 0, None)
        y_true_clipped = np.clip(y_true, 0, None)
        return np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2))

    def evaluate_model():
        _eval_train = pd.read_csv('input/train.csv')
        _tcols = ['formation_energy_ev_natom', 'bandgap_energy_ev']
        _fcols = [c for c in _eval_train.columns if c not in _tcols and c != 'id']
        _X = _eval_train[_fcols].values
        _Y = _eval_train[_tcols].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_scores = []
        for target_idx in range(_Y.shape[1]):
            y_col = _Y[:, target_idx]
            col_scores = []
            for train_idx, val_idx in kf.split(_X):
                X_tr, X_val = _X[train_idx], _X[val_idx]
                y_tr, y_val = y_col[train_idx], y_col[val_idx]
                try:
                    if 'predict_model' in dir():
                        y_pred = predict_model(X_tr, y_tr, X_val)
                    elif 'train_and_predict' in dir():
                        y_pred = train_and_predict(X_tr, y_tr, X_val)
                    else:
                        y_pred = np.zeros(len(X_val))
                    col_score = rmsle(y_val, y_pred)
                except Exception as e:
                    print(f"VALIDATION_ERROR: {e}", file=sys.stderr)
                    col_score = 9.999
                col_scores.append(col_score)
            all_scores.append(np.mean(col_scores))
        return np.mean(all_scores)

    try:
        final_score = evaluate_model()
        print(f"Final Validation Performance: {final_score:.6f}")
    except Exception as e:
        print(f"EXECUTION_ERROR: {e}", file=sys.stderr)
        sys.exit(1)
