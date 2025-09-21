import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


SUPERVISED_MODEL_PATH = "models/supervised_model.joblib"
UNSUPERVISED_MODEL_PATH = "models/unsupervised_iforest.joblib"


def train_supervised_model(X: pd.DataFrame, y: pd.Series, save_path: str = SUPERVISED_MODEL_PATH) -> Dict[str, Any]:
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	clf = GradientBoostingClassifier(random_state=42)
	clf.fit(X_train, y_train)
	probs = clf.predict_proba(X_val)[:, 1]
	preds = (probs >= 0.5).astype(int)
	metrics = {
		"f1": f1_score(y_val, preds),
		"roc_auc": roc_auc_score(y_val, probs),
		"num_features": X.shape[1],
	}
	joblib.dump({"model": clf, "feature_names": list(X.columns)}, save_path)
	return metrics


def train_unsupervised_model(X: pd.DataFrame, save_path: str = UNSUPERVISED_MODEL_PATH) -> Dict[str, Any]:
	iso = IsolationForest(random_state=42, contamination="auto", n_estimators=300)
	iso.fit(X)
	joblib.dump({"model": iso, "feature_names": list(X.columns)}, save_path)
	return {"n_estimators": iso.n_estimators_, "features": X.shape[1]}


def load_model(path: str) -> Dict[str, Any]:
	return joblib.load(path)


def predict_with_model(model_bundle: Dict[str, Any], X: pd.DataFrame, task: str = "supervised") -> Tuple[np.ndarray, np.ndarray]:
	feature_names = model_bundle.get("feature_names")
	X_in = X[feature_names] if feature_names else X
	model = model_bundle["model"]
	if task == "supervised":
		probs = model.predict_proba(X_in)[:, 1]
		preds = (probs >= 0.5).astype(int)
		return preds, probs
	else:
		scores = -model.decision_function(X_in)
		preds = (scores > np.percentile(scores, 95)).astype(int)
		return preds, scores 