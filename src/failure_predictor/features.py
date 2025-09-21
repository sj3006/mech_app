import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.signal import welch


def _rolling_features(series: pd.Series, windows: List[int]) -> pd.DataFrame:
	feats = {}
	for w in windows:
		roll = series.rolling(window=w, min_periods=max(5, w // 2))
		feats[f"mean_{w}"] = roll.mean()
		feats[f"std_{w}"] = roll.std(ddof=0)
		feats[f"min_{w}"] = roll.min()
		feats[f"max_{w}"] = roll.max()
		feats[f"p2p_{w}"] = feats[f"max_{w}"] - feats[f"min_{w}"]
		feats[f"rms_{w}"] = np.sqrt(roll.apply(lambda x: np.mean(np.square(x)), raw=True))
	return pd.DataFrame(feats)


def _spectral_bandpower(series: pd.Series, fs: float, bands: List[Tuple[float, float]]) -> pd.Series:
	x = series.fillna(method="ffill").fillna(method="bfill").to_numpy()
	if len(x) < 16:
		return pd.Series([np.nan] * len(bands))
	f, pxx = welch(x, fs=fs, nperseg=min(256, max(16, len(x)//2)))
	bp = []
	for lo, hi in bands:
		mask = (f >= lo) & (f < hi)
		bp.append(np.trapz(pxx[mask], f[mask]) if mask.any() else 0.0)
	return pd.Series(bp)


def build_feature_table(df: pd.DataFrame, time_col: str, value_cols: List[str], fs_hint_hz: float = 1.0) -> pd.DataFrame:
	"""Build feature table with rolling stats and spectral power per channel.

	fs_hint_hz: approximate sampling frequency in Hz for spectral features
	"""
	feat_frames = [df[[time_col]].copy()]
	for col in value_cols:
		roll = _rolling_features(df[col], windows=[10, 30, 60])
		roll.columns = [f"{col}_{c}" for c in roll.columns]
		bands = [(0.0, fs_hint_hz/8), (fs_hint_hz/8, fs_hint_hz/4), (fs_hint_hz/4, fs_hint_hz/2)]
		bp = _spectral_bandpower(df[col], fs=fs_hint_hz, bands=bands)
		bp.index = [f"{col}_bp_{i}" for i in range(len(bands))]
		bp_df = pd.DataFrame([bp.values] * len(df), columns=bp.index)
		feat_frames += [roll.reset_index(drop=True), bp_df]
	features = pd.concat(feat_frames, axis=1)
	features = features.dropna().reset_index(drop=True)
	return features 