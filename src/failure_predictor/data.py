import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

REQUIRED_TIME_COL = "timestamp"


def read_timeseries_csv(path: str, time_col: str = REQUIRED_TIME_COL) -> pd.DataFrame:
	"""Read CSV into a DataFrame, parse time, sort, drop duplicates.

	Assumes wide format where each column is a sensor/channel (e.g., vib, temp, rpm).
	"""
	df = pd.read_csv(path)
	if time_col not in df.columns:
		raise ValueError(f"Expected time column '{time_col}' in CSV. Found: {list(df.columns)}")
	df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
	df = df.dropna(subset=[time_col]).sort_values(time_col)
	df = df.drop_duplicates(subset=[time_col])
	df = df.reset_index(drop=True)
	return df


def infer_schema(df: pd.DataFrame, time_col: str = REQUIRED_TIME_COL) -> Tuple[str, List[str]]:
	"""Return (time_col, value_cols) inferring numeric columns as signals."""
	if time_col not in df.columns:
		raise ValueError(f"Missing '{time_col}' column")
	value_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
	if not value_cols:
		raise ValueError("No numeric signal columns detected")
	return time_col, value_cols


def validate_dataframe(df: pd.DataFrame, time_col: str, value_cols: List[str]) -> None:
	missing = [c for c in [time_col] + value_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	if df[time_col].isna().any():
		raise ValueError("Time column contains NaT values after parsing")
	for c in value_cols:
		if not pd.api.types.is_numeric_dtype(df[c]):
			raise ValueError(f"Column '{c}' must be numeric")


def resample_timeseries(df: pd.DataFrame, time_col: str, value_cols: List[str], rule: str = "1S", agg: str = "mean") -> pd.DataFrame:
	"""Resample to fixed frequency using pandas resample rule, forward-fill small gaps."""
	df = df.set_index(time_col)
	if agg == "mean":
		df_res = df[value_cols].resample(rule).mean()
	elif agg == "median":
		df_res = df[value_cols].resample(rule).median()
	else:
		df_res = df[value_cols].resample(rule).agg(agg)
	df_res = df_res.interpolate(limit=3).ffill().bfill()
	df_res.index.name = time_col
	return df_res.reset_index() 