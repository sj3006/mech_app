import argparse
from pathlib import Path
import pandas as pd
from failure_predictor.data import read_timeseries_csv, infer_schema, validate_dataframe, resample_timeseries
from failure_predictor.features import build_feature_table
from failure_predictor.model import train_supervised_model, train_unsupervised_model, SUPERVISED_MODEL_PATH, UNSUPERVISED_MODEL_PATH


def main():
	parser = argparse.ArgumentParser(description="Train failure prediction models from CSV")
	parser.add_argument("csv", type=str, help="Path to input CSV")
	parser.add_argument("--label_col", type=str, default=None, help="Binary label column for supervised training")
	parser.add_argument("--freq", type=str, default="1S", help="Resample rule, e.g., 1S, 100ms, 10min")
	parser.add_argument("--fs_hint", type=float, default=1.0, help="Sampling frequency hint (Hz) for spectral features")
	parser.add_argument("--out_sup", type=str, default=SUPERVISED_MODEL_PATH, help="Path to save supervised model")
	parser.add_argument("--out_unsup", type=str, default=UNSUPERVISED_MODEL_PATH, help="Path to save unsupervised model")
	args = parser.parse_args()

	df = read_timeseries_csv(args.csv)
	time_col, value_cols = infer_schema(df)
	if args.label_col:
		if args.label_col not in df.columns:
			raise ValueError(f"Label column '{args.label_col}' not found")
		label = df[args.label_col].astype(int)
	else:
		label = None

	df_res = resample_timeseries(df, time_col, value_cols, rule=args.freq)
	features = build_feature_table(df_res, time_col, value_cols, fs_hint_hz=args.fs_hint)
	X = features.drop(columns=[time_col])

	if label is not None:
		# Align label to features after dropna
		label_aligned = label.iloc[-len(X):].reset_index(drop=True)
		metrics = train_supervised_model(X, label_aligned, save_path=args.out_sup)
		print("Supervised metrics:", metrics)
	else:
		info = train_unsupervised_model(X, save_path=args.out_unsup)
		print("Unsupervised model:", info)


if __name__ == "__main__":
	main() 