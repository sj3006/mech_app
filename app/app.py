import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from failure_predictor.data import read_timeseries_csv, infer_schema, validate_dataframe, resample_timeseries
from failure_predictor.features import build_feature_table
from failure_predictor.model import load_model, predict_with_model, SUPERVISED_MODEL_PATH, UNSUPERVISED_MODEL_PATH
from failure_predictor.genai_analyzer import GenAIFailureAnalyzer, FailureAnalysis

st.set_page_config(page_title="Mechanical Component Failure Predictor", layout="wide")

st.title("Mechanical Component Failure Predictor")
st.write("Upload a time-series CSV (columns: timestamp,vibration,temp,rpm,...) to get failure risk.")

uploaded = st.file_uploader("CSV file", type=["csv"])
model_type = st.selectbox("Prediction mode", ["Unsupervised (Anomaly)", "Supervised (Requires trained model)"])

#since many mechanical engineering data is calculated at different time frequencies, here we're asking the user for specfic time so that that the data is normalised.
freq = st.text_input("Resample frequency (pandas rule)", value="1S")
fs_hint = st.number_input("Sampling frequency hint (Hz)", value=1.0, step=0.5)

model_path = Path(SUPERVISED_MODEL_PATH if model_type.startswith("Supervised") else UNSUPERVISED_MODEL_PATH)

if uploaded is not None:
	with st.spinner("Reading CSV..."):
		df = pd.read_csv(uploaded)
		if "timestamp" not in df.columns:
			st.error("CSV must have a 'timestamp' column.")
			st.stop()
		df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
		df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
		time_col, value_cols = infer_schema(df)
		validate_dataframe(df, time_col, value_cols)

	st.subheader("Preview")
	st.dataframe(df.head(20))

	with st.spinner("Resampling and building features..."):
		df_res = resample_timeseries(df, time_col, value_cols, rule=freq)
		features = build_feature_table(df_res, time_col, value_cols, fs_hint_hz=fs_hint)
		X = features.drop(columns=[time_col])

	st.subheader("Signals")
	for col in value_cols[:3]:
		fig = px.line(df_res, x=time_col, y=col, title=f"{col} over time")
		st.plotly_chart(fig, use_container_width=True)

	if not model_path.exists():
		st.warning(f"Model not found at {model_path}. Use training script to create one.")
	else:
		with st.spinner("Loading model and predicting..."):
			bundle = load_model(str(model_path))
			preds, scores = predict_with_model(bundle, X, task="supervised" if model_type.startswith("Supervised") else "unsupervised")
			features["risk_score"] = scores
			features["prediction"] = preds
			st.subheader("Predictions")
			st.dataframe(features[[time_col, "risk_score", "prediction"]].tail(100))
			fig2 = px.line(features, x=time_col, y="risk_score", title="Risk score over time")
			st.plotly_chart(fig2, use_container_width=True)
			
			# GenAI Failure Analysis Section
			st.subheader("🤖 AI-Powered Failure Analysis")
			
			with st.spinner("Analyzing failure patterns with AI..."):
				analyzer = GenAIFailureAnalyzer()
				analysis = analyzer.analyze_failure(
					sensor_data=df_res,
					predictions=features,
					risk_scores=scores,
					feature_names=bundle.get("feature_names", list(X.columns))
				)
			
			# Display analysis results
			col1, col2 = st.columns(2)
			
			with col1:
				st.markdown("### 🔍 Failure Analysis")
				st.markdown(f"**Failure Type:** {analysis.failure_type}")
				st.markdown(f"**Confidence:** {analysis.confidence:.1%}")
				st.markdown(f"**Urgency Level:** {analysis.urgency_level}")
				
				st.markdown("**Explanation:**")
				st.info(analysis.explanation)
				
				st.markdown("**Root Causes:**")
				for i, cause in enumerate(analysis.root_causes, 1):
					st.markdown(f"{i}. {cause}")
			
			with col2:
				st.markdown("### 🛠️ Recommended Solutions")
				for i, solution in enumerate(analysis.solutions, 1):
					st.markdown(f"{i}. {solution}")
				
				st.markdown("**Affected Components:**")
				for component in analysis.affected_components:
					st.markdown(f"• {component}")
			
			# Add expandable section for detailed analysis
			with st.expander("📊 Detailed Analysis Metrics"):
				st.markdown("**Risk Score Statistics:**")
				st.write(f"- Maximum Risk Score: {scores.max():.3f}")
				st.write(f"- Average Risk Score: {scores.mean():.3f}")
				st.write(f"- Risk Above Threshold (0.7): {(scores > 0.7).sum()} instances")
				
				st.markdown("**Prediction Summary:**")
				st.write(f"- Total Predictions: {len(preds)}")
				st.write(f"- Failure Predictions: {preds.sum()}")
				st.write(f"- Failure Rate: {preds.mean():.1%}")
else:
	st.info("Upload a CSV to begin.") 