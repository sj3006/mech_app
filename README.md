# Mechanical Component Failure Predictor

A simple ML app to detect/predict failures from multivariate time-series (vibration, temperature, RPM, etc.) with **AI-powered failure analysis and solution recommendations**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train from a labeled CSV (must contain 'timestamp' and a binary label, e.g., 'failed')
python scripts/train.py data/sample.csv --label_col failed --freq 1S --fs_hint 10

# Or train unsupervised anomaly detector
python scripts/train.py data/sample.csv --freq 1S --fs_hint 10

# Run the UI
streamlit run app/app.py
```

## CSV Format
- Required: `timestamp` column parseable as datetime
- Signals: any numeric columns (e.g., `vibration`, `temp`, `rpm`)
- Optional label: a binary column like `failed` for supervised training

## How it works
- Resamples to fixed rate
- Computes rolling stats (mean/std/min/max/peak-to-peak/RMS)
- Computes coarse spectral bandpower per signal
- Supervised: GradientBoosting on features
- Unsupervised: IsolationForest anomaly score
- **NEW**: AI-powered failure analysis with explanations and solutions

## 🤖 GenAI Failure Analysis

The app now includes an AI-powered failure analyzer that:

- **Analyzes failure patterns** from sensor data and predictions
- **Identifies failure types** (bearing failure, rotor imbalance, misalignment, etc.)
- **Provides detailed explanations** of what's happening mechanically
- **Suggests actionable solutions** to address the identified issues
- **Assesses urgency levels** and affected components
- **Uses rule-based analysis** (no external API required)

### Features:
- **Failure Type Classification**: Bearing failure, rotor imbalance, misalignment, lubrication issues, cavitation
- **Root Cause Analysis**: Identifies potential causes based on sensor patterns
- **Solution Recommendations**: Provides specific, actionable maintenance steps
- **Risk Assessment**: Determines urgency levels (CRITICAL, HIGH, MEDIUM, LOW)
- **Component Identification**: Lists affected mechanical components

### Demo:
```bash
# Run the demo script to see GenAI analysis in action
python examples/genai_demo.py

# Or use the web interface
streamlit run app/app.py
```

### Free GenAI Setup:
The app uses **Ollama** for AI-powered analysis:
- **Ollama**: Completely free, runs locally (recommended)
- **Rule-based analysis**: Always works as fallback, no setup required

See `FREE_GENAI_SETUP.md` for detailed setup instructions.

## Notes
- Tune `--freq` and `--fs_hint` to match your data rate
- Adjust thresholds in `predict_with_model` as needed
- GenAI analysis works with both supervised and unsupervised models 