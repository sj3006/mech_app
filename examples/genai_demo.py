#!/usr/bin/env python3
"""
Demo script showing the GenAI failure analysis feature.

This script demonstrates how to use the GenAIFailureAnalyzer to analyze
mechanical failure patterns and get AI-powered explanations and solutions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from failure_predictor.genai_analyzer import GenAIFailureAnalyzer, FailureAnalysis


def create_sample_data():
    """Create sample sensor data with failure patterns"""
    
    # Create time series data
    timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Simulate normal operation for first 800 points
    normal_vibration = np.random.normal(2.0, 0.3, 800)
    normal_temp = np.random.normal(45, 5, 800)
    normal_rpm = np.random.normal(1800, 50, 800)
    
    # Simulate bearing failure starting at point 800
    failure_vibration = np.random.normal(4.5, 0.8, 200)  # Higher vibration
    failure_temp = np.random.normal(65, 8, 200)  # Higher temperature
    failure_rpm = np.random.normal(1750, 60, 200)  # Slightly lower RPM due to increased friction
    
    # Combine normal and failure data
    vibration = np.concatenate([normal_vibration, failure_vibration])
    temperature = np.concatenate([normal_temp, failure_temp])
    rpm = np.concatenate([normal_rpm, failure_rpm])
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'vibration': vibration,
        'temperature': temperature,
        'rpm': rpm
    })
    
    return df


def create_sample_predictions(sensor_data):
    """Create sample predictions and risk scores"""
    
    # Simulate risk scores that increase over time
    risk_scores = np.linspace(0.1, 0.9, len(sensor_data))
    
    # Add some noise
    risk_scores += np.random.normal(0, 0.05, len(risk_scores))
    risk_scores = np.clip(risk_scores, 0, 1)
    
    # Create predictions (1 for failure, 0 for normal)
    predictions = (risk_scores > 0.7).astype(int)
    
    # Create features DataFrame
    features = sensor_data.copy()
    features['risk_score'] = risk_scores
    features['prediction'] = predictions
    
    return features, risk_scores


def main():
    """Main demo function"""
    
    print("Failure Analysis Demo")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample sensor data...")
    sensor_data = create_sample_data()
    predictions, risk_scores = create_sample_predictions(sensor_data)
    
    print(f"Data shape: {sensor_data.shape}")
    print(f"Time range: {sensor_data['timestamp'].min()} to {sensor_data['timestamp'].max()}")
    print(f"Risk scores range: {risk_scores.min():.3f} to {risk_scores.max():.3f}")
    print(f"Failure predictions: {predictions.sum()}")
    print()
    
    # Initialize analyzer
    print("Initializing GenAI analyzer...")
    print("Options:")
    print("1. Ollama (local, free) - Recommended")
    print("2. Rule-based analysis (always works)")
    
    # Try Ollama first, fallback to rule-based
    analyzer = GenAIFailureAnalyzer(use_ollama=True, ollama_model="llama3.2:3b")
    
    # Perform analysis
    print("Analyzing failure patterns...")
    analysis = analyzer.analyze_failure(
        sensor_data=sensor_data,
        predictions=predictions,
        risk_scores=risk_scores,
        feature_names=['vibration', 'temperature', 'rpm']
    )
    
    # Display results
    print("\nFAILURE ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Failure Type: {analysis.failure_type}")
    print(f"Confidence: {analysis.confidence:.1%}")
    print(f"Urgency Level: {analysis.urgency_level}")
    print()
    
    print("EXPLANATION:")
    print(analysis.explanation)
    print()
    
    print("ROOT CAUSES:")
    for i, cause in enumerate(analysis.root_causes, 1):
        print(f"{i}. {cause}")
    print()
    
    print("RECOMMENDED SOLUTIONS:")
    for i, solution in enumerate(analysis.solutions, 1):
        print(f"{i}. {solution}")
    print()
    
    print("AFFECTED COMPONENTS:")
    for component in analysis.affected_components:
        print(f"• {component}")
    print()
    
    print("Demo completed successfully!")
    print("\nTo see this in action with the web app, run:")
    print("streamlit run app/app.py")


if __name__ == "__main__":
    main()
