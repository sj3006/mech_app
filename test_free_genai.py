#!/usr/bin/env python3
"""
Test script to verify free GenAI options work correctly.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from failure_predictor.genai_analyzer import GenAIFailureAnalyzer

def test_rule_based_analysis():
    """Test rule-based analysis (always works)"""
    print("🧪 Testing rule-based analysis...")
    
    analyzer = GenAIFailureAnalyzer(use_ollama=False)
    
    # Create minimal test data
    import pandas as pd
    import numpy as np
    
    sensor_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'vibration': np.random.normal(2.0, 0.3, 100),
        'temperature': np.random.normal(45, 5, 100)
    })
    
    predictions = pd.DataFrame({
        'prediction': np.random.randint(0, 2, 100),
        'risk_score': np.random.uniform(0, 1, 100)
    })
    
    risk_scores = np.random.uniform(0, 1, 100)
    feature_names = ['vibration', 'temperature']
    
    try:
        analysis = analyzer.analyze_failure(sensor_data, predictions, risk_scores, feature_names)
        print(f"✅ Rule-based analysis works!")
        print(f"   Failure Type: {analysis.failure_type}")
        print(f"   Confidence: {analysis.confidence:.1%}")
        print(f"   Urgency: {analysis.urgency_level}")
        return True
    except Exception as e:
        print(f"❌ Rule-based analysis failed: {e}")
        return False

def test_ollama_availability():
    """Test if Ollama is available"""
    print("\n🧪 Testing Ollama availability...")
    
    analyzer = GenAIFailureAnalyzer(use_ollama=True)
    
    if analyzer._check_ollama_available():
        print("✅ Ollama is available!")
        return True
    else:
        print("⚠️ Ollama not available (install with: brew install ollama)")
        print("   The app will use rule-based analysis instead.")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Free GenAI Options")
    print("=" * 40)
    
    # Test rule-based analysis
    rule_based_works = test_rule_based_analysis()
    
    # Test Ollama availability
    ollama_available = test_ollama_availability()
    
    print("\n📊 Test Results:")
    print(f"Rule-based analysis: {'✅ Works' if rule_based_works else '❌ Failed'}")
    print(f"Ollama: {'✅ Available' if ollama_available else '⚠️ Not installed'}")
    
    if rule_based_works:
        print("\n🎉 Your app will work! Rule-based analysis is always available.")
        if ollama_available:
            print("🎉 Bonus: Ollama is available for enhanced AI analysis!")
        else:
            print("💡 Tip: Install Ollama for better AI analysis (see FREE_GENAI_SETUP.md)")
    else:
        print("\n❌ There's an issue with the basic functionality. Please check your setup.")

if __name__ == "__main__":
    main()
