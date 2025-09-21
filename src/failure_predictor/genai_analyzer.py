import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import subprocess


@dataclass
class FailureAnalysis:
    """Data class to hold failure analysis results"""
    failure_type: str
    confidence: float
    explanation: str
    root_causes: List[str]
    solutions: List[str]
    urgency_level: str
    affected_components: List[str]


class GenAIFailureAnalyzer:
    """
    GenAI-powered failure analyzer that explains mechanical failures and suggests solutions.
    
    This class analyzes sensor data patterns and failure predictions to provide
    human-readable explanations of mechanical failures and actionable solutions.
    """
    
    def __init__(self, use_ollama: bool = True, ollama_model: str = "llama3.2:3b"):
        """
        Initialize the GenAI analyzer.
        
        Args:
            use_ollama: Whether to use local Ollama LLM (recommended, free)
            ollama_model: Ollama model to use (e.g., "llama3.2:3b", "mistral:7b")
        """
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        
    def analyze_failure(
        self, 
        sensor_data: pd.DataFrame, 
        predictions: pd.DataFrame, 
        risk_scores: np.ndarray,
        feature_names: List[str]
    ) -> FailureAnalysis:
        """
        Analyze mechanical failure based on sensor data and predictions.
        
        Args:
            sensor_data: Original sensor data (timestamp, vibration, temp, rpm, etc.)
            predictions: DataFrame with predictions and risk scores
            risk_scores: Array of risk scores
            feature_names: List of feature names used in the model
            
        Returns:
            FailureAnalysis object with detailed failure information
        """
        # Extract key metrics from the data
        analysis_data = self._extract_analysis_metrics(sensor_data, predictions, risk_scores, feature_names)
        
        if self.use_ollama:
            return self._ollama_failure_analysis(analysis_data)
        else:
            return self._local_failure_analysis(analysis_data)
    
    def _extract_analysis_metrics(
        self, 
        sensor_data: pd.DataFrame, 
        predictions: pd.DataFrame, 
        risk_scores: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """Extract key metrics for failure analysis"""
        
        # Get sensor columns (exclude timestamp)
        sensor_cols = [col for col in sensor_data.columns if col != 'timestamp']
        
        # Calculate recent statistics
        recent_data = sensor_data.tail(100)  # Last 100 data points
        
        metrics = {
            'sensor_columns': sensor_cols,
            'recent_stats': {},
            'risk_trend': self._calculate_risk_trend(risk_scores),
            'anomaly_patterns': self._detect_anomaly_patterns(sensor_data, sensor_cols),
            'feature_importance': self._analyze_feature_importance(feature_names, risk_scores),
            'prediction_summary': {
                'total_predictions': len(predictions),
                'failure_predictions': int(predictions['prediction'].sum()),
                'max_risk_score': float(risk_scores.max()),
                'avg_risk_score': float(risk_scores.mean()),
                'risk_above_threshold': int((risk_scores > 0.7).sum())
            }
        }
        
        # Calculate statistics for each sensor
        for col in sensor_cols:
            if col in recent_data.columns:
                metrics['recent_stats'][col] = {
                    'mean': float(recent_data[col].mean()),
                    'std': float(recent_data[col].std()),
                    'max': float(recent_data[col].max()),
                    'min': float(recent_data[col].min()),
                    'trend': self._calculate_trend(recent_data[col])
                }
        
        return metrics
    
    def _calculate_risk_trend(self, risk_scores: np.ndarray) -> str:
        """Calculate the trend of risk scores over time"""
        if len(risk_scores) < 10:
            return "insufficient_data"
        
        recent_scores = risk_scores[-10:]
        early_scores = risk_scores[:10]
        
        recent_avg = np.mean(recent_scores)
        early_avg = np.mean(early_scores)
        
        if recent_avg > early_avg * 1.2:
            return "increasing"
        elif recent_avg < early_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_anomaly_patterns(self, sensor_data: pd.DataFrame, sensor_cols: List[str]) -> Dict:
        """Detect common anomaly patterns in sensor data"""
        patterns = {}
        
        for col in sensor_cols:
            if col in sensor_data.columns:
                data = sensor_data[col].dropna()
                if len(data) > 0:
                    patterns[col] = {
                        'has_spikes': self._detect_spikes(data),
                        'has_drifts': self._detect_drifts(data),
                        'has_oscillations': self._detect_oscillations(data),
                        'variance_change': self._detect_variance_change(data)
                    }
        
        return patterns
    
    def _detect_spikes(self, data: pd.Series) -> bool:
        """Detect if data has significant spikes"""
        if len(data) < 10:
            return False
        z_scores = np.abs((data - data.mean()) / data.std())
        return (z_scores > 3).any()
    
    def _detect_drifts(self, data: pd.Series) -> bool:
        """Detect if data has significant drift"""
        if len(data) < 20:
            return False
        # Simple linear trend detection
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        return abs(coeffs[0]) > data.std() * 0.1
    
    def _detect_oscillations(self, data: pd.Series) -> bool:
        """Detect if data has oscillatory patterns"""
        if len(data) < 20:
            return False
        # Simple oscillation detection using autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        return len(autocorr) > 1 and autocorr[1] < -0.3
    
    def _detect_variance_change(self, data: pd.Series) -> bool:
        """Detect significant changes in variance"""
        if len(data) < 20:
            return False
        mid = len(data) // 2
        early_var = data[:mid].var()
        late_var = data[mid:].var()
        return abs(late_var - early_var) / early_var > 0.5
    
    def _calculate_trend(self, data: pd.Series) -> str:
        """Calculate trend direction of a time series"""
        if len(data) < 5:
            return "insufficient_data"
        
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        slope = coeffs[0]
        
        if slope > data.std() * 0.1:
            return "increasing"
        elif slope < -data.std() * 0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_feature_importance(self, feature_names: List[str], risk_scores: np.ndarray) -> Dict:
        """Analyze which features are most important for high risk scores"""
        # This is a simplified analysis - in practice, you'd use feature importance from the model
        high_risk_indices = np.where(risk_scores > np.percentile(risk_scores, 90))[0]
        
        # Group features by type
        vibration_features = [f for f in feature_names if 'vibration' in f.lower() or 'vib' in f.lower()]
        temp_features = [f for f in feature_names if 'temp' in f.lower() or 'temperature' in f.lower()]
        rpm_features = [f for f in feature_names if 'rpm' in f.lower()]
        
        return {
            'vibration_features': vibration_features,
            'temperature_features': temp_features,
            'rpm_features': rpm_features,
            'high_risk_count': len(high_risk_indices)
        }
    
    def _local_failure_analysis(self, analysis_data: Dict) -> FailureAnalysis:
        """Perform failure analysis using local rule-based approach"""
        
        # Determine failure type based on patterns
        failure_type = self._classify_failure_type(analysis_data)
        
        # Generate explanation
        explanation = self._generate_explanation(analysis_data, failure_type)
        
        # Identify root causes
        root_causes = self._identify_root_causes(analysis_data, failure_type)
        
        # Generate solutions
        solutions = self._generate_solutions(failure_type, root_causes)
        
        # Determine urgency
        urgency_level = self._determine_urgency(analysis_data)
        
        # Identify affected components
        affected_components = self._identify_affected_components(failure_type, analysis_data)
        
        # Calculate confidence based on data quality and pattern strength
        confidence = self._calculate_confidence(analysis_data)
        
        return FailureAnalysis(
            failure_type=failure_type,
            confidence=confidence,
            explanation=explanation,
            root_causes=root_causes,
            solutions=solutions,
            urgency_level=urgency_level,
            affected_components=affected_components
        )
    
    def _classify_failure_type(self, analysis_data: Dict) -> str:
        """Classify the type of mechanical failure based on patterns"""
        
        patterns = analysis_data['anomaly_patterns']
        recent_stats = analysis_data['recent_stats']
        risk_trend = analysis_data['risk_trend']
        
        # Check for bearing failure (high vibration, temperature increase)
        if 'vibration' in recent_stats and 'temperature' in recent_stats:
            vib_high = recent_stats['vibration']['mean'] > recent_stats['vibration']['std'] * 2
            temp_high = recent_stats['temperature']['mean'] > recent_stats['temperature']['std'] * 1.5
            
            if vib_high and temp_high:
                return "Bearing Failure"
        
        # Check for imbalance (vibration spikes, RPM correlation)
        if 'vibration' in patterns and patterns['vibration'].get('has_spikes', False):
            return "Rotor Imbalance"
        
        # Check for misalignment (oscillatory patterns)
        if any(patterns.get(col, {}).get('has_oscillations', False) for col in patterns):
            return "Misalignment"
        
        # Check for lubrication issues (temperature increase, vibration increase)
        if 'temperature' in recent_stats and recent_stats['temperature']['trend'] == 'increasing':
            if 'vibration' in recent_stats and recent_stats['vibration']['trend'] == 'increasing':
                return "Lubrication Issues"
        
        # Check for cavitation (pressure/vibration patterns)
        if 'pressure' in patterns and patterns['pressure'].get('has_spikes', False):
            return "Cavitation"
        
        # Default based on risk trend
        if risk_trend == 'increasing':
            return "General Mechanical Degradation"
        else:
            return "Potential Component Wear"
    
    def _generate_explanation(self, analysis_data: Dict, failure_type: str) -> str:
        """Generate human-readable explanation of the failure"""
        
        prediction_summary = analysis_data['prediction_summary']
        risk_trend = analysis_data['risk_trend']
        
        base_explanation = f"Based on the sensor data analysis, the system has detected a {failure_type.lower()} condition. "
        
        if prediction_summary['failure_predictions'] > 0:
            base_explanation += f"The model has identified {prediction_summary['failure_predictions']} instances of potential failure. "
        
        if risk_trend == 'increasing':
            base_explanation += "The risk level is showing an increasing trend, indicating progressive deterioration. "
        elif risk_trend == 'stable':
            base_explanation += "The risk level has remained relatively stable, suggesting consistent wear patterns. "
        
        # Add specific details based on failure type
        if failure_type == "Bearing Failure":
            base_explanation += "This is typically characterized by elevated vibration levels and increased operating temperature, often caused by insufficient lubrication, contamination, or excessive loading."
        elif failure_type == "Rotor Imbalance":
            base_explanation += "This condition is indicated by periodic vibration spikes that correlate with rotational speed, usually caused by uneven mass distribution or accumulation of debris."
        elif failure_type == "Misalignment":
            base_explanation += "This is evidenced by oscillatory patterns in sensor data, typically resulting from improper installation, thermal expansion, or foundation settling."
        elif failure_type == "Lubrication Issues":
            base_explanation += "This condition shows both temperature and vibration increases, indicating insufficient or degraded lubricant quality."
        
        return base_explanation
    
    def _identify_root_causes(self, analysis_data: Dict, failure_type: str) -> List[str]:
        """Identify potential root causes of the failure"""
        
        root_causes = []
        patterns = analysis_data['anomaly_patterns']
        recent_stats = analysis_data['recent_stats']
        
        if failure_type == "Bearing Failure":
            if 'temperature' in recent_stats and recent_stats['temperature']['trend'] == 'increasing':
                root_causes.append("Insufficient or degraded lubrication")
            if 'vibration' in patterns and patterns['vibration'].get('has_spikes', False):
                root_causes.append("Contamination in bearing housing")
            root_causes.append("Excessive loading or overloading")
            root_causes.append("Bearing fatigue due to extended operation")
        
        elif failure_type == "Rotor Imbalance":
            root_causes.append("Accumulation of debris or deposits on rotor")
            root_causes.append("Uneven wear of rotating components")
            root_causes.append("Improper balancing during maintenance")
            root_causes.append("Material loss due to corrosion or erosion")
        
        elif failure_type == "Misalignment":
            root_causes.append("Improper installation or assembly")
            root_causes.append("Thermal expansion causing shaft movement")
            root_causes.append("Foundation settling or shifting")
            root_causes.append("Wear in coupling or mounting components")
        
        elif failure_type == "Lubrication Issues":
            root_causes.append("Insufficient lubricant quantity")
            root_causes.append("Lubricant contamination or degradation")
            root_causes.append("Incorrect lubricant type or viscosity")
            root_causes.append("Lubrication system malfunction")
        
        else:
            root_causes.append("General wear and tear from normal operation")
            root_causes.append("Insufficient maintenance intervals")
            root_causes.append("Operating conditions outside design parameters")
        
        return root_causes
    
    def _generate_solutions(self, failure_type: str, root_causes: List[str]) -> List[str]:
        """Generate actionable solutions for the identified failure"""
        
        solutions = []
        
        if failure_type == "Bearing Failure":
            solutions.extend([
                "Immediately check and replenish bearing lubrication",
                "Inspect bearing housing for contamination and clean if necessary",
                "Verify bearing load is within specifications",
                "Schedule bearing replacement if excessive wear is detected",
                "Implement more frequent lubrication schedule"
            ])
        
        elif failure_type == "Rotor Imbalance":
            solutions.extend([
                "Stop equipment and clean rotor surfaces",
                "Perform dynamic balancing of rotating components",
                "Inspect for missing or loose components",
                "Check for uneven wear and replace if necessary",
                "Implement regular cleaning and inspection schedule"
            ])
        
        elif failure_type == "Misalignment":
            solutions.extend([
                "Perform laser alignment of rotating components",
                "Check foundation and mounting points for stability",
                "Verify thermal expansion allowances are adequate",
                "Inspect coupling condition and replace if worn",
                "Schedule regular alignment checks"
            ])
        
        elif failure_type == "Lubrication Issues":
            solutions.extend([
                "Drain and replace lubricant with correct type and viscosity",
                "Clean lubrication system and replace filters",
                "Verify lubricant quantity meets specifications",
                "Check for lubricant contamination sources",
                "Implement oil analysis program for early detection"
            ])
        
        else:
            solutions.extend([
                "Schedule immediate inspection of affected components",
                "Review maintenance procedures and intervals",
                "Consider reducing operating load or speed temporarily",
                "Implement condition monitoring program",
                "Plan for component replacement or overhaul"
            ])
        
        # Add general solutions
        solutions.extend([
            "Document all findings and actions taken",
            "Update maintenance records and schedules",
            "Consider implementing predictive maintenance program",
            "Train operators on early warning signs"
        ])
        
        return solutions
    
    def _determine_urgency(self, analysis_data: Dict) -> str:
        """Determine urgency level based on risk scores and trends"""
        
        prediction_summary = analysis_data['prediction_summary']
        risk_trend = analysis_data['risk_trend']
        max_risk = prediction_summary['max_risk_score']
        
        if max_risk > 0.9 or prediction_summary['risk_above_threshold'] > prediction_summary['total_predictions'] * 0.5:
            return "CRITICAL"
        elif max_risk > 0.7 or risk_trend == 'increasing':
            return "HIGH"
        elif max_risk > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _identify_affected_components(self, failure_type: str, analysis_data: Dict) -> List[str]:
        """Identify which components are likely affected"""
        
        components = []
        
        if failure_type == "Bearing Failure":
            components.extend(["Bearings", "Bearing housings", "Shaft journals", "Lubrication system"])
        
        elif failure_type == "Rotor Imbalance":
            components.extend(["Rotor", "Impeller", "Fan blades", "Coupling", "Shaft"])
        
        elif failure_type == "Misalignment":
            components.extend(["Shaft alignment", "Coupling", "Mounting points", "Foundation"])
        
        elif failure_type == "Lubrication Issues":
            components.extend(["Lubrication system", "Bearings", "Gears", "Seals"])
        
        else:
            components.extend(["Primary rotating components", "Supporting structures", "Control systems"])
        
        return components
    
    def _calculate_confidence(self, analysis_data: Dict) -> float:
        """Calculate confidence level for the analysis"""
        
        prediction_summary = analysis_data['prediction_summary']
        risk_trend = analysis_data['risk_trend']
        
        # Base confidence on data quality and pattern strength
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on clear patterns
        if prediction_summary['failure_predictions'] > 0:
            confidence += 0.2
        
        if risk_trend in ['increasing', 'decreasing']:
            confidence += 0.1
        
        if prediction_summary['max_risk_score'] > 0.8:
            confidence += 0.2
        
        # Cap at 0.95
        return min(confidence, 0.95)
    
    def _ollama_failure_analysis(self, analysis_data: Dict) -> FailureAnalysis:
        """Perform failure analysis using local Ollama LLM"""
        try:
            # Check if Ollama is available
            if not self._check_ollama_available():
                print("⚠️ Ollama not available, falling back to rule-based analysis")
                return self._local_failure_analysis(analysis_data)
            
            # Create prompt for Ollama
            prompt = self._create_ollama_prompt(analysis_data)
            
            # Call Ollama
            response = self._call_ollama(prompt)
            
            # Parse response
            return self._parse_ollama_response(response, analysis_data)
            
        except Exception as e:
            print(f"⚠️ Ollama analysis failed: {e}, falling back to rule-based analysis")
            return self._local_failure_analysis(analysis_data)
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _create_ollama_prompt(self, analysis_data: Dict) -> str:
        """Create a detailed prompt for Ollama analysis"""
        
        prediction_summary = analysis_data['prediction_summary']
        recent_stats = analysis_data['recent_stats']
        risk_trend = analysis_data['risk_trend']
        patterns = analysis_data['anomaly_patterns']
        
        prompt = f"""You are a mechanical engineering expert analyzing equipment failure data. Based on the following sensor data and patterns, provide a detailed failure analysis.

SENSOR DATA SUMMARY:
- Risk Trend: {risk_trend}
- Max Risk Score: {prediction_summary['max_risk_score']:.3f}
- Average Risk Score: {prediction_summary['avg_risk_score']:.3f}
- Failure Predictions: {prediction_summary['failure_predictions']} out of {prediction_summary['total_predictions']}

RECENT SENSOR STATISTICS:
"""
        
        for sensor, stats in recent_stats.items():
            prompt += f"- {sensor}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, trend={stats['trend']}\n"
        
        prompt += f"""
ANOMALY PATTERNS DETECTED:
"""
        for sensor, pattern in patterns.items():
            prompt += f"- {sensor}: spikes={pattern.get('has_spikes', False)}, drifts={pattern.get('has_drifts', False)}, oscillations={pattern.get('has_oscillations', False)}\n"
        
        prompt += """
Please analyze this data and provide:

1. FAILURE_TYPE: The most likely type of mechanical failure (e.g., "Bearing Failure", "Rotor Imbalance", "Misalignment", "Lubrication Issues", "Cavitation", "General Mechanical Degradation")

2. CONFIDENCE: A confidence score between 0.0 and 1.0

3. EXPLANATION: A detailed explanation of what's happening mechanically (2-3 sentences)

4. ROOT_CAUSES: List 3-5 potential root causes (one per line, starting with "- ")

5. SOLUTIONS: List 5-7 actionable solutions (one per line, starting with "- ")

6. URGENCY: The urgency level (CRITICAL, HIGH, MEDIUM, or LOW)

7. AFFECTED_COMPONENTS: List 3-5 affected components (one per line, starting with "- ")

Format your response exactly like this:
FAILURE_TYPE: [type]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [detailed explanation]
ROOT_CAUSES:
- [cause 1]
- [cause 2]
- [cause 3]
SOLUTIONS:
- [solution 1]
- [solution 2]
- [solution 3]
URGENCY: [level]
AFFECTED_COMPONENTS:
- [component 1]
- [component 2]
- [component 3]
"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama with the given prompt"""
        try:
            # Use ollama run command
            cmd = ['ollama', 'run', self.ollama_model, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(f"Ollama command failed: {result.stderr}")
            
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise Exception("Ollama request timed out")
        except Exception as e:
            raise Exception(f"Ollama call failed: {e}")
    
    def _parse_ollama_response(self, response: str, analysis_data: Dict) -> FailureAnalysis:
        """Parse Ollama response into FailureAnalysis object"""
        try:
            lines = response.split('\n')
            
            # Initialize with defaults
            failure_type = "General Mechanical Degradation"
            confidence = 0.7
            explanation = "AI analysis completed"
            root_causes = ["Analysis in progress"]
            solutions = ["Review equipment status"]
            urgency_level = "MEDIUM"
            affected_components = ["Primary components"]
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("FAILURE_TYPE:"):
                    failure_type = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
                    except ValueError:
                        confidence = 0.7
                elif line.startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[1].strip()
                elif line.startswith("ROOT_CAUSES:"):
                    current_section = "root_causes"
                    root_causes = []
                elif line.startswith("SOLUTIONS:"):
                    current_section = "solutions"
                    solutions = []
                elif line.startswith("URGENCY:"):
                    urgency_level = line.split(":", 1)[1].strip().upper()
                    current_section = None
                elif line.startswith("AFFECTED_COMPONENTS:"):
                    current_section = "affected_components"
                    affected_components = []
                elif current_section == "root_causes" and line.startswith("- "):
                    root_causes.append(line[2:].strip())
                elif current_section == "solutions" and line.startswith("- "):
                    solutions.append(line[2:].strip())
                elif current_section == "affected_components" and line.startswith("- "):
                    affected_components.append(line[2:].strip())
            
            # Fallback if parsing failed
            if not root_causes or root_causes == ["Analysis in progress"]:
                root_causes = self._identify_root_causes(analysis_data, failure_type)
            if not solutions or solutions == ["Review equipment status"]:
                solutions = self._generate_solutions(failure_type, root_causes)
            if not affected_components or affected_components == ["Primary components"]:
                affected_components = self._identify_affected_components(failure_type, analysis_data)
            
            return FailureAnalysis(
                failure_type=failure_type,
                confidence=confidence,
                explanation=explanation,
                root_causes=root_causes,
                solutions=solutions,
                urgency_level=urgency_level,
                affected_components=affected_components
            )
            
        except Exception as e:
            print(f"⚠️ Failed to parse Ollama response: {e}, using fallback analysis")
            return self._local_failure_analysis(analysis_data)
    
