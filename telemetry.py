import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class RoboticTelemetryAnalyzer:
    def __init__(self, anomaly_threshold=0.7):
        self.anomaly_threshold = anomaly_threshold
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    def load_telemetry_data(self, file_path):
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith('.txt'):
                df = pd.read_csv(file_path, delim_whitespace=True)
            else:
                raise ValueError("Unsupported file format. Use .csv or .txt")
            
            required_columns = [
                'Latitude', 'Longitude', 'Altitude',
                'Speed', 'Acceleration', 'Deceleration',
                'Roll', 'Pitch', 'Yaw'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            return df
        
        except Exception as e:
            print(f"Error loading telemetry data: {e}")
            return None
    
    def calculate_risk_score(self, df):
        features = ['Latitude', 'Longitude', 'Altitude', 
                    'Speed', 'Acceleration', 'Deceleration', 
                    'Roll', 'Pitch', 'Yaw']
        
        scaled_data = self.scaler.fit_transform(df[features])
        anomaly_labels = self.isolation_forest.fit_predict(scaled_data)
        
        deviations = np.abs(scaled_data - np.mean(scaled_data, axis=0))
        anomaly_scores = np.max(deviations, axis=1)
        
        risk_scores = anomaly_scores * 100
        risk_levels = ['Low Risk' if score <= 50 else 'High Risk' for score in risk_scores]
        
        df['RiskScore'] = risk_scores
        df['RiskLevel'] = risk_levels
        
        return df
    
    def analyze_telemetry(self, file_path):
        df = self.load_telemetry_data(file_path)
        
        if df is None:
            return None
        
        analyzed_df = self.calculate_risk_score(df)
        return analyzed_df

if __name__ == "__main__":
    analyzer = RoboticTelemetryAnalyzer()
    results = analyzer.analyze_telemetry('robotic_telemetry_data.csv')
    
    if results is not None:
        """
        print("\nTelemetry Analysis Results:")
        print(results[['RiskScore', 'RiskLevel']].describe())
        """
        print("\nHigh-Risk Entries:")
        print(results[results['RiskLevel'] == 'High Risk'])
