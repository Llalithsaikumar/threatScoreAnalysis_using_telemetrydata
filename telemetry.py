import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    def visualize_telemetry(self, df):
        # Select 5 random features to plot
        features = ['Latitude', 'Longitude', 'Altitude', 'Speed', 'Acceleration']
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        for i, feature in enumerate(features):
            row = i // 3
            col = i % 3
            
            # Plot low and high risk data points
            low_risk_mask = (df['RiskLevel'] == 'Low Risk')
            high_risk_mask = (df['RiskLevel'] == 'High Risk')
            
            axes[row, col].plot(df.loc[low_risk_mask, feature], color='blue', label='Low Risk')
            axes[row, col].plot(df.loc[high_risk_mask, feature], color='red', label='High Risk')
            
            axes[row, col].set_title(feature)
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel(feature)
            axes[row, col].legend()
        
        plt.tight_layout()
        return fig
    
    def analyze_telemetry(self, file_path):
        df = self.load_telemetry_data(file_path)
        
        if df is None:
            return None, None
        
        analyzed_df = self.calculate_risk_score(df)
        telemetry_plot = self.visualize_telemetry(analyzed_df)
        
        return analyzed_df, telemetry_plot

if __name__ == "__main__":
    analyzer = RoboticTelemetryAnalyzer()
    results, plot = analyzer.analyze_telemetry('threatScoreAnalysis_using_telemetrydata/robotic_telemetry_data.csv')
    
    if results is not None:
        print("\nHigh-Risk Entries:")
        print(results[results['RiskLevel'] == 'High Risk'])
        plt.show()