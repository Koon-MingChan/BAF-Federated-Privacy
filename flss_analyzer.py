import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif

def analyze_baf():
    print("Loading Base.csv for analysis...")
    df = pd.read_csv('data/Base.csv')
    
    # Separate features and target
    X = df.drop(columns=['fraud_bool'])
    y = df['fraud_bool']
    
    # Handle categorical columns for MI calculation
    X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
    
    print("Calculating Feature-Level Sensitivity Scores (FLSS)...")
    results = []
    
    for col in X_numeric.columns:
        # 1. Shannon Entropy (Privacy Risk)
        prob_dist = X_numeric[col].value_counts(normalize=True)
        ent = entropy(prob_dist)
        
        # 2. Mutual Information (Utility/Importance)
        # We use a smaller sample (100k) for MI to save time/memory
        mi = mutual_info_classif(X_numeric[[col]].sample(100000, random_state=42), 
                                 y.sample(100000, random_state=42))[0]
        
        results.append({
            'Feature': col,
            'Entropy_Risk': ent,
            'Utility_Score': mi
        })
    
    analysis_df = pd.DataFrame(results).sort_values(by='Entropy_Risk', ascending=False)
    analysis_df.to_csv('feature_sensitivity_report.csv', index=False)
    print("\nTop 5 Most Sensitive Features (High Entropy):")
    print(analysis_df.head())

if __name__ == "__main__":
    analyze_baf()