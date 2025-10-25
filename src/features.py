import pandas as pd
import numpy as np
import argparse
import os
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_params():
    with open('params.yaml') as f:
        return yaml.safe_load(f)

def engineer_features(input_file, output_dir):
    params = load_params()
    
    df = pd.read_csv(input_file)
    print(f"Engineering features for {len(df)} samples")
    
    # Separate target
    target_col = params['features']['target_column']
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    
    # Handle categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split data
    test_size = params['features']['test_size']
    random_state = params['features']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train.values)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test.values)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save feature names and encoders
    joblib.dump(X.columns.tolist(), os.path.join(output_dir, 'feature_names.pkl'))
    joblib.dump(label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()
    
    engineer_features(args.input, args.output_dir)