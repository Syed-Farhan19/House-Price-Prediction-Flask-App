import numpy as np
import joblib
import argparse
import os
import yaml
from sklearn.ensemble import RandomForestRegressor

def load_params():
    with open('params.yaml') as f:
        return yaml.safe_load(f)

def train_model(data_dir, model_dir):
    params = load_params()['train']
    
    # Load data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=params['random_state'],
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'house_price_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()
    
    train_model(args.data_dir, args.model_dir)