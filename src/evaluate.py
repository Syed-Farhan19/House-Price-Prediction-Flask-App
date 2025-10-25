import numpy as np
import joblib
import argparse
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(data_dir, model_dir, metrics_dir):
    # Load data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load model
    model = joblib.load(os.path.join(model_dir, 'house_price_model.pkl'))
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2)
    }
    
    # Save metrics
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'eval.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--metrics_dir", type=str, default="metrics")
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model_dir, args.metrics_dir)