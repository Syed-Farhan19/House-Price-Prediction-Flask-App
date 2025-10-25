import pandas as pd
import argparse
import os
import yaml

def load_params():
    with open('params.yaml') as f:
        return yaml.safe_load(f)

def prepare_data(input_file, output_dir):
    """Load and perform initial data cleaning"""
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic cleaning
    df = df.dropna()
    
    # Save cleaned data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cleaned_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()
    
    prepare_data(args.input, args.output_dir)