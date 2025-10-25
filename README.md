# House Price Prediction - MLOps Lab

## Overview
End-to-end ML pipeline for Pakistan house price prediction using DVC, Git, and Flask deployment.

## Repository Structure
```
house-price-prediction/
├── data/                 # Dataset (tracked with DVC)
├── src/                  # Pipeline scripts
│   ├── prepare.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── models/              # Trained models (tracked with DVC)
├── metrics/             # Evaluation metrics
├── templates/           # Flask HTML templates
├── app.py               # Flask application
├── params.yaml          # Configuration parameters
├── dvc.yaml            # DVC pipeline definition
└── dvc.lock            # DVC pipeline lock file
```

## Features Implemented
✅ DVC pipeline for reproducible ML workflow  
✅ Data versioning with DVC  
✅ Model training with RandomForest (R² = 0.79)  
✅ Flask web application for predictions  
✅ GitHub integration for version control  

## Model Performance
- **MAE**: 4,227,201 PKR
- **RMSE**: 17,506,524 PKR
- **R² Score**: 0.79

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/Syed-Farhan19/House-Price-Prediction-Flask-App.git
cd House-Price-Prediction-Flask-App
```

### 2. Install Dependencies
```bash
pip install dvc pandas scikit-learn joblib pyyaml flask numpy
```

### 3. Pull Data and Models
```bash
dvc pull
```

### 4. Run Pipeline (Optional)
```bash
dvc repro
```

### 5. Start Flask App
```bash
python app.py
```

### 6. Access Application
Open browser: `http://localhost:5000`

## Technologies Used
- **Python**: ML and web development
- **DVC**: Data and model versioning
- **Git/GitHub**: Code version control
- **scikit-learn**: Machine learning
- **Flask**: Web deployment

## Dataset
Pakistan House Price Dataset from Kaggle  
- **Samples**: 168,446 houses
- **Features**: 20 columns
- **Target**: House price in PKR

## GitHub Repository
https://github.com/Syed-Farhan19/House-Price-Prediction-Flask-App
