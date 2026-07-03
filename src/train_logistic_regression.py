# train_logistic_regression.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
from preprocessing import run_preprocessing

# Constants
RANDOM_SEED = 42
MODELS_DIR = "models"
DATA_PATH = "data/ObesityDataSet_raw_and_data_sinthetic.csv"

def train_logistic_regression(X_train, y_train, X_test, y_test, C=1.0, max_iter=2000, solver='lbfgs'):
    """
    Train Logistic Regression model for multi-class classification
    """
    print(f"  Hyperparameter : C={C}, max_iter={max_iter}, solver={solver}")
    
    # Untuk multi-class classification, scikit-learn akan otomatis menggunakan 'multinomial'
    # jika solver mendukung (lbfgs, sag, saga) dan multi_class='auto'
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=RANDOM_SEED
        # Parameter multi_class tidak perlu ditentukan, default 'auto' sudah cukup
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  Training selesai")
    print(f"  Accuracy test : {accuracy:.4f}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def tune_hyperparameters(X_train, y_train, X_test, y_test):
    """
    Simple hyperparameter tuning for Logistic Regression
    """
    print("\n  Hyperparameter tuning...")
    
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    solvers = ['lbfgs', 'saga']  # kedua solver support multi-class
    
    best_accuracy = 0
    best_params = {}
    best_model = None
    
    for C in C_values:
        for solver in solvers:
            try:
                model = LogisticRegression(
                    C=C,
                    max_iter=2000,
                    solver=solver,
                    random_state=RANDOM_SEED
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"    C={C}, solver={solver:5s} → accuracy={accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'C': C, 'solver': solver}
                    best_model = model
                    
            except Exception as e:
                print(f"    C={C}, solver={solver:5s} → ERROR: {str(e)[:50]}")
    
    print(f"\n  Best accuracy: {best_accuracy:.4f} with params: {best_params}")
    
    return best_model, best_params, best_accuracy

def main():
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Preprocessing
    print("=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    X, y, scaler, oe = run_preprocessing(
        DATA_PATH,
        models_dir=MODELS_DIR,
        clean_data_path="data/obesity_cleaned_data.csv"
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Scale features
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    print(f"\n  Split data:")
    print(f"    Train: {X_train_sc.shape[0]} samples")
    print(f"    Test : {X_test_sc.shape[0]} samples")
    
    # Train model with default parameters
    print("\n" + "=" * 60)
    print("TRAINING: Logistic Regression")
    print("=" * 60)
    
    # Opsi 1: Training dengan default parameters
    result = train_logistic_regression(X_train_sc, y_train, X_test_sc, y_test)
    
    # Opsi 2: Dengan hyperparameter tuning (uncomment jika ingin)
    # best_model, best_params, best_acc = tune_hyperparameters(X_train_sc, y_train, X_test_sc, y_test)
    # result = {
    #     'model': best_model,
    #     'accuracy': best_acc,
    #     'predictions': best_model.predict(X_test_sc),
    #     'probabilities': best_model.predict_proba(X_test_sc)
    # }
    
    # Detailed classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, result['predictions']))
    
    # Confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_test, result['predictions'])
    print(cm)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    joblib.dump(result['model'], model_path)
    print(f"\n✓ Model logistic regression disimpan ke: {model_path}")
    
    # Save results
    results = {
        'accuracy': result['accuracy'],
        'model_params': {
            'C': result['model'].C,
            'solver': result['model'].solver,
            'max_iter': result['model'].max_iter,
            'multi_class': result['model'].multi_class if hasattr(result['model'], 'multi_class') else 'auto'
        }
    }
    
    results_path = os.path.join(MODELS_DIR, "logistic_regression_results.joblib")
    joblib.dump(results, results_path)
    print(f"✓ Results disimpan ke: {results_path}")
    
    print("\n" + "=" * 60)
    print(f"FINAL TEST ACCURACY: {result['accuracy']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()