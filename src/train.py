"""
train.py
========
Script training utama — melatih seluruh model (3 wajib + 8 pembanding)
menggunakan pipeline terpadu dan menyimpan semua hasil ke folder models/ dan reports/.

Cara penggunaan:
    python src/train.py
    python src/train.py --model KNN
    python src/train.py --model all
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

try:
    from src.data_generator import load_dataset, audit_dataset
    from src.preprocessing import run_preprocessing
    from src.ml_core import (get_baseline_models, get_all_models,
                               evaluate_model, print_report, RANDOM_SEED, ORDER_TARGET)
except ImportError:
    from data_generator import load_dataset, audit_dataset
    from preprocessing import run_preprocessing
    from ml_core import (get_baseline_models, get_all_models,
                          evaluate_model, print_report, RANDOM_SEED, ORDER_TARGET)

MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_PATH   = os.path.join(BASE_DIR, "data", "ObesityDataSet_raw_and_data_sinthetic.csv")
CLEAN_PATH  = os.path.join(BASE_DIR, "data", "obesity_cleaned_data.csv")


def setup():
    """Load dan preprocessing data, kembalikan split train/test yang sudah discale."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading dataset...")
    X, y, scaler, oe = run_preprocessing(DATA_PATH, models_dir=MODELS_DIR,
                                          clean_data_path=CLEAN_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    else:
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler disimpan ke: {scaler_path}")
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)

    print(f"Train: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


def train_model(name, model, X_train, y_train, X_test, y_test):
    """Latih satu model, simpan .joblib, dan kembalikan hasil metrik."""
    print(f"\n[Training] {name}...")
    result = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    print(f"  Accuracy={result['Accuracy']:.4f}  F1-macro={result['F1-macro']:.4f}  "
          f"Train={result['Train (ms)']}ms")

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    fpath = os.path.join(MODELS_DIR, fname)
    joblib.dump(result["_model"], fpath)
    print(f"  Model disimpan: {fpath}")

    return {k: v for k, v in result.items() if not k.startswith("_")}


def train_all(target: str = "all"):
    """Latih semua model atau model tertentu saja."""
    X_train, X_test, y_train, y_test, scaler = setup()

    if target == "baseline" or target == "all":
        print("\n" + "="*60)
        print("TRAINING 3 MODEL WAJIB (BASELINE)")
        print("="*60)
        baseline_results = []
        for name, model in get_baseline_models().items():
            res = train_model(name, model, X_train, y_train, X_test, y_test)
            baseline_results.append(res)

        df_base = pd.DataFrame(baseline_results)
        df_base.to_csv(os.path.join(REPORTS_DIR, "baseline_results.csv"), index=False)
        print(f"\nBaseline results disimpan ke reports/baseline_results.csv")

    if target == "all":
        print("\n" + "="*60)
        print("TRAINING SELURUH 11 MODEL")
        print("="*60)
        all_results = []
        cr_dict = {}

        for name, model in get_all_models().items():
            res = train_model(name, model, X_train, y_train, X_test, y_test)
            all_results.append(res)

            full = evaluate_model(name, model, X_train, y_train, X_test, y_test)
            cr_dict[name] = classification_report(
                y_test, full["_y_pred"], target_names=ORDER_TARGET,
                output_dict=True, zero_division=0
            )

        df_all = pd.DataFrame(all_results).sort_values("F1-macro", ascending=False)
        df_all.to_csv(os.path.join(REPORTS_DIR, "all_experiment_results.csv"), index=False)

        with open(os.path.join(REPORTS_DIR, "classification_reports.json"), "w") as f:
            json.dump(cr_dict, f, indent=2)

        print(f"\n{'='*60}")
        print("RINGKASAN HASIL (diurutkan F1-macro)")
        print("="*60)
        print(df_all[["Model","Accuracy","F1-macro","Balanced Acc"]].to_string(index=False))

        best = df_all.iloc[0]["Model"]
        best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        best_fname = best.lower().replace(" ","_").replace("(","").replace(")","") + ".joblib"
        import shutil
        shutil.copy(os.path.join(MODELS_DIR, best_fname), best_model_path)
        print(f"\nModel terbaik : {best} -> disimpan sebagai best_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline obesitas ML")
    parser.add_argument("--model", default="all",
                        choices=["all", "baseline", "KNN", "NaiveBayes", "SVM"],
                        help="Model yang akan dilatih (default: all)")
    args = parser.parse_args()

    if args.model in ["KNN", "NaiveBayes", "SVM"]:
        X_train, X_test, y_train, y_test, _ = setup()
        models = get_baseline_models()
        train_model(args.model, models[args.model], X_train, y_train, X_test, y_test)
    else:
        train_all(args.model)
