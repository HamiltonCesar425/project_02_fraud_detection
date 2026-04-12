"""
===============================================================================
ETAPA 5.3 – AVALIAÇÃO DE MODELOS
Project_02: Credit Card Fraud Detection
Autor: Comandante
-------------------------------------------------------------------------------
Objetivo:
Avaliar o desempenho dos modelos treinados (Logistic Regression e Random Forest)
utilizando métricas clássicas de classificação e curva ROC-AUC.
===============================================================================
"""

import os
import sys
import logging
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# 1. Configuração do ambiente
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "transactions_with_graph_features.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_evaluation.csv")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# 2. Função principal
# ---------------------------------------------------------------------------
def evaluate_models():
    """
    Avalia os modelos treinados e salva métricas de desempenho em CSV.
    """

    # === Carregamento dos dados ===
    logging.info("Carregando dataset para avaliação...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset não encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise KeyError("Coluna 'Class' ausente no dataset para avaliação.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # === Carregamento dos modelos ===
    logging.info("Carregando modelos salvos...")
    log_reg = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    # === Padronização para o modelo linear ===
    X_scaled = scaler.transform(X)

    # === Avaliação ===
    results = []

    models = {
        "Logistic Regression": (log_reg, X_scaled),
        "Random Forest": (rf, X),
    }

    for name, (model, X_eval) in models.items():
        logging.info(f"Avaliando modelo: {name}")

        y_pred = model.predict(X_eval)
        y_prob = (
            model.predict_proba(X_eval)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_prob) if y_prob is not None else None,
        }

        results.append(metrics)

        logging.info(
            f"\nRelatório {name}:\n" + classification_report(y, y_pred, digits=4)
        )
        logging.info(f"Matriz de confusão ({name}):\n{confusion_matrix(y, y_pred)}\n")

    # === Consolidação e salvamento ===
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    results_df.to_csv(REPORT_PATH, index=False)
    logging.info(f"✅ Avaliação concluída! Relatório salvo em: {REPORT_PATH}")

    return results_df


# ---------------------------------------------------------------------------
# 3. Execução direta
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        evaluate_models()
        return 0
    except Exception as e:
        logging.error(f"❌ Erro durante a avaliação: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
