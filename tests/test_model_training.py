"""
Módulo de treinamento de modelos supervisionados
Project_02: Credit Card Fraud Detection
Autor: Comandante

Objetivo:
Treinar modelos clássicos (Logistic Regression, Random Forest)
para detecção de fraudes, usando dataset enriquecido com features de grafo.
"""

import os
import sys
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Configuração de ambiente
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "transactions_with_graph_features.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def load_dataset(input_csv: str) -> pd.DataFrame:
    """Carrega dataset e valida presença da coluna alvo."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_csv}")

    df = pd.read_csv(input_csv)

    if "Class" not in df.columns:
        raise KeyError("Coluna alvo 'Class' não encontrada no dataset.")

    return df


def evaluate_model(model, X_test, y_test, model_name: str) -> None:
    """Avalia modelo e imprime métricas."""
    preds = model.predict(X_test)
    logging.info("--- %s ---", model_name)
    logging.info("\n%s", classification_report(y_test, preds, digits=4))
    logging.info("Matriz de confusão:\n%s", confusion_matrix(y_test, preds))


def save_models(models: dict) -> None:
    """Salva modelos e scaler em disco."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, obj in models.items():
        joblib.dump(obj, os.path.join(MODEL_DIR, f"{name}.pkl"))
    logging.info("Modelos salvos em: %s", MODEL_DIR)


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def train_models(input_csv: str = INPUT_PATH):
    """Treina modelos de Machine Learning para detecção de fraudes."""
    df = load_dataset(input_csv)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_regression_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    logistic_regression_model.fit(X_train_scaled, y_train)

    random_forest_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    )
    random_forest_model.fit(X_train, y_train)

    logging.info("Avaliação dos modelos:")
    evaluate_model(
        logistic_regression_model, X_test_scaled, y_test, "Logistic Regression"
    )
    evaluate_model(random_forest_model, X_test, y_test, "Random Forest")

    save_models(
        {
            "logistic_regression_model": logistic_regression_model,
            "random_forest_model": random_forest_model,
            "scaler": scaler,
        }
    )

    return logistic_regression_model, random_forest_model


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        train_models()
    except Exception as exc:
        logging.error("Erro durante o treinamento: %s", exc)
        sys.exit(1)
