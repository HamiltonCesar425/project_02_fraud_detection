"""
===============================================================================
ETAPA 5.2 – TREINAMENTO DE MODELOS SUPERVISIONADOS
Project_02: Credit Card Fraud Detection
Autor: Comandante
-------------------------------------------------------------------------------
Objetivo:
Treinar modelos clássicos (Logistic Regression, Random Forest)
para detecção de fraudes, usando dataset enriquecido com features de grafo.
===============================================================================
"""

import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------------------------
# 1. Configuração de ambiente
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "transactions_with_graph_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# 2. Função principal
# ---------------------------------------------------------------------------
def train_models(input_csv: str = INPUT_PATH):
    """
    Treina modelos de Machine Learning para detecção de fraudes.
    """

    # === Carregamento seguro ===
    logging.info("Carregando dataset enriquecido para treinamento...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_csv}")

    df = pd.read_csv(input_csv)
    logging.info(f"Dataset carregado — {df.shape[0]} linhas, {df.shape[1]} colunas.")

    # === Verificação de coluna alvo ===
    if "Class" not in df.columns:
        raise KeyError("Coluna alvo 'Class' não encontrada no dataset.")

    # === Separação X e y ===
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # === Divisão treino/teste ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === Padronização ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Modelo 1: Regressão Logística ===
    logging.info("Treinando modelo: Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    # === Modelo 2: Random Forest ===
    logging.info("Treinando modelo: Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)

    # === Avaliação ===
    logging.info("Avaliação dos modelos:")
    for name, model, X_eval in [
        ("Logistic Regression", log_reg, X_test_scaled),
        ("Random Forest", rf, X_test),
    ]:
        preds = model.predict(X_eval)
        logging.info(f"\n--- {name} ---")
        logging.info("\n" + classification_report(y_test, preds, digits=4))
        logging.info("\nMatriz de confusão:\n%s", confusion_matrix(y_test, preds))

    # === Salvamento ===
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(log_reg, os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    logging.info(f"✅ Modelos salvos em: {MODEL_DIR}")

    return log_reg, rf


# ---------------------------------------------------------------------------
# 3. Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        train_models()
    except Exception as e:
        logging.error(f"❌ Erro durante o treinamento: {e}")
        sys.exit(1)


