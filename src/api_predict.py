# api_predict.py - Project_02: Credit Card Fraud Detection
"""
API REST para predição de fraudes em tempo real.
Desenvolvida em FastAPI, baseada no modelo treinado (pkl).

Execução local:
    uvicorn src.api_predict:app --reload --port 8000
"""

from src.preprocessing import preprocess_data
import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging

# === CONFIGURAÇÃO BÁSICA ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# === INICIALIZAÇÃO DA API ===
app = FastAPI(
    title="Fraud Detection API",
    description="API de predição de fraudes em transações de cartão de crédito.",
    version="1.0.0",
)

# === CARREGAMENTO DO MODELO ===
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_baseline.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Modelo carregado com sucesso: {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Falha ao carregar modelo: {e}")
    model = None


# === ESQUEMA DE DADOS DE ENTRADA ===
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Valor da transação em moeda local")


# === ROTAS ===

@app.get("/")
def root():
    return {"status": "online", "message": "API de detecção de fraudes ativa."}


@app.post("/predict")
def predict(transactions: List[Transaction]):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    # Converte lista de objetos Transaction em DataFrame
    df = pd.DataFrame([t.model_dump() for t in transactions])
    logger.info(f"📦 Recebidas {len(df)} transações para predição.")

    # Pré-processamento (opcional, se houver transformações)
    try:
        X, _, _, _ = preprocess_data(df)
    except Exception:
        logger.warning("⚠️ Falha ao aplicar preprocessamento. Usando dados brutos.")
        X = df

    # Predição
    try:
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

    # Retorna resultados
    results = []
    for i in range(len(df)):
        results.append({
            "transaction_id": i + 1,
            "fraud_probability": round(float(probs[i]), 6),
            "fraud_prediction": int(preds[i]),
        })

    logger.info(f"✅ Predições concluídas ({len(results)} transações).")
    return {"results": results}
