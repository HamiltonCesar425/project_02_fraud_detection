# predict_model.py - Project_02: Credit Card Fraud Detection
"""
Script responsável por carregar modelos treinados e realizar predições
em novos dados de transações, para uso em produção ou validação.

Uso:
    python src/predict_model.py --model models/random_forest_baseline.pkl \
                                --input data/new_transactions.csv \
                                --output reports/predictions/random_forest_predictions.csv
"""

import os
import sys
import argparse
import joblib
import logging
import pandas as pd
from src.data_loading import load_data
from src.preprocessing import preprocess_data

# === GARANTE IMPORTAÇÃO DOS MÓDULOS LOCAIS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === CONFIGURAÇÕES BÁSICAS ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# === FUNÇÕES PRINCIPAIS ===
def load_model(model_path: str):
    """Carrega modelo previamente treinado (.pkl)."""
    if not os.path.exists(model_path):
        logger.error(f"❌ Modelo não encontrado: {model_path}")
        sys.exit(1)
    model = joblib.load(model_path)
    logger.info(f"✅ Modelo carregado com sucesso: {model_path}")
    return model


def load_new_data(input_path: str) -> pd.DataFrame:
    """Carrega novos dados de entrada para predição."""
    logger.info(f"📂 Carregando novos dados: {input_path}")
    df = load_data(input_path)

    # Remove coluna de label se existir (evita erro no predict)
    if "Class" in df.columns:
        logger.info("⚙️ Removendo coluna 'Class' dos dados de entrada (não usada em predição).")
        df = df.drop(columns=["Class"])

    return df


def preprocess_for_inference(df: pd.DataFrame):
    """Aplica o mesmo pipeline de pré-processamento usado no treinamento."""
    logger.info("🔧 Aplicando pré-processamento aos dados de entrada...")
    try:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        # Aproveita apenas a parte transformada (X_test) se função dividir dataset
        return X_test if X_test is not None else df
    except Exception as e:
        logger.warning(f"⚠️ Falha ao aplicar preprocessamento completo: {e}. Retornando dados originais.")
        return df


def predict_and_save(model, df: pd.DataFrame, output_path: str):
    """Realiza predição e salva resultados em CSV."""
    logger.info("🚀 Gerando predições...")

    try:
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)
    except Exception as e:
        logger.error(f"❌ Falha ao gerar predições: {e}")
        sys.exit(1)

    results = df.copy()
    results["fraud_probability"] = probs
    results["fraud_prediction"] = preds

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)

    logger.info(f"📈 Predições salvas com sucesso em: {output_path}")
    logger.info(results.head(5).to_string(index=False))


# === FUNÇÃO PRINCIPAL ===
def main(model_path: str, input_path: str, output_path: str):
    """Fluxo completo de predição."""
    model = load_model(model_path)
    new_data = load_new_data(input_path)
    processed_data = preprocess_for_inference(new_data)
    predict_and_save(model, processed_data, output_path)
    logger.info("🏁 Execução de predição concluída com sucesso.")


# === EXECUÇÃO DIRETA VIA TERMINAL ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa predições usando um modelo treinado para detecção de fraude."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(BASE_DIR, "models", "random_forest_baseline.pkl"),
        help="Caminho do modelo (.pkl). Padrão: models/random_forest_baseline.pkl",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho do CSV de novos dados (ex: data/new_transactions.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(BASE_DIR, "reports", "predictions", "predictions_output.csv"),
        help="Caminho para salvar as predições geradas (CSV).",
    )

    args = parser.parse_args()
    main(args.model, args.input, args.output)
