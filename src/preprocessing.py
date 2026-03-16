# src/preprocessing.py - Project_02: Credit Card Fraud Detection
"""
Módulo de pré-processamento dos dados.
Responsável por dividir o dataset, normalizar e preparar variáveis para o treinamento dos modelos.
"""

import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame):
    """
    Realiza o pré-processamento dos dados para modelagem.

    Etapas:
    - Remove valores nulos
    - Separa features (X) e target (y)
    - Padroniza variáveis numéricas
    - Divide dataset em treino e teste (estratificado)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados brutos.

    Retorna
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    """

    logger.info("Iniciando pré-processamento dos dados...")

    # 1. Limpeza básica
    df = df.dropna()
    logger.info(f"Dimensão após remoção de nulos: {df.shape}")

    # 2. Separação entre features e target
    if "Class" not in df.columns:
        raise ValueError("Coluna 'Class' não encontrada no dataset.")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 3. Padronização das variáveis numéricas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 4. Divisão treino/teste estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(
        f"Pré-processamento concluído | Treino: {X_train.shape} | Teste: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test


def save_processed_dataset(df: pd.DataFrame):
    """
    Salva dataset pré-processado em disco.
    """

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "transactions_processed.csv")

    df.to_csv(output_path, index=False)

    logger.info(f"Dataset pré-processado salvo em: {output_path}")


if __name__ == "__main__":

    from src.data_loading import load_data

    dataset_path = os.path.join("data", "raw", "creditcard.csv")

    if not os.path.exists(dataset_path):
        logger.error("Arquivo de dados não encontrado.")
        raise FileNotFoundError(dataset_path)

    logger.info("Carregando dataset bruto...")

    df = load_data(dataset_path)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # opcional: salvar dataset completo processado
    df_clean = df.dropna()

    save_processed_dataset(df_clean)

    logger.info("Pipeline de pré-processamento executado com sucesso.")
