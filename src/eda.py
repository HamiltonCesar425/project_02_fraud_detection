"""
EDA (Exploratory Data Analysis) - Project_02: Credit Card Fraud Detection
Script modular, com logging, visualizações (matplotlib + seaborn) e salvamento automático
para `reports/figures/`.

Uso:
    python src/eda.py --input ../data/raw/creditcard.csv --out_dir ../reports/figures/

Funções principais:
 - load_data (reaproveita data_loading.load_data quando disponível)
 - basic_info
 - missing_values_report
 - target_distribution
 - plot_correlation_matrix
 - detect_outliers (IQR)
 - plot_feature_distributions (hist + boxplot)
 - save_figure

"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Estilo dos plots
sns.set(style="whitegrid")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """Tenta usar data_loading.load_data se existir; fallback para pandas.read_csv."""
    try:
        from data_loading import load_data as external_load

        df = external_load(filepath)
        if df is None:
            raise Exception("data_loading.load_data retornou None")
        return df
    except Exception:
        logger.debug("Usando pandas.read_csv diretamente para carregar dados.")

    if not os.path.exists(filepath):
        logger.error(f"Arquivo não encontrado: {filepath}")
        return None

    return pd.read_csv(filepath)


# ----------------- Funções de EDA -----------------


def basic_info(df: pd.DataFrame) -> None:
    logger.info("===== Basic info =====")
    buffer = []
    buffer.append(f"Shape: {df.shape}")
    buffer.append("\nDtypes:")
    buffer.append(str(df.dtypes))
    logger.info("\n".join(buffer))


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isnull().sum()
    miss_percent = (miss / len(df) * 100).round(4)
    report = pd.concat([miss, miss_percent], axis=1)
    report.columns = ["missing_count", "missing_percent"]
    logger.info("===== Missing values (cols with NA) =====")
    logger.info(str(report[report.missing_count > 0]))
    return report


def target_distribution(
    df: pd.DataFrame, target_col: str = "Class", out_dir: Optional[str] = None
) -> None:
    if target_col not in df.columns:
        logger.error(f"Coluna alvo '{target_col}' nao existe no dataset")
        return

    counts = df[target_col].value_counts()
    percent = df[target_col].value_counts(normalize=True) * 100

    logger.info("===== Target distribution =====")
    logger.info(str(counts))
    logger.info(str(percent.round(4)))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title("Distribuição da Classe Alvo (0 = Legítimo, 1 = Fraude)")
    plt.xlabel("Classe")
    plt.ylabel("Contagem")

    if out_dir:
        path = os.path.join(out_dir, "target_distribution.png")
        save_figure(path)
    else:
        plt.show()


def plot_correlation_matrix(df: pd.DataFrame, out_dir: Optional[str] = None) -> None:
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.25)
    plt.title("Matriz de Correlação (Pearson)")

    if out_dir:
        path = os.path.join(out_dir, "correlation_matrix.png")
        save_figure(path)
    else:
        plt.show()


def detect_outliers(df: pd.DataFrame, features: Optional[list] = None) -> pd.DataFrame:
    """Detecta outliers por IQR e retorna um resumo com counts por feature."""
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_summary = {}
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = int(count)

    out_df = pd.DataFrame.from_dict(
        outlier_summary, orient="index", columns=["outlier_count"]
    )
    out_df = out_df.sort_values(by="outlier_count", ascending=False)
    logger.info("===== Outlier summary (IQR) =====")
    logger.info(str(out_df.head(20)))
    return out_df


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Optional[list] = None,
    out_dir: Optional[str] = None,
    max_plots: int = 12,
) -> None:
    """Gera histogramas e boxplots para features numéricas. Salva figuras separadas."""
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    features = [f for f in features if f != "Class"]
    n = min(len(features), max_plots)
    for i, col in enumerate(features[:n]):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histograma: {col}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot: {col}")

        plt.tight_layout()
        if out_dir:
            fname = f"feature_{i+1}_{col}.png"
            save_figure(os.path.join(out_dir, fname))
        else:
            plt.show()


def save_figure(path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Figura salva: {path}")


# ----------------- Orquestração -----------------


def run_eda(
    input_path: str, out_dir: str = "../reports/figures", target_col: str = "Class"
) -> None:
    _ensure_dir(out_dir)
    df = load_data(input_path)
    if df is None:
        logger.error("Falha ao carregar dados — abortando EDA")
        return

    basic_info(df)
    missing_values_report(df)
    df = df.copy()

    # remover coluna Time se existir (irrelevante para o modelo)
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
        logger.info("Coluna 'Time' removida do dataset (EDA).")

    target_distribution(df, target_col, out_dir)
    plot_correlation_matrix(
        df.drop(columns=[target_col]) if target_col in df.columns else df, out_dir
    )
    outliers = detect_outliers(
        df.drop(columns=[target_col]) if target_col in df.columns else df
    )
    plot_feature_distributions(df, out_dir=out_dir)

    # salvar resumo de outliers
    outliers.to_csv(os.path.join(out_dir, "outlier_summary.csv"))
    logger.info(
        f"Resumo de outliers salvo em: {os.path.join(out_dir, 'outlier_summary.csv')}"
    )


# ----------------- CLI -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA - Project_02 Fraud Detection")
    parser.add_argument(
        "--input", required=True, help="Caminho para data/raw/creditcard.csv"
    )
    parser.add_argument(
        "--out_dir",
        default="../reports/figures",
        help="Pasta para salvar figuras e relatórios",
    )
    parser.add_argument("--target", default="Class", help="Nome da coluna target")
    args = parser.parse_args()

    run_eda(args.input, args.out_dir, args.target)
