# visualization.py - Project_02: Credit Card Fraud Detection
"""
Gera gráficos descritivos e comparativos de desempenho dos modelos.
Salva as figuras em 'reports/figures' sem exibição interativa (modo headless).
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import matplotlib
matplotlib.use("Agg")  # evita bloqueio gráfico

# === Configuração de logs ===
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# === Diretórios ===
FIG_DIR = os.path.join("reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# === Funções ===


def plot_distribution(df, column, title):
    plt.figure(figsize=(8, 5))
    df[column].hist(bins=50, color="steelblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequência")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"{column}_distribution.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Gráfico salvo: {out_path}")


def plot_model_performance(metrics_path):
    if not os.path.exists(metrics_path):
        logger.warning("Arquivo de métricas não encontrado, pulando visualização.")
        return

    df = pd.read_csv(metrics_path)
    plt.figure(figsize=(10, 6))
    plt.bar(df["model"], df["roc_auc"], color="darkgreen", alpha=0.7)
    plt.title("Desempenho dos Modelos (ROC-AUC)")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "model_performance.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Gráfico salvo: {out_path}")


def main():
    logger.info("=== Início da Geração de Visualizações ===")

    # Exemplo de uso — pode adaptar conforme outputs dos seus módulos
    try:
        data_path = os.path.join("data", "processed", "train_data.csv")
        df = pd.read_csv(data_path)
        plot_distribution(df, "Amount", "Distribuição de Valores de Transação")
        plot_distribution(df, "Time", "Distribuição Temporal das Transações")
    except Exception as e:
        logger.warning(f"Falha ao gerar gráficos descritivos: {e}")

    try:
        metrics_path = os.path.join("reports", "model_metrics.csv")
        plot_model_performance(metrics_path)
    except Exception as e:
        logger.warning(f"Falha ao gerar gráfico de performance: {e}")

    logger.info("=== Fim da Geração de Visualizações ===")


if __name__ == "__main__":
    main()
