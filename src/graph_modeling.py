# graph_modeling.py - versão final otimizada e tolerante a tempo
"""
Modela o grafo ponderado de transações e extrai métricas principais
(degree, betweenness, pagerank) com limitação adaptativa de custo.

Projetado para rodar dentro de pipelines sem travar o sistema.
"""

import pandas as pd
import networkx as nx
import logging
import os
import time
import random
import signal
import sys

# === Configuração de logging ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# === Timeout interno em segundos (para segurança) ===
TIMEOUT_LIMIT = 300  # 5 minutos por módulo


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)


def build_weighted_graph(df):
    """Constrói grafo ponderado com base nas transações."""
    G = nx.Graph()
    for _, row in df.iterrows():
        src = row.get("entity_src", row.get("sender_id", None))
        dst = row.get("entity_dst", row.get("receiver_id", None))
        w = row.get("Amount", row.get("amount", 1))
        if src is None or dst is None:
            continue
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += w
        else:
            G.add_edge(src, dst, weight=w)
    return G


def compute_metrics(G):
    """Calcula métricas com controle adaptativo de performance."""
    n = len(G)
    logger.info(f"[INFO] Grafo com {n} nós e {len(G.edges)} arestas.")
    metrics = {}

    # Degree centrality — rápido
    metrics["degree"] = dict(G.degree(weight="weight"))

    # Betweenness (aproximado)
    k = min(500, max(50, n // 20))  # adapta conforme o tamanho
    logger.info(f"[INFO] Calculando betweenness (amostragem k={k})...")
    metrics["betweenness"] = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)

    # PageRank (limitado em iterações)
    logger.info("[INFO] Calculando PageRank (máx 30 iterações)...")
    metrics["pagerank"] = nx.pagerank(G, max_iter=30, tol=1e-4)

    df_metrics = pd.DataFrame({
        "entity_id": list(G.nodes),
        "degree": [metrics["degree"].get(n, 0) for n in G.nodes],
        "betweenness": [metrics["betweenness"].get(n, 0) for n in G.nodes],
        "pagerank": [metrics["pagerank"].get(n, 0) for n in G.nodes],
    })

    logger.info(f"[INFO] Métricas extraídas: {df_metrics.shape}")
    logger.info("[INFO] Amostra:\n%s", df_metrics.head())
    return df_metrics


def main():
    logger.info("=== Iniciando graph_modeling.py ===")

    df_path = os.path.join("data", "processed", "transactions_with_graph_features.csv")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {df_path}")

    df = pd.read_csv(df_path)
    start = time.time()
    logger.info("[INFO] Construindo grafo ponderado...")

    G = build_weighted_graph(df)
    logger.info(f"[INFO] Grafo construído: {len(G.nodes)} nós, {len(G.edges)} arestas.")

    signal.alarm(TIMEOUT_LIMIT)
    try:
        df_metrics = compute_metrics(G)
        signal.alarm(0)
    except TimeoutException:
        logger.warning(
            "[WARNING] Tempo limite atingido durante cálculo das métricas. Interrompendo suavemente.")
        signal.alarm(0)
        df_metrics = pd.DataFrame(list(G.degree(weight="weight")), columns=["entity_id", "degree"])
        df_metrics["betweenness"] = 0
        df_metrics["pagerank"] = 0

    out_path = os.path.join("data", "processed", "graph_metrics.csv")
    df_metrics.to_csv(out_path, index=False)

    logger.info(f"[INFO] Métricas salvas em: {out_path}")
    logger.info(f"[INFO] Tempo total: {round(time.time() - start, 2)}s")
    logger.info("=== Conclusão de graph_modeling.py ===")


if __name__ == "__main__":
    main()
