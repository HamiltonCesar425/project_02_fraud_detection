# graph_visualization.py - Project_02: Credit Card Fraud Detection
"""
Gera visualizações estruturais do grafo de transações fraudulentas.
Utiliza amostragem controlada para evitar travamentos e backend não interativo.
"""

import random
import pandas as pd
import networkx as nx
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
GRAPH_PATH = os.path.join("data", "processed", "graph_edges.csv")  # Exemplo
os.makedirs(FIG_DIR, exist_ok=True)


# === Funções auxiliares ===
def load_graph(graph_path):
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Arquivo de arestas não encontrado: {graph_path}")

    logger.info("Carregando dados de arestas para reconstrução do grafo...")
    edges_df = pd.read_csv(graph_path)
    if len(edges_df) == 0:
        raise ValueError("Arquivo de arestas vazio!")

    G = nx.from_pandas_edgelist(
        edges_df, source="source", target="target", edge_attr="weight", create_using=nx.Graph()
    )
    logger.info(f"Grafo carregado: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
    return G


def sample_graph(G, max_nodes=500):
    """Seleciona amostra representativa do grafo para visualização."""
    if G.number_of_nodes() <= max_nodes:
        return G.copy()

    sampled_nodes = random.sample(list(G.nodes()), max_nodes)
    subgraph = G.subgraph(sampled_nodes).copy()
    logger.info(
        f"Amostra gerada: {subgraph.number_of_nodes()} nós, {subgraph.number_of_edges()} arestas.")
    return subgraph


def plot_graph_structure(G, title, filename):
    """Renderiza e salva visualização básica do grafo."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=20)
    nx.draw(
        G,
        pos,
        node_size=30,
        node_color="steelblue",
        edge_color="lightgray",
        alpha=0.8,
        with_labels=False,
    )
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Visualização de grafo salva: {out_path}")


def plot_centrality_distribution(G, filename):
    """Gera gráfico de distribuição de centralidades."""
    centrality = nx.degree_centrality(G)
    values = list(centrality.values())
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, color="darkorange", edgecolor="black")
    plt.title("Distribuição de Centralidade de Grau (Amostra)")
    plt.xlabel("Centralidade de Grau")
    plt.ylabel("Frequência")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Gráfico de centralidade salvo: {out_path}")


def main():
    logger.info("=== Início da Visualização de Grafos ===")

    try:
        G = load_graph(GRAPH_PATH)
    except Exception as e:
        logger.error(f"Erro ao carregar grafo: {e}")
        return

    try:
        G_sample = sample_graph(G, max_nodes=500)
        plot_graph_structure(G_sample, "Amostra da Estrutura do Grafo",
                             "graph_sample_structure.png")
        plot_centrality_distribution(G_sample, "graph_centrality_distribution.png")
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {e}")
        return

    logger.info("=== Fim da Visualização de Grafos ===")


if __name__ == "__main__":
    main()
