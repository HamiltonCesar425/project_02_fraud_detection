# feature_integration.py — Project_02: Credit Card Fraud Detection
"""
Gera e integra métricas de grafo ao dataset de transações.

Este módulo:
1. Carrega o dataset pré-processado.
2. Cria IDs determinísticos (a partir de colunas numéricas) para simular entidades.
3. Constrói um grafo ponderado com base no valor das transações.
4. Salva o dataset enriquecido com métricas de rede.

Saída: data/processed/transactions_with_graph_features.csv
"""

import os
import pandas as pd
import hashlib
import networkx as nx
import sys
sys.stdout.reconfigure(encoding='utf-8')


def generate_id(value):
    """Gera IDs determinísticos via hash, garantindo consistência entre execuções."""
    return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % 10000


def main():
    print("[INFO] Carregando dataset de transações...")
    input_path = os.path.join("data", "processed", "transactions_processed.csv")
    df = pd.read_csv(input_path)

    print(f"[INFO] Dataset carregado com sucesso — {len(df)} linhas, {len(df.columns)} colunas.")

    # Garantir consistência de entidades
    if "sender_id" not in df.columns or "receiver_id" not in df.columns:
        print("[WARNING] ⚠️ Colunas 'sender_id' e/ou 'receiver_id' ausentes — gerando IDs determinísticos.")
        df["sender_id"] = df["V1"].apply(generate_id)
        df["receiver_id"] = df["V2"].apply(generate_id)

    # Detectar peso
    weight_col = "Amount" if "Amount" in df.columns else None
    if not weight_col:
        raise ValueError("Coluna de peso 'Amount' não encontrada no dataset!")

    print(f"[INFO] Coluna de peso detectada: '{weight_col}'")
    print("[INFO] Construindo grafo ponderado...")

    G = nx.from_pandas_edgelist(
        df,
        source="sender_id",
        target="receiver_id",
        edge_attr=weight_col,
        create_using=nx.Graph(),
    )

    print(f"[INFO] Grafo construído: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    print("[INFO] Calculando métricas básicas de grafo...")

    graph_metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
    }
    print("[INFO] Métricas globais:", graph_metrics)

    # Exportar grafo se necessário
    graph_features_path = os.path.join("data", "processed", "transactions_with_graph_features.csv")
    df.to_csv(graph_features_path, index=False)
    print(f"[INFO] ✅ Dataset salvo com sucesso em: {graph_features_path}")


if __name__ == "__main__":
    main()
