# data_loading.py - Project_02: Credit Card Fraud Detection
"""
Responsável por carregar o dataset de fraude em cartões de crédito,
garantindo integridade, validação e logs apropriados.

Uso:
    from src.data_loading import load_data
    df = load_data("data/raw/creditcard.csv")

Execução direta (automática):
    python src/data_loading.py
"""
import os
import sys
import logging
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === Configuração de logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega o dataset CSV e executa validações básicas.

    Parâmetros
    ----------
    file_path : str
        Caminho do arquivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame contendo os dados carregados.

    Levanta
    -------
    FileNotFoundError: se o arquivo não for encontrado.
    ValueError: se o arquivo estiver vazio ou sem colunas esperadas.
    """
    logger.info(f"📂 Verificando caminho do dataset: {file_path}")

    # Verifica existência do arquivo
    if not os.path.exists(file_path):
        logger.error(f"❌ Arquivo não encontrado: {file_path}")
        raise FileNotFoundError(f"O arquivo especificado não existe: {file_path}")

    # Carrega CSV com detecção automática de delimitador
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.exception(f"⚠️ Falha ao ler o CSV: {e}")
        raise

    # Valida DataFrame
    if df.empty:
        raise ValueError("O dataset está vazio.")

    # Verificação mínima de colunas esperadas
    expected_cols = {"Time", "Amount", "Class"}
    if not expected_cols.issubset(df.columns):
        logger.warning(
            f"⚠️ Colunas esperadas ausentes: {expected_cols - set(df.columns)}"
        )

    logger.info(
        f"✅ Dataset carregado com sucesso — {df.shape[0]} linhas, {df.shape[1]} colunas."
    )

    # Remove duplicatas, se existirem
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(
            f"⚠️ {duplicates} registros duplicados encontrados — serão removidos."
        )
        df = df.drop_duplicates().reset_index(drop=True)

    logger.info("📊 Amostra dos dados:")
    logger.info(df.head(3).to_string(index=False))

    return df


def main(argv: list[str] | None = None) -> int:
    """Executa o carregamento via CLI e retorna código de saída."""
    # Caminho padrão para o pipeline automatizado
    DEFAULT_PATH = os.path.join("data", "raw", "creditcard.csv")

    # Se o usuário passar argumento, usa-o; senão, usa o padrão
    argv = sys.argv[1:] if argv is None else argv
    file_path = argv[0] if argv else DEFAULT_PATH

    if not os.path.exists(file_path):
        logger.error(f"❌ ERRO: arquivo '{file_path}' não encontrado.")
        return 1

    try:
        load_data(file_path)
        logger.info("✅ Teste de carregamento concluído com sucesso.")
        return 0
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {e}")
        return 1


# === Execução direta (modo independente ou pelo pipeline) ===
if __name__ == "__main__":
    sys.exit(main())
