# run_pipeline.py - Project_02: Credit Card Fraud Detection
"""
Executa, em sequência, todos os módulos do pipeline de detecção de fraudes.
Gera logs detalhados, cronometra a execução e interrompe em caso de falhas.

Uso:
    python run_pipeline.py
"""

import subprocess
import logging
import datetime
import os
import sys
import time

# === Configuração de logging ===
os.makedirs("logs", exist_ok=True)
log_file = os.path.join(
    "logs", f"pipeline_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)

# === Caminhos base ===
DATASET_PATH = os.path.join("data", "raw", "creditcard.csv")
SRC_DIR = "src"

# === Ordem dos módulos ===
PIPELINE_STEPS = [
    "data_loading.py",
    "preprocessing.py",
    "feature_integration.py",
    "graph_modeling.py",
    "model_training.py",
    "evaluate.py",
    "visualization.py",
    "graph_visualization.py",
    "predict_model.py",
]


# === Função auxiliar para logar de forma segura ===
def log_message(level: str, message: str):
    """Evita erros de encoding ao imprimir mensagens no log."""
    safe_msg = message.encode("utf-8", errors="replace").decode("utf-8")
    if level == "INFO":
        logger.info(safe_msg)
    elif level == "WARNING":
        logger.warning(safe_msg)
    elif level == "ERROR":
        logger.error(safe_msg)


# === Função principal ===
def run_pipeline():
    start_time = datetime.datetime.now()
    logger.info("=== INÍCIO DO PIPELINE DE FRAUDE ===")
    success_steps = 0

    for step in PIPELINE_STEPS:
        script_path = os.path.join(SRC_DIR, step)
        logger.info(f"Iniciando: {step}")

        cmd = [sys.executable, script_path]
        if step == "data_loading.py":
            cmd.append(DATASET_PATH)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,  # 10 min de limite para cada módulo
            )
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Tempo excedido ao executar {step}.")
            break
        except Exception as e:
            logger.exception(f"Erro inesperado ao executar {step}: {e}")
            break

        # === Captura do output ===
        if result.stdout:
            log_message("INFO", result.stdout.strip())
        else:
            log_message("WARNING", f"Sem saída capturada do módulo: {step}")

        if result.stderr:
            log_message("ERROR", result.stderr.strip())

        # === Verificação de retorno ===
        if result.returncode != 0:
            logger.error(f"❌ ERRO em {step} (retcode {result.returncode})")
            logger.error(f"Pipeline interrompido devido a falha no módulo: {step}")
            break
        else:
            logger.info(f"✅ Etapa concluída: {step}")
            success_steps += 1
            time.sleep(1)  # pausa leve entre módulos

    # === Conclusão ===
    end_time = datetime.datetime.now()
    total_minutes = round((end_time - start_time).total_seconds() / 60, 2)

    logger.info("=== FIM DO PIPELINE ===")
    logger.info(f"Etapas concluídas: {success_steps}/{len(PIPELINE_STEPS)}")
    logger.info(f"Tempo total: {total_minutes} minutos")
    logger.info(f"Log salvo em: {log_file}")

    print("\n>> Execução finalizada. Consulte o log completo em:", log_file)


# === Execução direta ===
if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.exception(f"Falha inesperada na execução do pipeline: {e}")
        sys.exit(1)
