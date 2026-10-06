"""
DAG 3: RAG Evaluation Pipeline (Faithfulness & Answer Relevancy)
"""
import os
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def evaluate_retrieval():
    print("Evaluating MRR, Hit Rate@K, and Precision@K...")
    from scripts.run_phase7_evaluation import run_phase7
    return run_phase7()

def evaluate_generation():
    print("Evaluating Faithfulness (NLI) and Answer Relevancy...")
    from scripts.run_phase10_evaluation import run_phase10
    return run_phase10()

if AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        'dag_3_rag_evaluation',
        default_args=default_args,
        description='RAGOps Retrieval & LLM Generation Quality Evaluation',
        schedule='@daily',
        catchup=False,
        tags=['evaluation_v1'],
    ) as dag:

        t1 = PythonOperator(
            task_id='eval_retrieval_quality',
            python_callable=evaluate_retrieval,
        )

        t2 = PythonOperator(
            task_id='eval_generation_quality',
            python_callable=evaluate_generation,
        )

        t1 >> t2

