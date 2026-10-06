"""
MLOps End-to-End Pipeline DAG
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
    'owner': 'mlops_student',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def train_model():
    print("Indexing vector retrieval & reranking models...")
    from scripts.run_phase5_pipeline import run_phase5
    return run_phase5()

def evaluate_model():
    print("Evaluating trained model metrics and logging to MLflow...")
    from scripts.run_phase7_evaluation import run_phase7
    from scripts.run_phase8_mlflow import run_phase8
    eval_res = run_phase7()
    run_phase8()
    return eval_res

if AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        'mlops_pipeline_dag',
        default_args=default_args,
        description='MLOps End-to-End Training & Evaluation Pipeline',
        schedule='@weekly',
        catchup=False,
        tags=['mlops_pipe'],
    ) as dag:

        t1 = PythonOperator(
            task_id='train_retriever_model',
            python_callable=train_model,
        )

        t2 = PythonOperator(
            task_id='evaluate_model_performance',
            python_callable=evaluate_model,
        )

        t1 >> t2

