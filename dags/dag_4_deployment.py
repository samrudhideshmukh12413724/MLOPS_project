"""
DAG 4: Automated Model & Retriever Deployment Pipeline
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

def log_mlflow_model():
    print("Registering new Retriever & Synthesizer models in MLflow Model Registry...")
    from scripts.run_phase8_mlflow import run_phase8
    return run_phase8()

def deploy_fastapi():
    print("Verifying containerized FastAPI endpoint with updated index...")
    from scripts.run_phase12_pipeline import main as run_phase12
    return run_phase12()

if AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        'dag_4_model_deployment',
        default_args=default_args,
        description='RAGOps Model Registration & Automated Deployment Pipeline',
        schedule='@weekly',
        catchup=False,
        tags=['deployment_v1'],
    ) as dag:

        t1 = PythonOperator(
            task_id='register_mlflow_model',
            python_callable=log_mlflow_model,
        )

        t2 = PythonOperator(
            task_id='deploy_api_service',
            python_callable=deploy_fastapi,
        )

        t1 >> t2

