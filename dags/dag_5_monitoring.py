"""
DAG 5: Performance & Latency Monitoring Pipeline
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

def monitor_latency():
    print("Monitoring p95 and p99 query latency metrics...")
    from scripts.run_phase14_monitoring import main as run_phase14
    return run_phase14()

def check_hallucination_drift():
    print("Checking hallucination rate and embedding drift metrics...")
    from scripts.run_phase15_drift import main as run_phase15
    return run_phase15()

if AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        'dag_5_performance_monitoring',
        default_args=default_args,
        description='RAGOps Latency, Throughput & Hallucination Rate Monitoring',
        schedule='@daily',
        catchup=False,
        tags=['monitoring_v1'],
    ) as dag:

        t1 = PythonOperator(
            task_id='check_api_latency',
            python_callable=monitor_latency,
        )

        t2 = PythonOperator(
            task_id='check_hallucination_drift',
            python_callable=check_hallucination_drift,
        )

        t1 >> t2

