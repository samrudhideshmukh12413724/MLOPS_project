"""
Data Pipeline DAG for Automated Data Extraction & Ingestion
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

def extract_data():
    print("Extracting raw document corpus...")
    from src.data.doc_validation import DocumentValidator
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    validator = DocumentValidator()
    valid_docs, _, report = validator.validate_directory(raw_dir)
    print(f"Extracted {len(valid_docs)} valid documents.")
    return len(valid_docs)

def process_data():
    print("Processing and cleaning text data...")
    from scripts.run_phase4_pipeline import run_pipeline
    return run_pipeline()

if AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        'data_pipeline_dag',
        default_args=default_args,
        description='Automated Data Ingestion and Preprocessing Pipeline',
        schedule='@daily',
        catchup=False,
        tags=['data_pipe'],
    ) as dag:

        t1 = PythonOperator(
            task_id='extract_raw_data',
            python_callable=extract_data,
        )

        t2 = PythonOperator(
            task_id='preprocess_data',
            python_callable=process_data,
        )

        t1 >> t2

