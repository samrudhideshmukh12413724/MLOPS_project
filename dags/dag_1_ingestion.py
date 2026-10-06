"""
DAG 1: Document Ingestion Pipeline for RAGOps
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

def ingest_documents():
    print("Ingesting enterprise documents into staging...")
    from scripts.run_phase4_pipeline import run_pipeline
    return run_pipeline()

def parse_metadata():
    print("Extracting document metadata and chunking text...")
    from src.data.text_processor import process_documents, load_chunk_params
    from src.data.doc_validation import DocumentValidator
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    params_path = os.path.join(PROJECT_ROOT, "params.yaml")
    validator = DocumentValidator()
    valid_docs, _, _ = validator.validate_directory(raw_dir)
    chunk_size, chunk_overlap = load_chunk_params(params_path)
    processed_docs, chunks = process_documents(valid_docs, processed_dir, chunk_size, chunk_overlap)
    print(f"Extracted metadata & chunked {len(chunks)} text items.")
    return len(chunks)

if AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        'dag_1_document_ingestion',
        default_args=default_args,
        description='RAGOps Document Ingestion & Chunking Pipeline',
        schedule='@daily',
        catchup=False,
        tags=['ingestion_v1'],
    ) as dag:

        t1 = PythonOperator(
            task_id='ingest_docs',
            python_callable=ingest_documents,
        )

        t2 = PythonOperator(
            task_id='parse_chunk_docs',
            python_callable=parse_metadata,
        )

        t1 >> t2

