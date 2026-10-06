"""
DAG 2: Vector Indexing & Embedding Pipeline for RAGOps
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

def generate_embeddings():
    print("Generating dense vector embeddings for document chunks...")
    from src.vectorstore.embedding_engine import EmbeddingEngine
    engine = EmbeddingEngine()
    test_vec = engine.embed_text("Sample embedding generation test")
    print(f"Embedding Engine Initialized: {engine.model_name}, dim: {len(test_vec)}")
    return len(test_vec)

def update_vector_store():
    print("Upserting vector embeddings into ChromaDB database...")
    from scripts.run_phase5_pipeline import run_phase5
    return run_phase5()

if AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        'dag_2_vector_indexing',
        default_args=default_args,
        description='RAGOps Dense Vector Embedding & FAISS Indexing',
        schedule='@daily',
        catchup=False,
        tags=['indexing_v1'],
    ) as dag:

        t1 = PythonOperator(
            task_id='embed_chunks',
            python_callable=generate_embeddings,
        )

        t2 = PythonOperator(
            task_id='upsert_vector_index',
            python_callable=update_vector_store,
        )

        t1 >> t2

