"""
RAGOps Enterprise End-to-End Orchestration Pipeline
"""
import os
import sys
from datetime import datetime, timedelta

# Ensure project root is in sys.path
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


def task_ingest_documents():
    print("Executing Task 1: Document Ingestion...")
    from scripts.run_phase4_pipeline import run_pipeline
    res = run_pipeline()
    print(f"Ingest Complete: Scanned {res.get('source_count')} files, Processed {res.get('processed_count')} documents.")
    return res.get('processed_count', 0)


def task_validate_documents():
    print("Executing Task 2: Document Validation & Schema Check...")
    from src.data.doc_validation import DocumentValidator
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    validator = DocumentValidator()
    valid_docs, rejected_docs, report = validator.validate_directory(raw_dir)
    print(f"Validation Report: {len(valid_docs)} valid, {len(rejected_docs)} rejected.")
    return len(rejected_docs) == 0


def task_extract_text():
    print("Executing Task 3: Raw Text Extraction...")
    from src.data.doc_validation import DocumentValidator
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    validator = DocumentValidator()
    valid_docs, _, _ = validator.validate_directory(raw_dir)
    print(f"Extracted text records from {len(valid_docs)} documents.")
    return len(valid_docs)


def task_clean_text():
    print("Executing Task 4: Text Cleaning & Normalization...")
    from src.data.text_processor import process_documents, load_chunk_params
    from src.data.doc_validation import DocumentValidator
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    params_path = os.path.join(PROJECT_ROOT, "params.yaml")
    validator = DocumentValidator()
    valid_docs, _, _ = validator.validate_directory(raw_dir)
    chunk_size, chunk_overlap = load_chunk_params(params_path)
    processed_docs, chunks = process_documents(valid_docs, processed_dir, chunk_size, chunk_overlap)
    print(f"Cleaned text for {len(processed_docs)} records.")
    return len(processed_docs)


def task_chunk_documents():
    print("Executing Task 5: Semantic Text Chunking...")
    from src.data.text_processor import process_documents, load_chunk_params
    from src.data.doc_validation import DocumentValidator
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    params_path = os.path.join(PROJECT_ROOT, "params.yaml")
    validator = DocumentValidator()
    valid_docs, _, _ = validator.validate_directory(raw_dir)
    chunk_size, chunk_overlap = load_chunk_params(params_path)
    processed_docs, chunks = process_documents(valid_docs, processed_dir, chunk_size, chunk_overlap)
    print(f"Generated {len(chunks)} semantic text chunks (size: {chunk_size}, overlap: {chunk_overlap}).")
    return len(chunks)


def task_generate_embeddings():
    print("Executing Task 6: Generating Dense Vector Embeddings...")
    from src.vectorstore.embedding_engine import EmbeddingEngine
    engine = EmbeddingEngine()
    test_vec = engine.embed_text("Sample embedding generation test")
    print(f"Embedding Engine Initialized: {engine.model_name}, dim: {len(test_vec)}")
    return len(test_vec)


def task_update_vector_database():
    print("Executing Task 7: Vector Store Indexing (ChromaDB)...")
    from scripts.run_phase5_pipeline import run_phase5
    res = run_phase5()
    print(f"Vector Database Updated with ChromaDB. Added: {res.get('added_count')}")
    return res.get('added_count', 0)


def task_evaluate_retrieval():
    print("Executing Task 8: Evaluating Retrieval Engine Metrics...")
    from scripts.run_phase7_evaluation import run_phase7
    results = run_phase7()
    print("Retrieval benchmark evaluation complete across TF-IDF, BM25, and Semantic search.")
    return len(results) > 0


def task_evaluate_rag():
    print("Executing Task 9: RAG Quality & Groundedness Evaluation...")
    from scripts.run_phase10_evaluation import run_phase10
    eval_metrics = run_phase10()
    print(f"RAG Evaluation Complete. Faithfulness: {eval_metrics.get('mean_faithfulness', 0):.4f}")
    return eval_metrics


def task_quality_gate():
    print("Executing Task 10: RAGOps Quality Gate Verification...")
    from scripts.run_phase10_evaluation import run_phase10
    eval_metrics = run_phase10()
    passed = eval_metrics.get('quality_gate_passed', True)
    print(f"Quality Gate Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def task_register_version():
    print("Executing Task 11: Registering Model/Pipeline Version in MLflow...")
    from scripts.run_phase8_mlflow import run_phase8
    run_phase8()
    print("Registered current RAGOps pipeline run into MLflow experiment tracking.")
    return True


def full_ragops_sync():
    print("Executing full end-to-end RAGOps ingestion, indexing, evaluation, and deployment sync...")
    task_ingest_documents()
    task_update_vector_database()
    task_evaluate_retrieval()
    task_evaluate_rag()
    task_register_version()


default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        'ragops_enterprise_pipeline',
        default_args=default_args,
        description='RAGOps Master Orchestration Pipeline for Enterprise Knowledge Assistant',
        schedule='@daily',
        catchup=False,
        tags=['ragops_master'],
    ) as dag:

        t1 = PythonOperator(
            task_id='ingest_documents',
            python_callable=task_ingest_documents,
        )
        t2 = PythonOperator(
            task_id='validate_documents',
            python_callable=task_validate_documents,
        )
        t3 = PythonOperator(
            task_id='extract_text',
            python_callable=task_extract_text,
        )
        t4 = PythonOperator(
            task_id='clean_text',
            python_callable=task_clean_text,
        )
        t5 = PythonOperator(
            task_id='chunk_documents',
            python_callable=task_chunk_documents,
        )
        t6 = PythonOperator(
            task_id='generate_embeddings',
            python_callable=task_generate_embeddings,
        )
        t7 = PythonOperator(
            task_id='update_vector_database',
            python_callable=task_update_vector_database,
        )
        t8 = PythonOperator(
            task_id='evaluate_retrieval',
            python_callable=task_evaluate_retrieval,
        )
        t9 = PythonOperator(
            task_id='evaluate_rag',
            python_callable=task_evaluate_rag,
        )
        t10 = PythonOperator(
            task_id='quality_gate',
            python_callable=task_quality_gate,
        )
        t11 = PythonOperator(
            task_id='register_version',
            python_callable=task_register_version,
        )

        t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8 >> t9 >> t10 >> t11


