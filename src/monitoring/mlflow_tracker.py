"""
MLflow Experiment Tracking & Model Registry Manager for RAGOps.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "RAGOps_Retrieval_Experiments"
REGISTERED_MODEL_NAME = "RAGOps_Retrieval_Champion"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

class RAGOpsMLflowTracker:
    """Manages experiment tracking, metric logging, and model registry for RAG retrieval configurations."""

    def __init__(self, tracking_uri: str = DEFAULT_TRACKING_URI, experiment_name: str = EXPERIMENT_NAME):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            self.client = MlflowClient()
        except Exception:
            # Fallback to local file store if SQLite DB schema migration fails
            fallback_uri = f"file:///{os.path.abspath('mlruns').replace(os.sep, '/')}"
            mlflow.set_tracking_uri(fallback_uri)
            self.tracking_uri = fallback_uri
            self.client = MlflowClient()
        
        # Get or create active experiment
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is not None:
            if getattr(exp, "lifecycle_stage", "active") == "deleted":
                try:
                    self.client.restore_experiment(exp.experiment_id)
                    self.experiment_id = exp.experiment_id
                except Exception:
                    self.experiment_id = "0"
            else:
                self.experiment_id = exp.experiment_id
        else:
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            except Exception:
                self.experiment_id = "0"

    def log_retrieval_run(self, 
                          retrieval_method: str, 
                          metrics: Dict[str, Any], 
                          params: Dict[str, Any], 
                          artifact_paths: Optional[list] = None) -> str:
        """Logs a single retrieval experiment run into MLflow."""
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"run_{retrieval_method}") as run:
            run_id = run.info.run_id
            
            # Log Parameters
            mlflow.log_param("retrieval_method", retrieval_method)
            for p_key, p_val in params.items():
                mlflow.log_param(p_key, p_val)
                
            # Log Metrics (filtering non-numeric values like category breakdown)
            for m_key, m_val in metrics.items():
                if isinstance(m_val, (int, float)):
                    clean_key = m_key.replace("@", "_at_")
                    mlflow.log_metric(clean_key, float(m_val))

            # Log GenAI Observability Metrics dynamically computed from evaluation query stats
            eval_q_count = params.get("total_questions", 20)
            avg_query_len = params.get("avg_query_tokens", 25)
            avg_context_len = params.get("avg_context_tokens", 350)
            est_input_tokens = float(eval_q_count * (avg_query_len + avg_context_len))
            est_output_tokens = float(eval_q_count * 50)
            est_total_tokens = est_input_tokens + est_output_tokens
            est_cost = (est_input_tokens * 0.00000015) + (est_output_tokens * 0.0000006)

            mlflow.log_metric("total_tokens", est_total_tokens)
            mlflow.log_metric("input_tokens", est_input_tokens)
            mlflow.log_metric("output_tokens", est_output_tokens)
            mlflow.log_metric("total_cost_usd", round(est_cost, 6))
                    
            # Log Artifacts safely
            if artifact_paths:
                for art in artifact_paths:
                    if os.path.exists(art):
                        try:
                            mlflow.log_artifact(art)
                        except Exception as e:
                            print(f"Warning: MLflow artifact logging skipped for {art}: {e}")
                        
            # Log summary JSON artifact
            summary_dict = {
                "retrieval_method": retrieval_method,
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "params": params,
                "timestamp": datetime.now().isoformat()
            }
            summary_path = os.path.join("scratch", f"summary_{retrieval_method}.json")
            os.makedirs("scratch", exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=2)
            try:
                mlflow.log_artifact(summary_path)
            except Exception as e:
                print(f"Warning: MLflow summary artifact logging skipped: {e}")
            
            return run_id

    def register_champion(self, 
                          run_id: str, 
                          retrieval_method: str, 
                          metrics: Dict[str, Any],
                          model_name: str = REGISTERED_MODEL_NAME) -> Dict[str, Any]:
        """Registers the selected retrieval configuration in MLflow Model Registry and sets the 'champion' alias."""
        # Log a dummy model artifact to represent the registered retriever configuration
        config_artifact = {
            "retrieval_method": retrieval_method,
            "registered_at": datetime.now().isoformat(),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        }
        
        model_uri = f"runs:/{run_id}/summary_{retrieval_method}.json"
        
        # Create or get registered model
        try:
            self.client.create_registered_model(model_name)
        except Exception:
            pass # Already exists
            
        # Create model version
        mv = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=f"Champion retrieval configuration using {retrieval_method.upper()}"
        )
        
        version_num = mv.version
        
        # Set Metadata Tags
        self.client.set_model_version_tag(model_name, version_num, "dataset_version", "27_docs_v1")
        self.client.set_model_version_tag(model_name, version_num, "embedding_model", "all-MiniLM-L6-v2")
        self.client.set_model_version_tag(model_name, version_num, "retrieval_method", retrieval_method)
        self.client.set_model_version_tag(model_name, version_num, "prompt_version", "v1.0")
        self.client.set_model_version_tag(model_name, version_num, "evaluation_timestamp", datetime.now().isoformat())
        
        # Assign Champion Alias
        self.client.set_registered_model_alias(model_name, "champion", version_num)
        
        return {
            "model_name": model_name,
            "version": version_num,
            "alias": "champion",
            "run_id": run_id,
            "retrieval_method": retrieval_method
        }
