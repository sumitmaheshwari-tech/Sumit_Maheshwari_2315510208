from typing import Tuple, Any, Optional, Dict
from core.db_executor import DBExecutor
from core.prompt_builder import PromptBuilder
from core.llm_client import LLMClient

class SelfCorrectionEngine:
    """
    Executes generated SQL queries within DuckDB and automatically reflects/repairs
    queries if syntax or database errors arise.
    """
    def __init__(self, db_executor: DBExecutor, prompt_builder: PromptBuilder, llm_client: LLMClient):
        self.executor = db_executor
        self.prompt_builder = prompt_builder
        self.llm = llm_client

    def execute_with_self_healing(
        self,
        query: str,
        initial_sql: str,
        initial_explanation: str,
        initial_confidence: float = 0.9,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        current_sql = initial_sql
        current_explanation = initial_explanation
        current_confidence = initial_confidence
        retries_used = 0
        last_error = None

        for attempt in range(max_retries + 1):
            success, result, error = self.executor.execute_query(current_sql)
            if success:
                return {
                    "sql": current_sql,
                    "success": True,
                    "result": result,
                    "explanation": current_explanation,
                    "retries": retries_used,
                    "self_confidence": current_confidence,
                    "error": None
                }

            last_error = error
            retries_used += 1

            if attempt < max_retries:
                # Initiate self-correction loop
                correction_prompt = self.prompt_builder.build_correction_prompt(
                    query=query,
                    failed_sql=current_sql,
                    error_message=str(error)
                )
                try:
                    repaired = self.llm.generate(correction_prompt)
                    current_sql = repaired.get("generated_logic", current_sql)
                    current_explanation = repaired.get("explanation", current_explanation)
                    current_confidence = float(repaired.get("self_confidence", 0.8))
                except Exception:
                    break

        return {
            "sql": current_sql,
            "success": False,
            "result": None,
            "explanation": f"Failed after {retries_used} attempts. Error: {last_error}",
            "retries": retries_used,
            "self_confidence": 0.1,
            "error": last_error
        }
