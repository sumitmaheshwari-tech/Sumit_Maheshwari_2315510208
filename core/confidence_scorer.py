from typing import Dict, Any, List

class ConfidenceScorer:
    """
    Computes calibrated confidence score (0.0 to 1.0) based on:
    - Execution success and retry count
    - Schema validation integrity
    - Semantic alignment with known metrics & dimensions
    - Model self-assessment
    """
    def calculate_score(
        self,
        execution_success: bool,
        retry_count: int,
        sql_query: str,
        llm_self_confidence: float = 0.9,
        has_ambiguity: bool = False
    ) -> float:
        if not execution_success:
            return 0.10  # Unreliable query if execution failed

        # 1. Execution Stability (up to 0.40)
        if retry_count == 0:
            exec_score = 0.40
        elif retry_count == 1:
            exec_score = 0.30
        else:
            exec_score = 0.20

        # 2. Schema Integrity (up to 0.25)
        # Check that query targets registered views (sales, targets)
        sql_lower = sql_query.lower()
        if "sales" in sql_lower or "targets" in sql_lower:
            schema_score = 0.25
        else:
            schema_score = 0.10

        # 3. Domain Metric & Query Structure (up to 0.20)
        # Bonus for proper aggregation and joins
        metric_score = 0.10
        if any(keyword in sql_lower for keyword in ["sum(", "count(", "avg(", "round("]):
            metric_score += 0.05
        if any(keyword in sql_lower for keyword in ["group by", "over (", "join", "where"]):
            metric_score += 0.05

        # 4. Ambiguity Deduction
        ambiguity_deduction = 0.05 if has_ambiguity else 0.0

        # 5. LLM Confidence Weight (up to 0.15)
        model_score = max(0.0, min(1.0, llm_self_confidence)) * 0.15

        final_score = exec_score + schema_score + metric_score + model_score - ambiguity_deduction
        return round(max(0.0, min(1.0, final_score)), 2)
