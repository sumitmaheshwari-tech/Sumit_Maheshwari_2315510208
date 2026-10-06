from typing import Dict, Any, Optional

class PromptBuilder:
    """
    Constructs schema-aware, feedback-guided prompts for LLM synthesis.
    """
    def __init__(self, schema_context: str):
        self.schema_context = schema_context

    def build_generation_prompt(self, query: str, feedback_context: Optional[str] = None) -> str:
        feedback_section = ""
        if feedback_context:
            feedback_section = f"""
RELEVANT LESSONS FROM PREVIOUS FEEDBACK (Avoid Past Mistakes):
{feedback_context}
"""

        prompt = f"""You are an elite Business Intelligence & Analytical Database Engineer.
Your task is to convert the user's natural language analytical query into an executable, highly-optimized DuckDB SQL query.

{self.schema_context}
{feedback_section}

ANALYTICAL SQL RULES & GUIDELINES:
1. TARGET VIEWS: Query against the `sales` view (which already pre-calculates `revenue`, `month`, `year`, `quarter`) and `targets` table.
2. REVENUE METRIC: Always use `revenue` (or `quantity * unit_price * (1 - discount)`). Never use unit_price alone.
3. TOP N WITHIN GROUPS: Use window functions like `DENSE_RANK() OVER (PARTITION BY <group> ORDER BY <metric> DESC)` inside a CTE, then filter `WHERE rnk <= N` or `rnk = 1`. Do not use simple LIMIT for grouped top-N.
4. CONTRIBUTION PERCENTAGES: Compute `ROUND(SUM(revenue) * 100.0 / SUM(SUM(revenue)) OVER (), 2) AS contribution_pct`.
5. TARGET COMPARISONS: Join `sales` with `targets` on BOTH `region` and `month` (e.g. `s.region = t.region AND t.month = '2024-02'`).
6. TIME QUERIES (YoY / MoM): Use CTEs or `LAG()` window functions over year/month or compare annual sums.
7. NESTED AGGREGATIONS: Use Common Table Expressions (CTEs, e.g. `WITH ... AS (...)`) for multi-step logic.
8. RETURN FORMAT: Respond ONLY with a valid JSON object matching this exact schema:
{{
  "generated_logic": "<Executable DuckDB SQL query string, terminated by semicolon>",
  "explanation": "<Step-by-step plain English explanation of what was understood and how the result is computed>",
  "assumptions": "<Any domain assumptions made regarding filters, metrics, or edge cases>",
  "self_confidence": <Float between 0.0 and 1.0 indicating model certainty>
}}

USER QUERY:
"{query}"

Generate the JSON response now:"""
        return prompt.strip()

    def build_correction_prompt(self, query: str, failed_sql: str, error_message: str) -> str:
        prompt = f"""You are an expert DuckDB SQL Engineer.
A previously generated SQL query failed execution with a database error. Please correct the query.

USER QUERY:
"{query}"

FAILED SQL:
{failed_sql}

DATABASE ERROR MESSAGE:
{error_message}

{self.schema_context}

CRITICAL INSTRUCTIONS:
- Carefully analyze the error message (e.g. column name mismatch, syntax error, missing join condition, aggregation error).
- Fix the SQL query so it runs successfully in DuckDB and correctly answers the user query.
- Output ONLY valid JSON matching this schema:
{{
  "generated_logic": "<Corrected executable DuckDB SQL query>",
  "explanation": "<Explanation of what caused the error and how it was fixed>",
  "assumptions": "<Any assumptions>",
  "self_confidence": <Float between 0.0 and 1.0>
}}
"""
        return prompt.strip()
