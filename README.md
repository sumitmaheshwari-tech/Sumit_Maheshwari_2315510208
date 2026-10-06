# Intelligent Analytics Query Engine
### Technical Assessment Solution for Office AI Solution

An end-to-end, production-grade Natural Language to OLAP analytical query engine built with **DuckDB**, **Generative AI (Gemini & OpenAI)**, **In-Memory Analytical Sandboxing**, and a **Calibrated Multi-Factor Confidence Scorer**.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Component Breakdown](#3-component-breakdown)
4. [Why DuckDB SQL vs Pandas (Key Tradeoffs)](#4-why-duckdb-sql-vs-pandas-key-tradeoffs)
5. [Complex Analytical Patterns & Edge Cases](#5-complex-analytical-patterns--edge-cases)
6. [Multi-Factor Confidence Scoring Methodology](#6-multi-factor-confidence-scoring-methodology)
7. [In-Context Feedback Loop](#7-in-context-feedback-loop)
8. [Installation & Quickstart](#8-installation--quickstart)
9. [Sample Query Outputs](#9-sample-query-outputs)
10. [Future Improvements](#10-future-improvements)

---

## 1. Executive Summary & Problem Statement

The goal is to build an analytical engine that:
1. **Understands Natural Language** analytical requests and grounds them in business definitions.
2. **Generates Executable Logic** supporting aggregations, groupings, window functions, and multi-table joins.
3. **Safely Executes Logic** against data files (`sales_data.csv` and `targets.csv`) in memory.
4. **Calculates Confidence Scores (0.0 to 1.0)** reflecting execution validity and semantic clarity.
5. **Provides Explanations** detailing what was understood and how results were derived.
6. **Incorporates Feedback Loops** from previous runs to avoid repeating past failure modes.

---

## 2. High-Level Architecture

The system is implemented as a **Compound AI System** combining deterministic database engineering with Generative AI reasoning:

```
+-----------------------------------------------------------------------------------+
|                               USER NATURAL LANGUAGE QUERY                         |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                              1. SEMANTIC & CONTEXT LAYER                          |
|   - Ingests data_dictionary.json (metrics, dimensions, synonyms, time mappings)   |
|   - Ingests feedback_log.csv (retrieves few-shot exemplars & past mistakes)       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                              2. GENAI SYNTHESIS ENGINE                            |
|   - Dual-Provider Client: Google Gemini 2.5 Flash & OpenAI GPT-4o-mini            |
|   - Prompt Builder binds schema, metric rules, and few-shot feedback              |
|   - Generates executable DuckDB SQL + Step-by-Step Rationale                      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                      3. IN-MEMORY DUCKDB EXECUTION SANDBOX                        |
|   - Auto-registers virtual views for sales_data and targets                       |
|   - Pre-computes revenue formula: quantity * unit_price * (1 - discount)          |
|   - Dry-run validation & instant in-memory execution                              |
+---------------------+-------------------+-----------------------------------------+
                      |                   |
            (Runtime Error)           (Execution Success)
                      |                   |
                      v                   v
+---------------------+----+     +--------+-----------------------------------------+
| 4. SELF-CORRECTION LOOP  |     |         5. MULTI-FACTOR CONFIDENCE SCORER        |
| - Feeds DB error to LLM  |     | - Execution stability (+0.40)                    |
| - Reprompts & self-heals |     | - Schema column integrity (+0.25)                |
| - Max 2 retry attempts   |     | - Query complexity & joins (+0.20)               |
+--------------------------+     | - LLM self-confidence & ambiguity penalty (+0.15)|
                                 +--------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                                6. OUTPUT CONTRACT (JSON)                          |
|   { "query", "generated_logic", "result", "confidence_score", "explanation" }     |
+-----------------------------------------------------------------------------------+
```

---

## 3. Component Breakdown

| Module | File | Responsibility |
| :--- | :--- | :--- |
| **Semantic Layer** | `core/semantic_layer.py` | Parses `data_dictionary.json`, maps business terms (e.g. *sales* $\to$ *revenue*), and extracts few-shot feedback from `feedback_log.csv`. |
| **In-Memory OLAP** | `core/db_executor.py` | Embeds DuckDB to execute SQL directly over CSVs in-memory without server setup or disk writes. |
| **Prompt Builder** | `core/prompt_builder.py` | Assembles context-rich prompts enforcing DuckDB SQL dialect, analytical rules, and JSON output contracts. |
| **LLM Client** | `core/llm_client.py` | Unified client supporting Gemini and OpenAI with automatic failover and offline fallback. |
| **Self-Correction Engine** | `core/self_corrector.py` | Detects syntax or schema binder errors and guides the LLM to fix queries iteratively. |
| **Confidence Scorer** | `core/confidence_scorer.py` | Computes an objective 0.0 to 1.0 confidence score combining deterministic execution indicators and semantic clarity. |
| **CLI Runner** | `run_engine.py` | Orchestrates batch processing of queries and writes `output.json`. |

---

## 4. Why DuckDB SQL vs Pandas (Key Tradeoffs)

| Criteria | DuckDB SQL (Chosen Approach) | Pandas / Python `eval()` |
| :--- | :--- | :--- |
| **Security & Sandboxing** | **High:** Read-only SQL queries in an isolated in-memory DB cannot execute arbitrary OS commands. | **Severe Vulnerability:** Executing LLM-generated Python code using `eval()` or `exec()` risks code injection. |
| **Analytical Expressiveness**| Native support for Window Functions (`PARTITION BY`), CTEs (`WITH`), and joins. | Complex multi-step index resets, chained groupby operations, and temporary columns prone to bugs. |
| **LLM Accuracy** | LLMs are trained on billions of lines of SQL, yielding higher first-pass accuracy. | Multi-line Pandas syntax has higher hallucination rates for edge cases. |
| **Speed & Resource Footprint**| Columnar vector execution engine; processes millions of rows in milliseconds. | Row-by-row Python interpreter overhead for complex transformations. |

---

## 5. Complex Analytical Patterns & Edge Cases

The engine is specifically engineered to handle complex requirements:

### A. Top N within Groups
* **Query:** *"Top product in each region"*
* **Pattern:** Uses DuckDB window functions partitioned by group:
  ```sql
  WITH ranked_products AS (
      SELECT region, product_name, SUM(revenue) AS total_revenue,
             DENSE_RANK() OVER (PARTITION BY region ORDER BY SUM(revenue) DESC) AS rnk
      FROM sales
      GROUP BY region, product_name
  )
  SELECT region, product_name, total_revenue
  FROM ranked_products
  WHERE rnk = 1;
  ```

### B. Multi-Table Target Comparisons
* **Query:** *"Which region missed its target in Feb?"*
* **Pattern:** Joins `sales` with `targets` matching on both `region` and `month`:
  ```sql
  WITH feb_actuals AS (
      SELECT region, SUM(revenue) AS actual_revenue
      FROM sales
      WHERE month = '2024-02'
      GROUP BY region
  )
  SELECT t.region, COALESCE(a.actual_revenue, 0) AS actual_revenue, t.target_revenue
  FROM targets t
  LEFT JOIN feb_actuals a ON t.region = a.region
  WHERE t.month = '2024-02' AND COALESCE(a.actual_revenue, 0) < t.target_revenue;
  ```

### C. Contribution Percentages
* **Query:** *"Sales contribution % by category"*
* **Pattern:** Evaluates ratio against total sales using window sums:
  ```sql
  SELECT product_category,
         ROUND(SUM(revenue), 2) AS category_revenue,
         ROUND(SUM(revenue) * 100.0 / SUM(SUM(revenue)) OVER (), 2) AS contribution_percentage
  FROM sales
  GROUP BY product_category;
  ```

### D. Nested Logic
* **Query:** *"Revenue of top 3 customers per region"*
* **Pattern:** Two-tier CTE: first ranks customers within regions, then aggregates top 3 customer revenue per region.

---

## 6. Multi-Factor Confidence Scoring Methodology

Rather than relying purely on LLM self-reported confidence (which is often overconfident), the engine calculates a **calibrated composite score**:

$$\text{Confidence Score} = S_{\text{execution}} + S_{\text{schema}} + S_{\text{metric}} + S_{\text{model}} - P_{\text{ambiguity}}$$

1. **Execution Stability ($S_{\text{execution}}$, max 0.40):**
   * $0.40$: Query executed cleanly on 1st attempt.
   * $0.30$: Query succeeded after 1 self-correction pass.
   * $0.10$: Query failed all execution attempts.
2. **Schema & View Integrity ($S_{\text{schema}}$, max 0.25):**
   * Verifies table references match registered views (`sales`, `targets`).
3. **Query Structure & Aggregation ($S_{\text{metric}}$, max 0.20):**
   * Awards points for explicit aggregations (`SUM`, `COUNT`, `AVG`, `ROUND`) and structural operators (`GROUP BY`, `OVER`, `JOIN`).
4. **Model Certainty ($S_{\text{model}}$, max 0.15):**
   * Scaled from the model's reported certainty.
5. **Ambiguity Penalty ($P_{\text{ambiguity}}$, -0.05):**
   * Deducted if the query relies on unstated temporal or grouping assumptions.

---

## 7. In-Context Feedback Loop

The system actively ingests `dataset/feedback_log.csv`.
* Prior to generating queries, the `SemanticLayer` matches query intent against historical feedback records.
* Discovered lessons (e.g., *"Join targets table on both region and month"*, *"Always use window functions for top N per group"*) are automatically injected into the LLM prompt as few-shot constraints.

---

## 8. Installation & Quickstart

### Prerequisites
* Python 3.10+
* (Optional) Gemini or OpenAI API Key

### Setup
```bash
# 1. Navigate to the project directory
cd intelligent_query_engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys in .env
# (The repository already includes support for GEMINI_API_KEY and OPENAI_API_KEY)
```

### Running the Engine
```bash
# Run all queries from nl_queries.json using Gemini (default)
python run_engine.py --provider gemini

# Run all queries using OpenAI
python run_engine.py --provider openai

# Run a custom ad-hoc natural language query
python run_engine.py --query "Show total profit for Corporate customer segment"
```

---

## 9. Sample Query Outputs

All outputs strictly conform to the required JSON schema:

```json
[
  {
    "query": "Total sales in India for March",
    "generated_logic": "SELECT ROUND(SUM(revenue), 2) AS total_sales FROM sales WHERE country = 'India' AND month = '2024-03';",
    "result": 108.0,
    "confidence_score": 0.95,
    "explanation": "Filtered sales table for country 'India' and month '2024-03' and calculated total revenue using the metric formula quantity * unit_price * (1 - discount)."
  },
  {
    "query": "Top 2 cities by profit",
    "generated_logic": "SELECT city, ROUND(SUM(profit), 2) AS total_profit FROM sales GROUP BY city ORDER BY total_profit DESC LIMIT 2;",
    "result": [
      {
        "city": "New York",
        "total_profit": 200.0
      },
      {
        "city": "San Francisco",
        "total_profit": 180.0
      }
    ],
    "confidence_score": 0.95,
    "explanation": "Grouped records by city, aggregated profit using SUM(), and selected the top 2 cities descending."
  },
  {
    "query": "Which region missed its target in Feb?",
    "generated_logic": "WITH feb_actuals AS (SELECT region, SUM(revenue) AS actual_revenue FROM sales WHERE month = '2024-02' GROUP BY region) SELECT t.region, ROUND(COALESCE(a.actual_revenue, 0), 2) AS actual_revenue, t.target_revenue, ROUND(t.target_revenue - COALESCE(a.actual_revenue, 0), 2) AS shortfall FROM targets t LEFT JOIN feb_actuals a ON t.region = a.region WHERE t.month = '2024-02' AND COALESCE(a.actual_revenue, 0) < t.target_revenue;",
    "result": [
      {
        "region": "APAC",
        "actual_revenue": 75.0,
        "target_revenue": 6000.0,
        "shortfall": 5925.0
      },
      {
        "region": "EMEA",
        "actual_revenue": 255.0,
        "target_revenue": 7500.0,
        "shortfall": 7245.0
      },
      {
        "region": "NA",
        "actual_revenue": 1116.0,
        "target_revenue": 9500.0,
        "shortfall": 8384.0
      }
    ],
    "confidence_score": 0.95,
    "explanation": "Aggregated February actual revenue per region, joined with targets table on region and month '2024-02', and filtered for actual revenue less than target revenue."
  }
]
```

---

## 10. Future Improvements
1. **Dynamic Vector Store for Feedback:** Scale `feedback_log.csv` to thousands of user annotations using ChromaDB / FAISS for semantic embedding retrieval.
2. **Interactive Streamlit Web Dashboard:** Provide business analysts with an interactive conversational chat UI and instant charts.
3. **Automated Data Profiling:** Automatically detect column types, distributions, and cardinality upon dataset upload.
4. **Query Performance Optimization:** Support partition pruning and index hints for massive datasets (> 100M rows).
