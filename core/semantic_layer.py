import json
import csv
from pathlib import Path
from typing import Dict, Any, List

class SemanticLayer:
    """
    Manages data dictionary, metric definitions, dimensions, synonyms,
    and feedback history for in-context grounding.
    """
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.data_dict_file = self.dataset_dir / "data_dictionary.json"
        self.feedback_file = self.dataset_dir / "feedback_log.csv"
        self.data_dict = self._load_data_dictionary()
        self.feedback_entries = self._load_feedback_log()

    def _load_data_dictionary(self) -> Dict[str, Any]:
        if self.data_dict_file.exists():
            with open(self.data_dict_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "metrics": {
                "revenue": "quantity * unit_price * (1 - discount)",
                "profit": "profit",
                "orders": "count(order_id)",
                "avg_order_value": "revenue / orders"
            },
            "dimensions": [
                "region", "country", "city", "customer_segment",
                "product_category", "product_subcategory", "order_date"
            ],
            "synonyms": {
                "sales": "revenue",
                "income": "revenue",
                "earnings": "profit",
                "orders": "count(order_id)",
                "aov": "avg_order_value"
            },
            "time_mappings": {
                "last month": "previous calendar month",
                "this quarter": "current quarter",
                "yoy": "year over year"
            }
        }

    def _load_feedback_log(self) -> List[Dict[str, str]]:
        if not self.feedback_file.exists():
            return []
        entries = []
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entries.append(row)
        except Exception:
            pass
        return entries

    def get_schema_context(self) -> str:
        """
        Builds a comprehensive prompt string describing database views,
        columns, calculations, and domain rules.
        """
        metrics = self.data_dict.get("metrics", {})
        dimensions = self.data_dict.get("dimensions", [])
        synonyms = self.data_dict.get("synonyms", {})
        time_mappings = self.data_dict.get("time_mappings", {})

        metrics_str = "\n".join([f"  - {k}: {v}" for k, v in metrics.items()])
        dims_str = ", ".join(dimensions)
        synonyms_str = ", ".join([f"'{k}' -> '{v}'" for k, v in synonyms.items()])
        time_str = ", ".join([f"'{k}' -> '{v}'" for k, v in time_mappings.items()])

        schema_prompt = f"""
AVAILABLE DATABASE TABLES / VIEWS (in DuckDB):

1. View `sales` (Pre-processed Analytical View of sales_data):
   Columns:
     - order_id (VARCHAR / INTEGER): unique order identifier
     - order_date (DATE): order timestamp (YYYY-MM-DD)
     - month (VARCHAR): formatted YYYY-MM (e.g. '2024-01', '2024-02', '2024-03')
     - year (INTEGER): calendar year (e.g. 2024)
     - quarter (VARCHAR): calendar quarter (e.g. 'Q1')
     - region (VARCHAR): e.g. 'APAC', 'EMEA', 'NA'
     - country (VARCHAR): e.g. 'India', 'USA', 'Germany', 'UK', 'France'
     - city (VARCHAR): e.g. 'Mumbai', 'Berlin', 'Delhi', 'New York', 'London', 'Bangalore', 'Chicago', 'Paris', 'San Francisco'
     - customer_id (VARCHAR): e.g. 'C001', 'C002', etc.
     - customer_segment (VARCHAR): 'Consumer', 'Corporate', 'Home Office'
     - product_category (VARCHAR): 'Furniture', 'Technology', 'Office Supplies'
     - product_subcategory (VARCHAR): 'Chairs', 'Phones', 'Binders', 'Laptops', 'Tables', 'Accessories', 'Paper'
     - product_name (VARCHAR): 'Ergo Chair', 'iPhone 14', 'MacBook Air', etc.
     - quantity (INTEGER): items sold
     - unit_price (DOUBLE): price per unit
     - discount (DOUBLE): decimal discount (0.0 to 1.0)
     - shipping_cost (DOUBLE): shipping expense
     - profit (DOUBLE): recorded profit
     - revenue (DOUBLE): pre-computed formula: quantity * unit_price * (1 - discount)

2. View `targets` (Targets dataset):
   Columns:
     - region (VARCHAR): e.g. 'APAC', 'EMEA', 'NA'
     - month (VARCHAR): 'YYYY-MM' (e.g. '2024-01', '2024-02', '2024-03')
     - target_revenue (DOUBLE): revenue target for the region in that month

BUSINESS DEFINITIONS & METRIC FORMULAS:
{metrics_str}

DIMENSIONS:
{dims_str}

BUSINESS SYNONYMS:
{synonyms_str}

TIME MAPPINGS:
{time_str}
"""
        return schema_prompt.strip()

    def get_relevant_feedback(self, query: str) -> str:
        """
        Retrieves relevant feedback lessons learned from feedback_log.csv
        based on token overlap.
        """
        if not self.feedback_entries:
            return ""

        query_tokens = set(query.lower().split())
        scored_entries = []

        for entry in self.feedback_entries:
            past_q = entry.get("query", "").lower()
            past_tokens = set(past_q.split())
            overlap = len(query_tokens.intersection(past_tokens))
            if overlap > 0:
                scored_entries.append((overlap, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        top_entries = [item[1] for item in scored_entries[:3]]

        if not top_entries:
            # If no direct match, provide general guidance from log
            top_entries = self.feedback_entries[:2]

        formatted = []
        for e in top_entries:
            formatted.append(
                f"- For query like: \"{e.get('query')}\"\n"
                f"  Feedback: {e.get('feedback')}\n"
                f"  Correct Pattern: {e.get('correct_logic')}\n"
                f"  Lesson: {e.get('lesson_learned')}"
            )
        return "\n\n".join(formatted)
