import os
import sys
import json
import csv
from typing import Any
from pathlib import Path
from http.server import BaseHTTPRequestHandler
import urllib.parse

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from core.semantic_layer import SemanticLayer
from core.db_executor import DBExecutor
from core.prompt_builder import PromptBuilder
from core.llm_client import LLMClient
from core.self_corrector import SelfCorrectionEngine
from core.confidence_scorer import ConfidenceScorer

DATASET_DIR = BASE_DIR / "dataset"
STATIC_DIR = BASE_DIR / "static"

semantic_layer = SemanticLayer(DATASET_DIR)
schema_context = semantic_layer.get_schema_context()
db_executor = DBExecutor(DATASET_DIR)
prompt_builder = PromptBuilder(schema_context)
confidence_scorer = ConfidenceScorer()

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path in ["/", "/index.html"]:
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/datasets":
            self._handle_get_datasets()
        elif path == "/api/feedback":
            self._handle_get_feedback()
        elif path == "/api/samples":
            self._handle_get_samples()
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/api/query":
            self._handle_post_query()
        else:
            self.send_error(404, "Endpoint not found")

    def _serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(404, "File not found")
            return
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_get_datasets(self):
        sales_data = []
        sales_csv = DATASET_DIR / "sales_data.csv"
        if sales_csv.exists():
            with open(sales_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    sales_data.append({
                        "order_id": r.get("order_id"),
                        "order_date": r.get("order_date"),
                        "region": r.get("region"),
                        "country": r.get("country"),
                        "city": r.get("city"),
                        "product_name": r.get("product_name"),
                        "quantity": int(r.get("quantity", 0)),
                        "unit_price": float(r.get("unit_price", 0.0)),
                        "discount": float(r.get("discount", 0.0)),
                        "profit": float(r.get("profit", 0.0))
                    })

        targets_data = []
        targets_csv = DATASET_DIR / "targets.csv"
        if targets_csv.exists():
            with open(targets_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    targets_data.append({
                        "region": r.get("region"),
                        "month": r.get("month"),
                        "target_revenue": float(r.get("target_revenue", 0.0))
                    })

        data = {
            "sales": sales_data,
            "targets": targets_data
        }
        self._send_json(data)

    def _handle_get_feedback(self):
        feedback_list = semantic_layer.feedback_entries
        self._send_json(feedback_list)

    def _handle_get_samples(self):
        samples_file = DATASET_DIR / "nl_queries.json"
        if samples_file.exists():
            with open(samples_file, "r", encoding="utf-8") as f:
                samples = json.load(f)
        else:
            samples = []
        self._send_json(samples)

    def _handle_post_query(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(body)
            raw_query = req.get("query", "").strip()
            query_text = raw_query.strip('"').strip("'").rstrip('*').strip()
            provider = req.get("provider", "gemini").strip()
            
            if not query_text:
                self._send_json({"error": "Empty query provided"}, status=400)
                return

            llm_client = LLMClient(provider=provider)
            self_corrector = SelfCorrectionEngine(db_executor, prompt_builder, llm_client)

            feedback_context = semantic_layer.get_relevant_feedback(query_text)
            prompt = prompt_builder.build_generation_prompt(query_text, feedback_context)
            gen_res = llm_client.generate(prompt)

            exec_out = self_corrector.execute_with_self_healing(
                query=query_text,
                initial_sql=gen_res.get("generated_logic", ""),
                initial_explanation=gen_res.get("explanation", ""),
                initial_confidence=float(gen_res.get("self_confidence", 0.9))
            )

            final_conf = confidence_scorer.calculate_score(
                execution_success=exec_out["success"],
                retry_count=exec_out["retries"],
                sql_query=exec_out["sql"],
                llm_self_confidence=exec_out["self_confidence"]
            )

            response_data = {
                "query": query_text,
                "generated_logic": exec_out["sql"].strip(),
                "result": exec_out["result"],
                "confidence_score": final_conf,
                "explanation": exec_out["explanation"].strip(),
                "retries": exec_out["retries"]
            }
            self._send_json(response_data)

        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
