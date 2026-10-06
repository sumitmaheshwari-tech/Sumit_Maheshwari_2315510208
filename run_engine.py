import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from core.semantic_layer import SemanticLayer
from core.db_executor import DBExecutor
from core.prompt_builder import PromptBuilder
from core.llm_client import LLMClient
from core.self_corrector import SelfCorrectionEngine
from core.confidence_scorer import ConfidenceScorer

def run_pipeline(
    dataset_dir: Path,
    provider: str = "gemini",
    single_query: str = None,
    output_path: Path = None
) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("  INTELLIGENT ANALYTICAL QUERY ENGINE (Office AI Solution)")
    print(f"  Provider: {provider.upper()} | Dataset: {dataset_dir}")
    print("=" * 70)

    # 1. Initialize Core Components
    semantic_layer = SemanticLayer(dataset_dir)
    schema_context = semantic_layer.get_schema_context()

    db_executor = DBExecutor(dataset_dir)
    prompt_builder = PromptBuilder(schema_context)
    llm_client = LLMClient(provider=provider)
    self_corrector = SelfCorrectionEngine(db_executor, prompt_builder, llm_client)
    confidence_scorer = ConfidenceScorer()

    # 2. Determine Queries
    queries_to_run = []
    if single_query:
        queries_to_run.append({"query": single_query})
    else:
        nl_queries_file = dataset_dir / "nl_queries.json"
        if nl_queries_file.exists():
            with open(nl_queries_file, "r", encoding="utf-8") as f:
                queries_to_run = json.load(f)
        else:
            print(f"Warning: {nl_queries_file} not found.")

    final_outputs = []

    # 3. Process Queries
    for idx, item in enumerate(queries_to_run, 1):
        query_text = item.get("query")
        print(f"\n[{idx}/{len(queries_to_run)}] PROCESSING QUERY:")
        print(f"  Natural Language: \"{query_text}\"")

        # Step A: In-Context Feedback Retrieval
        feedback_context = semantic_layer.get_relevant_feedback(query_text)

        # Step B: LLM Generation
        generation_prompt = prompt_builder.build_generation_prompt(
            query=query_text,
            feedback_context=feedback_context
        )
        llm_response = llm_client.generate(generation_prompt)

        initial_sql = llm_response.get("generated_logic", "")
        initial_explanation = llm_response.get("explanation", "")
        initial_confidence = float(llm_response.get("self_confidence", 0.9))

        # Step C: Execution & Self-Correction
        exec_outcome = self_corrector.execute_with_self_healing(
            query=query_text,
            initial_sql=initial_sql,
            initial_explanation=initial_explanation,
            initial_confidence=initial_confidence
        )

        # Step D: Multi-Factor Confidence Scoring
        final_confidence = confidence_scorer.calculate_score(
            execution_success=exec_outcome["success"],
            retry_count=exec_outcome["retries"],
            sql_query=exec_outcome["sql"],
            llm_self_confidence=exec_outcome["self_confidence"],
            has_ambiguity=("baseline" in exec_outcome["explanation"].lower())
        )

        # Step E: Construct Output Object
        output_item = {
            "query": query_text,
            "generated_logic": exec_outcome["sql"].strip(),
            "result": exec_outcome["result"],
            "confidence_score": final_confidence,
            "explanation": exec_outcome["explanation"].strip()
        }
        final_outputs.append(output_item)

        print(f"  Generated Logic:\n    {exec_outcome['sql'].strip()}")
        print(f"  Result: {exec_outcome['result']}")
        print(f"  Confidence Score: {final_confidence}")
        print(f"  Explanation: {exec_outcome['explanation'].strip()}")

    # 4. Save Final Outputs
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_outputs, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Results successfully saved to: {output_path}")

    db_executor.close()
    return final_outputs

def main():
    parser = argparse.ArgumentParser(description="Intelligent Analytics Query Engine")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--provider", type=str, default="office_ai_lab", help="LLM Provider Engine")
    parser.add_argument("--query", type=str, default=None, help="Run a single ad-hoc query")
    parser.add_argument("--output", type=str, default="output.json", help="Path to save output JSON")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat session with your data")

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    dataset_path = (base_dir / args.dataset).resolve()
    output_file = (base_dir / args.output).resolve()

    if args.interactive:
        print("=" * 70)
        print("  INTERACTIVE ANALYTICS CHAT (Office AI Solution)")
        print("  Type your questions in plain English (or type 'exit' / 'quit' to stop)")
        print("=" * 70)
        
        semantic_layer = SemanticLayer(dataset_path)
        schema_context = semantic_layer.get_schema_context()
        db_executor = DBExecutor(dataset_path)
        prompt_builder = PromptBuilder(schema_context)
        llm_client = LLMClient(provider=args.provider)
        self_corrector = SelfCorrectionEngine(db_executor, prompt_builder, llm_client)
        confidence_scorer = ConfidenceScorer()

        while True:
            try:
                user_q = input("\nAsk a question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    print("Exiting interactive session. Goodbye!")
                    break

                feedback_ctx = semantic_layer.get_relevant_feedback(user_q)
                prompt = prompt_builder.build_generation_prompt(user_q, feedback_ctx)
                gen_res = llm_client.generate(prompt)

                exec_out = self_corrector.execute_with_self_healing(
                    query=user_q,
                    initial_sql=gen_res.get("generated_logic", ""),
                    initial_explanation=gen_res.get("explanation", ""),
                    initial_confidence=float(gen_res.get("self_confidence", 0.9))
                )

                conf = confidence_scorer.calculate_score(
                    execution_success=exec_out["success"],
                    retry_count=exec_out["retries"],
                    sql_query=exec_out["sql"],
                    llm_self_confidence=exec_out["self_confidence"]
                )

                print("\n" + "-" * 50)
                print(f"Generated Logic:\n  {exec_out['sql']}")
                print(f"Result:\n  {exec_out['result']}")
                print(f"Confidence Score: {conf}")
                print(f"Explanation:\n  {exec_out['explanation']}")
                print("-" * 50)

            except (KeyboardInterrupt, EOFError):
                print("\nExiting session.")
                break

        db_executor.close()
        return

    run_pipeline(
        dataset_dir=dataset_path,
        provider=args.provider,
        single_query=args.query,
        output_path=output_file
    )

if __name__ == "__main__":
    main()
