import json
from typing import List, Dict, Any
from framework.metrics import evaluate_rag_output
from framework.utils import FileUpload


def evaluate_rag_metrics_from_jsonl(
    jsonl_file_path: str,
    model: str = "gpt-4o-mini",
    verbose: bool = False
) -> List[Dict[str, Any]]:

    results = []
    
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                test_case = json.loads(line)
                
                input_query = test_case.get("input", "")
                actual_output = test_case.get("actual_output", "")
                retrieval_context = test_case.get("retrieval_context", [])
                expected_output = test_case.get("expected_output")
                
                if not input_query or not actual_output:
                    print(f"Warning: Skipping line {line_num} - missing required fields")
                    continue
                
                rag_results = evaluate_rag_output(
                    input_query=input_query,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context,
                    expected_output=expected_output,
                    model=model,
                    verbose=verbose,
                )
                
                formatted_result = {
                    "input": input_query,
                    "actual_output": actual_output,
                    "metrics": {}
                }
                
                for metric_name, metric_data in rag_results.items():
                    score = metric_data.get("score", 0.0)
                    reason = metric_data.get("reason", "")
                    
                    formatted_result["metrics"][metric_name] = {
                        "score": score,
                        "reason": reason,
                        "pass": score >= 0.5 
                    }
                
                results.append(formatted_result)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return results


def evaluate_rag_metrics_with_upload(
    jsonl_file_path: str,
    model: str = "gpt-4o-mini",
    verbose: bool = False
) -> Dict[str, Any]:

    uploader = FileUpload()
    file_id = uploader.filepath(jsonl_file_path)
    created_on = uploader.get_created_datetime(file_id)
    filename = uploader.filename(file_id)
    status = uploader.status(file_id)
    
    print(f"\nFile ID: {file_id}")
    print(f"Uploaded: {created_on}")
    print(f"Filename: {filename}")
    print(f"Status: {status}")
    print("-" * 50)
    
    results = evaluate_rag_metrics_from_jsonl(
        jsonl_file_path=jsonl_file_path,
        model=model,
        verbose=verbose
    )
    
    return {
        "file_id": file_id,
        "created_on": created_on,
        "filename": filename,
        "status": status,
        "results": results
    }

def evaluate_rag_metrics_multi(jsonl_file_path: str,models: list,verbose: bool = False):
    all_results = {}

    for model in models:
        print(f"\n===== RAG Evaluation for Model: {model} =====")

        result = evaluate_rag_metrics_with_upload(
            jsonl_file_path=jsonl_file_path,
            model=model,
            verbose=verbose
        )

        all_results[model] = result["results"]

    return all_results
