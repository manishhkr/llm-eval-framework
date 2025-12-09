from typing import Dict, Any, Optional, List

from .test_case import RAGTestCase
from .relevancy import AnswerRelevancyMetric
from .faithfulness import FaithfulnessMetric
from .contextual_precision import ContextualPrecisionMetric
from .contextual_recall import ContextualRecallMetric
from .contextual_relevancy import ContextualRelevancyMetric


def evaluate_rag_output(
    input_query: str,
    actual_output: str,
    retrieval_context: List[str],
    expected_output: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:

    tc = RAGTestCase(
        input=input_query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    answer_relevancy = AnswerRelevancyMetric(model=model, verbose=verbose)
    faithfulness = FaithfulnessMetric(model=model, verbose=verbose)
    contextual_precision = ContextualPrecisionMetric(model=model, verbose=verbose)
    contextual_recall = ContextualRecallMetric(model=model, verbose=verbose)
    contextual_relevancy = ContextualRelevancyMetric(model=model, verbose=verbose)

    metrics = {
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
        "contextual_precision": contextual_precision,
        "contextual_recall": contextual_recall,
        "contextual_relevancy": contextual_relevancy,
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, metric in metrics.items():
        score = metric.measure(tc)
        results[name] = {
            "score": score,
            "reason": metric.reason,
        }

    return results
