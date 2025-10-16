"""Metrics for evaluating solver performance."""

import dspy
from typing import Optional, Union
from .judge import ComponentJudge


def basic_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None
) -> int:
    """
    Basic integer comparison metric for math problems.

    Args:
        example: Example with 'answer' field
        prediction: Prediction with 'result_text' field
        trace: Unused (for GEPA compatibility)
        pred_name: Unused (for GEPA compatibility)
        pred_trace: Unused (for GEPA compatibility)

    Returns:
        1 if correct, 0 otherwise
    """
    try:
        correct_answer = int(example['answer'])
        llm_answer = int(prediction.result_text)
        return int(correct_answer == llm_answer)
    except (ValueError, KeyError, AttributeError, TypeError):
        return 0


class MetricWithFeedback:
    """
    Metric that provides component feedback via LLM judge.

    This metric is compatible with GEPA's reflection-based optimization.
    It evaluates predictions and optionally generates feedback for
    specific components using an LLM judge.
    """

    def __init__(self, judge: ComponentJudge):
        """
        Initialize metric with judge.

        Args:
            judge: ComponentJudge instance for generating feedback
        """
        self.judge = judge

    def __call__(
        self,
        example,
        prediction,
        trace=None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[dict] = None
    ) -> Union[int, dspy.Prediction]:
        """
        Evaluate prediction with optional component feedback.

        Args:
            example: Example with 'answer' field
            prediction: Prediction with 'result_text' field
            trace: Full execution trace (optional)
            pred_name: Component name for feedback (provided by GEPA)
            pred_trace: Component trace for feedback (provided by GEPA)

        Returns:
            dspy.Prediction with score and feedback if pred_name provided,
            otherwise returns scalar score
        """

        # Calculate base score
        score = basic_metric(example, prediction)

        # If no component feedback requested, return scalar
        if pred_name is None:
            return score

        # Generate component feedback
        feedback = "None"
        if pred_trace is not None:
            try:
                prediction_trace = prediction.output_trace
                feedback_response = self.judge(
                    component_name=pred_name,
                    component_trace=pred_trace,
                    prediction_trace=prediction_trace if prediction_trace is not None else ""
                )
                feedback = feedback_response
            except Exception as e:
                feedback = f"Judge error: {str(e)}"

        print(f"Feedback: {feedback}")
        print(f"Score: {score}")
        return dspy.Prediction(score=score, feedback=feedback)

    async def __acall__(
        self,
        example,
        prediction,
        trace=None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[dict] = None
    ) -> Union[int, dspy.Prediction]:
        """
        Async evaluate prediction with optional component feedback.

        Args:
            example: Example with 'answer' field
            prediction: Prediction with 'result_text' field
            trace: Full execution trace (optional)
            pred_name: Component name for feedback (provided by GEPA)
            pred_trace: Component trace for feedback (provided by GEPA)

        Returns:
            dspy.Prediction with score and feedback if pred_name provided,
            otherwise returns scalar score
        """

        # Calculate base score
        score = basic_metric(example, prediction)

        # If no component feedback requested, return scalar
        if pred_name is None:
            return score

        # Generate component feedback asynchronously
        feedback = "None"
        if pred_trace is not None:
            try:
                prediction_trace = prediction.output_trace
                feedback_response = await self.judge.__acall__(
                    component_name=pred_name,
                    component_trace=pred_trace,
                    prediction_trace=prediction_trace if prediction_trace is not None else ""
                )
                feedback = feedback_response
            except Exception as e:
                feedback = f"Judge error: {str(e)}"

        print(f"Feedback: {feedback}")
        print(f"Score: {score}")
        return dspy.Prediction(score=score, feedback=feedback)
