"""CodalBench evaluator similar to safe_llm_finetune."""

from typing import Optional, List

from inspect_ai import Task, eval_set, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import mean, model_graded_qa, stderr
from inspect_ai.solver import chain_of_thought, generate

from .base import Evaluator
from .prompt_templates import CODAL_INSTRUCTION_TEMPLATE, CODAL_PROMPT_TEMPLATES


class CodalBench(Evaluator):
    """Run CodalBench evaluation on a given model."""

    def __init__(self, debug: bool = False, preference: Optional[str] = None, judge_model: str = "openai/gpt-4o-mini"):
        super().__init__(debug)
        self.preference = preference
        self.judge_model = judge_model

        if preference is not None and preference not in CODAL_PROMPT_TEMPLATES:
            raise ValueError(
                f"Invalid preference '{preference}'. Must be one of: {list(CODAL_PROMPT_TEMPLATES.keys())}"
            )

        self.dataset = hf_dataset(
            path="coseal/codal-bench",
            split="test",
            sample_fields=FieldSpec(
                input="instruction",
                target="claude-3-sonnet-20240229_response",
                metadata=["preference"],
            ),
        )

    def get_name(self) -> str:
        return "CodalBench"

    def available_preferences(self) -> List[str]:
        return list(CODAL_PROMPT_TEMPLATES.keys())

    def _create_single_task(self, preference: str):
        @task(name=f"codal_{preference}")
        def pref_task():
            filtered = self.dataset.filter(lambda s: s.metadata.get("preference") == preference)
            scorer = model_graded_qa(
                template=CODAL_PROMPT_TEMPLATES[preference],
                instructions=CODAL_INSTRUCTION_TEMPLATE,
                grade_pattern=r"GRADE:\s*(\d+)/10",
                model=self.judge_model,
            )
            return Task(
                dataset=filtered,
                solver=[chain_of_thought(), generate()],
                scorer=scorer,
                metrics=[mean(), stderr()],
            )

        return pref_task()

    def create_task(self):
        if self.preference is not None:
            return self._create_single_task(self.preference)

        tasks = []
        for pref in CODAL_PROMPT_TEMPLATES.keys():
            tasks.append(self._create_single_task(pref))
        return tasks
