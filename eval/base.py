from abc import ABC, abstractmethod
from typing import List, Optional

from inspect_ai import eval as inspect_eval, Task
from inspect_ai.log import EvalLog


class Evaluator(ABC):
    """Abstract base class for evaluation classes."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.dataset = None

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the evaluation."""
        raise NotImplementedError

    @abstractmethod
    def create_task(self) -> Task | List[Task]:
        """Create the inspect_ai task object."""
        raise NotImplementedError

    def run_eval(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[EvalLog]:
        """Run the evaluation and return the EvalLog list."""

        task = self.create_task()
        if self.debug:
            return inspect_eval(
                tasks=task,
                model="openai/gpt-4o-mini",
                log_dir=log_dir,
                limit=10 if limit is None else limit,
            )

        eval_args = dict(
            tasks=task,
            model="hf/local",
            model_args=dict(model_path=model_path, tokenizer_path=tokenizer_path or model_path),
            log_dir=log_dir,
            fail_on_error=False,
            retry_on_error=5,
            trace=False,
        )
        if limit is not None:
            eval_args["limit"] = limit
        return inspect_eval(**eval_args)
