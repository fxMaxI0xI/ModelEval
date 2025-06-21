# LLM Benchmark Evaluation

Standalone-Repo f\u00fcr die Evaluation von HF-Modellen auf Benchmarks wie CodalBench, AirBench etc.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Usage

python eval/eval_codalbench.py --model_name_or_path "google/gemma-2b"
```

---

### **requirements.txt** *(minimal, ggf. anpassen)*

transformers
datasets
torch
inspect_ai
openai

---

### **eval/eval_codalbench.py** *(Skelett f\u00fcr CodalBench; adaptierbar!)*

```python
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from inspect_ai.dataset import hf_dataset, FieldSpec
from inspect_ai.scorer import model_graded_qa, mean, stderr
from inspect_ai import eval_set, Task, task
from safe_llm_finetune.evaluation.prompt_templates import CODAL_INSTRUCTION_TEMPLATE, CODAL_PROMPT_TEMPLATES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    args = parser.parse_args()

    # Lade HF Model + Tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Lade CodalBench Datensatz
    dataset = hf_dataset(
        path="coseal/codal-bench",
        split="test",
        sample_fields=FieldSpec(
            input="instruction",
            target="claude-3-sonnet-20240229_response",
            metadata=["preference"],
        ),
    )

    # Ein Beispiel-Task, alle anderen kannst du genauso bauen!
    @task(name="codal_instruction-following")
    def t():
        filtered = dataset.filter(lambda s: s.metadata["preference"] == "instruction-following")
        scorer = model_graded_qa(
            template=CODAL_PROMPT_TEMPLATES["instruction-following"],
            instructions=CODAL_INSTRUCTION_TEMPLATE,
            grade_pattern=r"GRADE:\s*(\d+)/10",
            model="openai/gpt-4o-mini",  # Oder "hf/local" f\u00fcr dein eigenes Modell
        )
        return Task(dataset=filtered, scorer=scorer, metrics=[mean(), stderr()])

    results = eval_set(tasks=t, model="hf/local", model_args={"model": model, "tokenizer": tokenizer})
    print(results)

if __name__ == "__main__":
    main()
```

Wie geht\u2019s weiter?
Repo klonen/erstellen
Files wie oben einfügen (evtl. nur einen eval_codalbench.py zum Start)
Optional: Codes f\u00fcr andere Benches kopieren oder adaptieren
requirements.txt installieren
Fertig!
Meld dich, wenn du ein konkretes Template-Repo willst (z.\u202fB. auf GitHub),
oder sag, welche Benchmarks/Modelle/Optionen du brauchst, dann mach ich\u2019s ready!
