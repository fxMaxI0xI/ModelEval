import argparse

from eval.codalbench import CodalBench


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CodalBench evaluation")
    parser.add_argument("--model_name_or_path", required=True, help="Path or HF model id")
    parser.add_argument("--preference", help="Optional preference category")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    args = parser.parse_args()

    evaluator = CodalBench(debug=args.debug, preference=args.preference)
    results = evaluator.run_eval(
        model_path=args.model_name_or_path,
        tokenizer_path=args.model_name_or_path,
        log_dir=None,
        limit=args.limit,
    )
    print(results)


if __name__ == "__main__":
    main()
