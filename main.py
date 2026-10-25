import argparse
from evaluation.runner import run_pointwise

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation pipeline")
    parser.add_argument("--suite", default="data/test_suite.json", help="Path to test suite (.json or .yaml)")
    parser.add_argument("--backend", default=None, help="Backend to use: 'gemini' or 'mock'")
    parser.add_argument("--model", default=None, help="Gemini model identifier")
    args = parser.parse_args()

    run_pointwise(suite_path=args.suite, backend=args.backend, model=args.model)

if __name__ == "__main__":
    main()
