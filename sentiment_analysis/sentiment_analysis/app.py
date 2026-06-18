import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def run_setup():
    print("=== Setup: downloading NLTK data & generating sample dataset ===")
    from src.preprocessing import download_nltk_resources
    download_nltk_resources()
    subprocess.run([sys.executable, "generate_sample_data.py"], check=True)


def run_train():
    print("=== Training sentiment model ===")
    subprocess.run([sys.executable, "src/train_model.py"], check=True)


def run_visualize():
    print("=== Generating visualizations ===")
    subprocess.run([sys.executable, "src/visualize.py"], check=True)


def run_predict(args):
    cmd = [sys.executable, "src/predict.py"]
    if args.csv:
        cmd += ["--csv", args.csv, "--output", args.output]
    subprocess.run(cmd, check=True)


def run_analyze():
    from src.predict import SentimentPredictor
    predictor = SentimentPredictor()
    print(f"Loaded model: {predictor.model_name} | Labels: {predictor.labels}")
    print("Type a sentence to analyze its sentiment (type 'exit' to quit).\n")

    while True:
        text = input(">> ")
        if text.strip().lower() in ("exit", "quit"):
            break
        if not text.strip():
            continue
        result = predictor.predict(text)
        print(f"Sentiment: {result['sentiment']}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Project CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("setup", help="Download NLTK data and generate sample dataset")
    subparsers.add_parser("train", help="Train the sentiment model")
    subparsers.add_parser("visualize", help="Generate EDA plots")
    subparsers.add_parser("analyze", help="Interactive sentiment analysis")

    predict_parser = subparsers.add_parser("predict", help="Run predictions")
    predict_parser.add_argument("--csv", type=str, help="Path to CSV with 'text' column")
    predict_parser.add_argument("--output", type=str, default="output/predictions.csv")

    args = parser.parse_args()

    if args.command == "setup":
        run_setup()
    elif args.command == "train":
        run_train()
    elif args.command == "visualize":
        run_visualize()
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "analyze":
        run_analyze()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
