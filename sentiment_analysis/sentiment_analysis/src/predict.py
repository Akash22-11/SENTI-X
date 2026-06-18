import os
import sys
import argparse
import joblib
import pandas as pd

from preprocessing import TextPreprocessor, download_nltk_resources




# New
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "sentiment_pipeline.joblib")

class SentimentPredictor:
    def __init__(self, model_path=MODEL_PATH):
        

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_model.py first."
            )
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.preprocessor = bundle["preprocessor"]
        self.model_name = bundle["model_name"]
        self.labels = bundle["labels"]

    def predict(self, text: str) -> dict:
        clean = self.preprocessor.preprocess(text)
        pred = self.pipeline.predict([clean])[0]

        result = {"text": text, "sentiment": pred}

        # Add probability scores if the model supports it
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            probs = self.pipeline.predict_proba([clean])[0]
            classes = self.pipeline.named_steps["clf"].classes_
            result["confidence"] = dict(zip(classes, [round(float(p), 4) for p in probs]))

        return result

    def predict_batch(self, texts) -> pd.DataFrame:
        clean_texts = self.preprocessor.preprocess_batch(texts)
        preds = self.pipeline.predict(clean_texts)
        return pd.DataFrame({"text": texts, "sentiment": preds})


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of text.")
    parser.add_argument("--text", type=str, help="Single text string to analyze.")
    parser.add_argument("--csv", type=str, help="Path to CSV file with a 'text' column.")
    parser.add_argument("--output", type=str, default="output/predictions.csv",
                        help="Output path for batch predictions.")
    args = parser.parse_args()

    download_nltk_resources()
    predictor = SentimentPredictor()
    print(f"Loaded model: {predictor.model_name} | Labels: {predictor.labels}\n")

    if args.text:
        result = predictor.predict(args.text)
        print(f"Text     : {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        if "text" not in df.columns:
            sys.exit("CSV must contain a 'text' column.")
        preds_df = predictor.predict_batch(df["text"])
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        preds_df.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")
        print(preds_df["sentiment"].value_counts())

    else:
        # Interactive demo mode
        demo_texts = [
            "I really love how easy this app is to use!",
            "This was a terrible experience, I want a refund.",
            "The product arrived on time, nothing special.",
        ]
        for t in demo_texts:
            result = predictor.predict(t)
            print(f"Text     : {result['text']}")
            print(f"Sentiment: {result['sentiment']}")
            if "confidence" in result:
                print(f"Confidence: {result['confidence']}")
            print("-" * 50)


if __name__ == "__main__":
    main()