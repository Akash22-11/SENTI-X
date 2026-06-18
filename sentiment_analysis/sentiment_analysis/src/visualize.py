import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from preprocessing import TextPreprocessor, download_nltk_resources

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "output"))


def plot_sentiment_distribution(df, save_path):
    plt.figure(figsize=(6, 4))
    order = df["sentiment"].value_counts().index
    sns.countplot(data=df, x="sentiment", order=order, hue="sentiment",
            palette="Greens_d", legend=False)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_text_length_distribution(df, save_path):
    text_lengths = df["text"].str.split().str.len()
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df.assign(text_length=text_lengths), x="text_length",
                hue="sentiment", multiple="stack", bins=15)
    plt.title("Text Length Distribution by Sentiment")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_wordclouds(df, preprocessor, save_path):
    sentiments = df["sentiment"].unique()
    fig, axes = plt.subplots(1, len(sentiments), figsize=(6 * len(sentiments), 5))

    if len(sentiments) == 1:
        axes = [axes]

    for ax, sentiment in zip(axes, sentiments):
        texts = df[df["sentiment"] == sentiment]["text"]
        clean_texts = preprocessor.preprocess_batch(texts)
        combined_text = " ".join(clean_texts)

        if not combined_text.strip():
            ax.set_title(f"{sentiment.capitalize()} — No Data", fontsize=14)
            ax.axis("off")
            continue

        wc = WordCloud(width=500, height=400, background_color="white",
                    colormap="Greens").generate(combined_text)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{sentiment.capitalize()} Feedback", fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def main():
    download_nltk_resources()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "feedback.csv"))
    df = pd.read_csv(csv_path)
    preprocessor = TextPreprocessor()

    plot_sentiment_distribution(df, os.path.join(OUTPUT_DIR, "sentiment_distribution.png"))
    plot_text_length_distribution(df, os.path.join(OUTPUT_DIR, "text_length_distribution.png"))
    plot_wordclouds(df, preprocessor, os.path.join(OUTPUT_DIR, "wordclouds.png"))


if __name__ == "__main__":
    main()