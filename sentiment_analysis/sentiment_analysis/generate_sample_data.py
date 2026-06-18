import os
import pandas as pd
import random

random.seed(42)

positive_samples = [
    "I absolutely love this product, it exceeded my expectations!",
    "Great customer service, very responsive and helpful.",
    "The quality is amazing, will definitely buy again.",
    "Fast shipping and excellent packaging, very satisfied.",
    "This is the best purchase I've made all year.",
    "Outstanding value for the price, highly recommend.",
    "The team resolved my issue quickly and professionally.",
    "Five stars! Works perfectly and looks great too.",
    "Very happy with the experience, will recommend to friends.",
    "Excellent product quality and the design is fantastic.",
    "Customer support went above and beyond to help me.",
    "I'm impressed with how durable and reliable this is.",
    "Amazing experience from start to finish, thank you!",
    "The app is intuitive and makes my life so much easier.",
    "Top notch service, exactly what I was looking for.",
    "Couldn't be happier with my purchase, great job team.",
    "The product works exactly as described, very pleased.",
    "Loved the smooth checkout process and quick delivery.",
    "Such a wonderful product, my whole family enjoys it.",
    "Brilliant! Easy to use and great value for money."
]

negative_samples = [
    "Terrible experience, the product broke after one day.",
    "Customer service was rude and unhelpful.",
    "I'm very disappointed with the quality, not worth the price.",
    "The shipping took forever and the box arrived damaged.",
    "This is the worst purchase I've made, total waste of money.",
    "Poor build quality, it stopped working within a week.",
    "I had to wait weeks for a response from support.",
    "One star, completely unsatisfied with this purchase.",
    "Very unhappy with the experience, would not recommend.",
    "The product description was misleading and inaccurate.",
    "Support never resolved my issue, very frustrating.",
    "Cheaply made and feels like it will fall apart soon.",
    "Awful experience overall, do not buy this product.",
    "The app constantly crashes and is full of bugs.",
    "Substandard service, I expected much better.",
    "Extremely disappointed, requesting a refund immediately.",
    "The product didn't work as advertised at all.",
    "Horrible packaging led to a damaged item on arrival.",
    "Such a frustrating product, nothing works correctly.",
    "Waste of money, would never purchase from here again."
]

neutral_samples = [
    "The product arrived on time as expected.",
    "It's an okay product, does what it says.",
    "The packaging was standard, nothing special.",
    "I received the item, will update after using it more.",
    "The service was fine, no complaints but nothing extraordinary.",
    "It works as described, average quality overall.",
    "The delivery was on schedule, product looks normal.",
    "Not bad, not great, just a regular product.",
    "The instructions were clear but the design is basic.",
    "It does the job, nothing more nothing less.",
    "Average experience overall, met basic expectations.",
    "The product is functional but could use improvements.",
    "I have mixed feelings, some parts good some not.",
    "Standard quality for the price point, acceptable.",
    "It's fine for everyday use, nothing remarkable.",
    "The website was easy to navigate, order was processed normally.",
    "Received as described, no issues so far.",
    "Decent product overall, meets basic requirements.",
    "It's an average item, works most of the time.",
    "Okay quality, comparable to similar products in this range."
]

data = []
for text in positive_samples:
    data.append({"text": text, "sentiment": "positive"})
for text in negative_samples:
    data.append({"text": text, "sentiment": "negative"})
for text in neutral_samples:
    data.append({"text": text, "sentiment": "neutral"})

# Expand dataset with slight variations for a more robust training set
expanded_data = []
suffixes = ["", " Overall a solid take.", " Just my honest opinion.", " That's my feedback.", " Thanks for reading."]
for row in data:
    for suffix in suffixes:
        expanded_data.append({"text": row["text"] + suffix, "sentiment": row["sentiment"]})

df = pd.DataFrame(expanded_data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Fix: anchor output path to script location and create folder if missing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "feedback.csv")

df.to_csv(output_path, index=False)
print(f"Generated dataset with {len(df)} rows -> {output_path}")
print(df["sentiment"].value_counts())