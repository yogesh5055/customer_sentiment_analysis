import pandas as pd
import re
from collections import Counter

# Load file
df = pd.read_csv("customer_review.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text

df["cleaned"] = df["Feedback"].apply(clean_text)

# Rule-based sentiment keywords
positive_words = ["good", "great", "excellent", "amazing", "love", "fast", "happy", "helpful", "satisfied"]
negative_words = ["bad", "poor", "slow", "worst", "hate", "problem", "issue", "disappointed", "frustrating"]

def classify(text):
    p = sum(1 for w in positive_words if w in text)
    n = sum(1 for w in negative_words if w in text)
    if p > n:
        return "Positive"
    elif n > p:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["cleaned"].apply(classify)

# Summary
total = len(df)
pos = (df["Sentiment"] == "Positive").sum()
neu = (df["Sentiment"] == "Neutral").sum()
neg = (df["Sentiment"] == "Negative").sum()

print("=== SENTIMENT SUMMARY ===")
print("Total:", total)
print("Positive:", pos, f"({pos/total*100:.2f}%)")
print("Neutral:", neu, f"({neu/total*100:.2f}%)")
print("Negative:", neg, f"({neg/total*100:.2f}%)")

# Top words
stopwords = ["the","and","a","an","is","it","to","for","of","in","on","with",
             "this","that","was","are","i","we","you","they","be","but","have","has"]

all_words = []

for text in df["cleaned"]:
    words = text.split()
    all_words.extend([w for w in words if w not in stopwords and len(w) > 2])

print("\n=== TOP 20 WORDS ===")
for word, freq in Counter(all_words).most_common(20):
    print(word, ":", freq)

# Save output
df.to_csv("feedback_sentiment_output.csv", index=False)

print("\nAnalysis saved to 'feedback_sentiment_output.csv'")
