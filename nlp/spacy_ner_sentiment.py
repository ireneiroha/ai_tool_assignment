import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
# Create a sample amazon_reviews.csv file if it doesn't exist
try:
    df = pd.read_csv("amazon_reviews.csv")
except FileNotFoundError:
    data = {
        "review_id": [1, 2, 3, 4, 5],
        "review_text": [
            "I love the Apple AirPods. They're far better than my old Samsung earbuds.",
            "The product arrived late and the customer support was terrible.",
            "Excellent quality! The camera resolution is amazing.",
            "I hate the design. It's the worst phone I've ever used.",
            "The packaging was good but I didn't like the color."
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv("amazon_reviews.csv", index=False)
    print("Sample dataset created as amazon_reviews.csv")

# Define sentiment words
positive_words = ['love', 'great', 'excellent', 'amazing', 'good', 'better', 'satisfied']
negative_words = ['bad', 'worst', 'hate', 'terrible', 'late', 'poor', 'didn’t like']

# Processing each review
results = []

for idx, row in df.iterrows():
    text = row['review_text']
    doc = nlp(text)

    # Named Entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Sentiment Analysis
    sentiment_score = 0
    for token in doc:
        token_lower = token.text.lower()
        if token_lower in positive_words:
            sentiment_score += 1
        elif token_lower in negative_words:
            sentiment_score -= 1

    sentiment = "Positive" if sentiment_score > 0 else ("Negative" if sentiment_score < 0 else "Neutral")

    results.append({
        "review_id": row['review_id'],
        "review_text": text,
        "entities": entities,
        "sentiment": sentiment
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("amazon_reviews_results.csv", index=False)

# Display sample output
print(results_df.head())
print("\n✅ NLP processing completed. Results saved to amazon_reviews_results.csv")
