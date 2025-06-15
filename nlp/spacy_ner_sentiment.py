import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample reviews (replace with Amazon review data if needed)
reviews = [
    "This Samsung TV has amazing picture quality!",
    "Absolutely love my new Apple iPhone. It's fast and sleek.",
    "I had issues with the Lenovo laptop battery dying too quickly.",
    "The Nike running shoes are super comfortable and stylish.",
    "Terrible experience with the Sony headphones. Very poor sound."
]

# Rule-based sentiment keywords
positive_keywords = ["love", "amazing", "fast", "comfortable", "great", "excellent", "best"]
negative_keywords = ["terrible", "issues", "poor", "bad", "slow", "worst"]

for review in reviews:
    doc = nlp(review)
    
    # Named Entity Recognition (NER)
    print(f"\nReview: {review}")
    print("Named Entities:")
    found_entity = False
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            print(f" - {ent.text} ({ent.label_})")
            found_entity = True
    if not found_entity:
        print(" - None found")
    
    # Rule-based Sentiment
    review_lower = review.lower()
    sentiment = "Neutral"
    if any(word in review_lower for word in positive_keywords):
        sentiment = "Positive"
    elif any(word in review_lower for word in negative_keywords):
        sentiment = "Negative"
    
    print(f"Sentiment: {sentiment}")
