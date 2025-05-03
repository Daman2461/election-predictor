import os
import json
import numpy as np
from collections import Counter

# --- Membership Functions ---
def fuzzify_emotion(avg_emotion_score):
    if avg_emotion_score < 0.2:
        return {'low': 1.0, 'medium': 0.0, 'high': 0.0}
    elif avg_emotion_score < 0.5:
        low = (0.5 - avg_emotion_score) / 0.3
        medium = (avg_emotion_score - 0.2) / 0.3
        return {'low': low, 'medium': medium, 'high': 0.0}
    elif avg_emotion_score < 0.8:
        medium = (0.8 - avg_emotion_score) / 0.3
        high = (avg_emotion_score - 0.5) / 0.3
        return {'low': 0.0, 'medium': medium, 'high': high}
    else:
        return {'low': 0.0, 'medium': 0.0, 'high': 1.0}

def fuzzify_sentiment(top_sentiment):
    mapping = {
        'very negative': 'negative',
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',
        'very positive': 'positive'
    }
    fuzzy_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    sentiment_level = mapping.get(top_sentiment.lower(), 'neutral')
    return fuzzy_map[sentiment_level]

def fuzzify_political(political):
    political = political.lower()
    return {'left': 0, 'center': 1, 'right': 2}.get(political, 1)

# --- Fuzzy Inference Engine ---
def fuzzy_decide(article):
    sentiment_data = article['sentiment_analysis']
    top_sentiment, _ = sentiment_data['sorted_average_sentiment'][0]
    sentiment_score = fuzzify_sentiment(top_sentiment)

    emotions = article['emotion_detection']['sorted_emotions']
    avg_emotion_score = np.mean([score for _, score in emotions])
    emotion_fuzzy = fuzzify_emotion(avg_emotion_score)

    political_raw = article['political_bias_prediction']['predicted_bias']
    political_score = fuzzify_political(political_raw)

    # Rule: Combine fuzzy weighted values
    weights = {
        'sentiment': 0.25,
        'emotion_high': 0.35,
        'political': 0.40
    }

    # Fuzzy rule logic
    # Defuzzify decision:
    # 0 = Left, 1 = Center, 2 = Right
    decision_value = (
        weights['sentiment'] * sentiment_score +
        weights['emotion_high'] * emotion_fuzzy['high'] * 2 +  # High emotion intensifies polarity
        weights['political'] * political_score
    )

    # Final decision based on defuzzified value
    if decision_value < 0.75:
        return 'left'
    elif decision_value < 1:
        return 'center'
    else:
        return 'right'

# --- Load JSON articles ---
def load_articles(folder_path):
    articles = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                articles.append(data)
    return articles

# --- Aggregate and Decide Winner ---
def predict_winner(articles):
    votes = []
    for article in articles:
        vote = fuzzy_decide(article)
        votes.append(vote)

    vote_counter = Counter(votes)
    print("Votes:", dict(vote_counter))

    if vote_counter:
        winner = vote_counter.most_common(1)[0][0]
        return winner.capitalize()
    else:
        return "No clear winner"

# --- Main Execution ---
if __name__ == "__main__":
    folder_path = 'combine'  # folder with JSON articles
    articles = load_articles(folder_path)

    if articles:
        winner = predict_winner(articles)
        print(f"🏆 Winner: {winner}")
    else:
        print("No articles found in the folder.")