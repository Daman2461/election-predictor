import os
import json
import numpy as np
from collections import Counter

# Step 1: Fuzzy rules per article
# New emotion groups
positive_emotions = {'amusement', 'approval', 'pride', 'optimism', 'joy', 'gratitude', 'relief', 'love'}
negative_emotions = {'sadness', 'anger', 'disappointment', 'fear', 'disgust', 'embarrassment', 'grief', 'nervousness', 'remorse'}
neutral_emotions = {'realization', 'confusion', 'surprise', 'curiosity', 'admiration', 'excitement', 'desire', 'caring', 'neutral'}

def fuzzy_decide(article):
    sentiment = article['sentiment_analysis']['average_sentiment']
    sorted_sentiment = article['sentiment_analysis']['sorted_average_sentiment']
    top_sentiment, top_sentiment_value = sorted_sentiment[0]

    emotions = article['emotion_detection']['sorted_emotions']
    top_emotion, top_emotion_score = emotions[0]

    # Determine emotion polarity
    if top_emotion in positive_emotions:
        emotion_polarity = 'positive'
    elif top_emotion in negative_emotions:
        emotion_polarity = 'negative'
    else:
        emotion_polarity = 'neutral'

    political = article['political_bias_prediction']['predicted_bias'].lower()

    # New Fuzzy Rules
    if top_sentiment in ['Negative', 'Very Negative'] and emotion_polarity == 'negative' and political in ['left', 'right']:
        return political
    if top_sentiment in ['Positive', 'Very Positive'] and emotion_polarity == 'positive' and political in ['left', 'right']:
        return political
    if top_sentiment == 'Neutral' and emotion_polarity == 'neutral':
        return 'center'
    if top_sentiment in ['Positive', 'Very Positive'] and emotion_polarity == 'positive' and political == 'center':
        return 'center'
    if top_sentiment == 'Neutral' and emotion_polarity == 'negative' and political in ['left', 'right']:
        return political

    # Default: fallback to base prediction
    return political


# Step 2: Load JSONs from folder
def load_articles(folder_path):
    articles = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                articles.append(data)
    return articles

# Step 3: Predict overall winner
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

# ============================
# Main

if __name__ == "__main__":
    folder_path = 'combine'  # your folder name
    articles = load_articles(folder_path)
    
    if articles:
        winner = predict_winner(articles)
        print(f"🏆 Winner: {winner}")
    else:
        print("No articles found in the folder.")
