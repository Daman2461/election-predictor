import re
import json
from stanfordcorenlp import StanfordCoreNLP

class SentimentAnalyzer:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=30000)
        self.props = {
            'annotators': 'tokenize,ssplit,parse,sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def split_text(self, text):
        # Split on punctuation (. ! ?) and clean spaces
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [sentence for sentence in sentences if sentence]

    def get_sentiments(self, text):
        sentences = self.split_text(text)
        all_sentiments = []

        for sent in sentences:
            output = json.loads(self.nlp.annotate(sent, properties=self.props))
            for sentence in output["sentences"]:
                sent_text = " ".join([token["word"] for token in sentence["tokens"]])
                sentiment = sentence["sentiment"]
                sentiment_distribution = sentence["sentimentDistribution"]
                all_sentiments.append((sent_text, sentiment, sentiment_distribution))
        
        return all_sentiments

    def average_sentiments(self, sentiments):
        avg_distribution = [0.0] * 5  # [Very Negative, Negative, Neutral, Positive, Very Positive]
        for _, _, distribution in sentiments:
            for i in range(5):
                avg_distribution[i] += distribution[i]
        num_sentences = len(sentiments)
        avg_distribution = [score / num_sentences for score in avg_distribution]
        return avg_distribution

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    text = """The weather was gloomy throughout the week. However, the weekend brought unexpected sunshine and warmth!  
People rushed to parks, beaches, and outdoor cafes.  
There was laughter in the air and children playing everywhere.  
Meanwhile, some businesses struggled to recover from the recent storms."""
    results = analyzer.get_sentiments(text)
    
    for i, (sent, senti, senti_dist) in enumerate(results, 1):
        print(f"Sentence {i}: {sent}")
        print(f"  Sentiment: {senti}")
        print(f"  Probabilities: {senti_dist}")

    # Show the average sentiment distribution
    avg_sentiment = analyzer.average_sentiments(results)
    classes = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    
    # Map classes to avg probs
    avg_sentiment_dict = {cls: prob for cls, prob in zip(classes, avg_sentiment)}
    
    # Sort by probability (highest first)
    sorted_avg_sentiment = sorted(avg_sentiment_dict.items(), key=lambda x: -x[1])

    print("\nAverage Sentiment Distribution (Sorted):")
    for cls, prob in sorted_avg_sentiment:
        print(f"{cls}: {prob:.4f}")