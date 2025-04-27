import re
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stanfordcorenlp import StanfordCoreNLP

# ---------- Sentiment Analyzer ----------
class SentimentAnalyzer:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=30000)
        self.props = {
            'annotators': 'tokenize,ssplit,parse,sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def split_text(self, text):
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
        avg_distribution = [0.0] * 5
        for _, _, distribution in sentiments:
            for i in range(5):
                avg_distribution[i] += distribution[i]
        num_sentences = len(sentiments)
        avg_distribution = [score / num_sentences for score in avg_distribution]
        return avg_distribution

# ---------- Emotion Analyzer ----------
class EmotionAnalyzer:
    def __init__(self):
        model_name = "monologg/bert-base-cased-goemotions-original"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        dataset = load_dataset("go_emotions", "simplified")
        self.emotion_labels = dataset["train"].features["labels"].feature.names

    def get_all_emotion_probs_sorted(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        emotion_intensities = {
            label: float(prob)
            for label, prob in zip(self.emotion_labels, probabilities)
        }
        sorted_emotions = sorted(emotion_intensities.items(), key=lambda x: -x[1])
        return sorted_emotions

# ---------- Political Polarization Analyzer ----------
class PolarizationAnalyzer:
    def __init__(self):
        model_path = "/Users/daman/Downloads/Article-Bias-Prediction-main/output/final_model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict_bias(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        label_mapping = {0: "left", 1: "center", 2: "right"}
        label = label_mapping.get(predicted_class, "unknown")
        return label

# ---------- Main ----------
if __name__ == "__main__":

    # ======= YOUR INPUT TEXT HERE =======
    text = """he recent tax cuts have sparked heated debate across the nation. While many on the right believe that reducing taxes will stimulate economic growth, critics argue that it primarily benefits the wealthy and increases income inequality. Democratic leaders have called for higher taxes on the rich to fund social programs. Despite the partisan rhetoric, the debate continues to evolve as more data comes in"""
    print("📋 Input Text:")
    print(text)
    print("\n")

    # ----- Sentiment -----
    sentiment_analyzer = SentimentAnalyzer()
    sentiments = sentiment_analyzer.get_sentiments(text)
    avg_sentiment = sentiment_analyzer.average_sentiments(sentiments)
    classes = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    avg_sentiment_dict = {cls: prob for cls, prob in zip(classes, avg_sentiment)}
    sorted_avg_sentiment = sorted(avg_sentiment_dict.items(), key=lambda x: -x[1])

    print("🧠 Sentiment Analysis:")
    for cls, prob in sorted_avg_sentiment:
        print(f"{cls}: {prob:.4f}")
    print("\n")

    # ----- Emotions -----
    emotion_analyzer = EmotionAnalyzer()
    sorted_emotions = emotion_analyzer.get_all_emotion_probs_sorted(text)

    print("💬 Emotion Detection:")
    for emotion, intensity in sorted_emotions[:10]:  # Top 10 emotions
        print(f"{emotion}: {intensity:.4f}")
    print("\n")

    # ----- Political Bias -----
    polarization_analyzer = PolarizationAnalyzer()
    bias = polarization_analyzer.predict_bias(text)

    print("🏛️ Political Bias Prediction:")
    print(f"Predicted Bias: {bias}")
    output_data = {
        "sentiment_analysis": {
            "average_sentiment": avg_sentiment_dict,
            "sorted_average_sentiment": sorted_avg_sentiment
        },
        "emotion_detection": {
            "sorted_emotions": sorted_emotions[:10]  # Top 10 emotions
        },
        "political_bias_prediction": {
            "predicted_bias": bias
        }
    }

    with open("analysis_results.json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)