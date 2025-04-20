from stanfordcorenlp import StanfordCoreNLP
import json

class SentimentAnalyzer:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=30000)
        self.props = {
            'annotators': 'tokenize,ssplit,parse,sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def get_sentiments(self, text):
        output = json.loads(self.nlp.annotate(text, properties=self.props))
        sentiments = []
        for sentence in output["sentences"]:
            sent_text = " ".join([token["word"] for token in sentence["tokens"]])
            sentiment = sentence["sentiment"]
            sentiments.append((sent_text, sentiment))
        return sentiments

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    text = """The economy showed signs of recovery. 
    The president addressed the nation with optimism. 
    Reforms were discussed without much debate. 
    The healthcare system continues to face severe challenges. 
    Voters remain divided over key policies. 
    The media coverage has been consistent."""
    
    results = analyzer.get_sentiments(text)
    for i, (sent, senti) in enumerate(results, 1):
        print(f"Sentence {i}: {senti}")