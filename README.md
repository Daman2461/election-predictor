# Emotion and Sentiment Analysis Tools

This repository contains two Python scripts for analyzing emotions and sentiments in text:

1. `emotion.py`: Uses BERT-based model for emotion detection
2. `stanfordnlp.py`: Uses Stanford CoreNLP for sentiment analysis

## Requirements

### For emotion.py
- Python 3.6+
- PyTorch
- Transformers library
- Datasets library

Install dependencies:
```bash
pip install torch transformers datasets
```

### For stanfordnlp.py
- Python 3.6+
- Stanford CoreNLP server running locally
- stanfordcorenlp Python package

Install dependencies:
```bash
pip install stanfordcorenlp
```

## Setup

### Stanford CoreNLP Setup
1. Download Stanford CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/download.html)
2. Start the CoreNLP server:
```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## Usage

### Emotion Analysis (emotion.py)

This script uses a pre-trained BERT model fine-tuned on the GoEmotions dataset to detect emotions in text.

```python
from emotion import get_emotion_intensities

text = "I am thrilled with the recent developments in the job!"
emotions = get_emotion_intensities(text, threshold=0.3)
print(emotions)
```

The function returns a dictionary of emotions and their intensities (probabilities) that exceed the specified threshold.

### Sentiment Analysis (stanfordnlp.py)

This script uses Stanford CoreNLP to perform sentiment analysis on text.

```python
from stanfordnlp import SentimentAnalyzer

analyzer = SentimentAnalyzer()
text = "The economy showed signs of recovery. The president addressed the nation with optimism."
sentiments = analyzer.get_sentiments(text)
print(sentiments)
```

The function returns a list of tuples containing sentences and their sentiment scores.

## Example Outputs

### Emotion Analysis
```python
{
    'excitement': 0.85,
    'joy': 0.72,
    'optimism': 0.65
}
```

### Sentiment Analysis
```python
[
    ("The economy showed signs of recovery.", "Positive"),
    ("The president addressed the nation with optimism.", "Positive")
]
```

## Notes

- The emotion analysis model (`emotion.py`) uses a threshold of 0.3 by default to filter out low-confidence emotions
- The sentiment analysis (`stanfordnlp.py`) requires a running Stanford CoreNLP server on port 9000
- Both scripts can process multiple sentences and return detailed analysis results

## License

This project is open source and available under the MIT License. 