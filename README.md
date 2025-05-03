# Election Predictor

This repository contains a collection of Python scripts for analyzing various aspects of text, including emotions, sentiments, and more.

## Available Scripts

1. `emotion.py`: Uses BERT-based model for emotion detection
2. `stanfordnlp.py`: Uses Stanford CoreNLP for sentiment analysis
3. `fuzzy.py`: Implements fuzzy logic for text analysis
4. `ollama.py`: Integration with Ollama for text processing
5. `app.py`: Main application interface
6. `run_all_models.py`: Script to run all analysis models

## Requirements

### Core Dependencies
- Python 3.6+
- PyTorch
- Transformers library
- Datasets library
- Stanford CoreNLP
- Ollama (for ollama.py)

Install dependencies:
```bash
pip install -r requirements.txt
```

### Additional Setup

#### Stanford CoreNLP Setup
1. Download Stanford CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/download.html)
2. Start the CoreNLP server:
```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

#### Ollama Setup
1. Install Ollama from [here](https://ollama.ai/)
2. Start the Ollama service

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

### Running All Models (run_all_models.py)

To run all analysis models on a text file:

```bash
python run_all_models.py input_article.txt
```

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

## Project Structure

- `data/`: Directory for input data files
- `output/`: Directory for analysis results
- `combine/`: Directory for combined analysis outputs
- `input_article.txt`: Sample input text file

## Notes

- The emotion analysis model (`emotion.py`) uses a threshold of 0.3 by default to filter out low-confidence emotions
- The sentiment analysis (`stanfordnlp.py`) requires a running Stanford CoreNLP server on port 9000
- Both scripts can process multiple sentences and return detailed analysis results
- The fuzzy logic implementation (`fuzzy.py`) provides additional text analysis capabilities
- The Ollama integration (`ollama.py`) offers alternative text processing options

## License

This project is open source and available under the MIT License. 
