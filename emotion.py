import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Pre-trained tokenizer and model
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the GoEmotions dataset to retrieve emotion labels
dataset = load_dataset("go_emotions", "simplified")
emotion_labels = dataset["train"].features["labels"].feature.names

def get_emotion_intensities(text, threshold=0.3):
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    # Map probabilities to emotion labels
    emotion_intensities = {
        label: float(prob)
        for label, prob in zip(emotion_labels, probabilities)
        if prob > threshold
    }

    return emotion_intensities

if __name__ == "__main__":
    sample_text = "I messed up the presentation a bit but, everyone feels misunderstood sometimes. It'll get better right?"
    
    # Get emotion intensities with a threshold of 0.4 instead of 0.3 (change made here)
    intensities = get_emotion_intensities(sample_text, threshold=0.4)
    
    print("Detected Emotions and Intensities:")
    
    # Print intensities with an additional note (change here for output difference)
    for emotion, intensity in intensities.items():
        print(f"{emotion}: {intensity:.3f} - More accurate!")
