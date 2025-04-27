import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====== Paths ======
MODEL_PATH = "/Users/daman/Downloads/Article-Bias-Prediction-main/output/final_model"

# ====== Load model and tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# ====== Predict function ======
def predict_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Manual mapping from model's LABEL_x to human-readable
    label_mapping = {
        0: "left",
        1: "center",
        2: "right"
    }
    
    label = label_mapping.get(predicted_class, "unknown")
    return label


# ====== Example usage ======
if __name__ == "__main__":
    article_text = """The reckless actions of conservative lawmakers continue to threaten the rights and freedoms of ordinary Americans. 
                    Their policies prioritize corporate greed over the well-being of working families, and their blatant disregard for climate change is endangering the planet for future generations. 
                    Progressive leadership is the only hope for restoring justice, equity, and human rights in a society increasingly corrupted by right-wing extremism.
                    """
    bias = predict_bias(article_text)
    print("\n🎯 Predicted Bias:", bias)
