import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class BiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_dir):
    texts = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'content' in item and 'bias' in item:
                            texts.append(item['content'])
                            labels.append(item['bias'])
                elif isinstance(data, dict):
                    if 'content' in data and 'bias' in data:
                        texts.append(data['content'])
                        labels.append(data['bias'])
    
    return texts, labels

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('roberta_confusion_matrix.png')
    plt.close()

def evaluate_model():
    # Set up device for M3 Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Load model and tokenizer
    model_path = "/Users/daman/Downloads/Article-Bias-Prediction-main/output/roberta_model"
    
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load test data
    data_dir = "/Users/daman/Downloads/Article-Bias-Prediction-main/data/jsons"
    texts, labels = load_data(data_dir)
    
    # Create dataset and dataloader
    dataset = BiasDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Get predictions
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # Calculate and print metrics
    print("\nDetailed Classification Report:")
    report = classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=['Left', 'Center', 'Right'],
        digits=4,
        output_dict=True  # Get report as dictionary
    )
    
    # Print per-class metrics
    print("\nPer-class Metrics:")
    for label in ['Left', 'Center', 'Right']:
        print(f"\n{label}:")
        print(f"Precision: {report[label]['precision']:.4f}")
        print(f"Recall: {report[label]['recall']:.4f}")
        print(f"F1-score: {report[label]['f1-score']:.4f}")
        print(f"Support: {report[label]['support']}")

    # Print overall model metrics
    print("\nOverall Model Metrics:")
    print(f"Macro Average Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Average Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Average F1-score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Average Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Weighted Average Recall: {report['weighted avg']['recall']:.4f}")
    print(f"Weighted Average F1-score: {report['weighted avg']['f1-score']:.4f}")
    print(f"Accuracy: {report['accuracy']:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    print("\nConfusion Matrix:")
    print("True Label (rows) vs Predicted Label (columns)")
    print("              Pred Left  Pred Center  Pred Right")
    print(f"True Left     {cm[0][0]:^10} {cm[0][1]:^12} {cm[0][2]:^11}")
    print(f"True Center   {cm[1][0]:^10} {cm[1][1]:^12} {cm[1][2]:^11}")
    print(f"True Right    {cm[2][0]:^10} {cm[2][1]:^12} {cm[2][2]:^11}")

    # Calculate per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(class_accuracies):
        class_name = ['Left', 'Center', 'Right'][i]
        print(f"{class_name}: {acc:.4f}")

    # Print some example predictions
    print("\nExample Predictions:")
    for i in range(min(5, len(texts))):
        print(f"\nText: {texts[i][:150]}...")
        print(f"True: {['Left', 'Center', 'Right'][all_true_labels[i]]}")
        print(f"Predicted: {['Left', 'Center', 'Right'][all_predictions[i]]}")

if __name__ == "__main__":
    evaluate_model()