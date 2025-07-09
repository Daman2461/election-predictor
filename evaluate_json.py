import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def load_random_sample(data_dir, sample_size=1000):
    all_texts = []
    all_labels = []
    
    # Load all JSON files from the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'content' in item and 'bias' in item:
                            all_texts.append(item['content'])
                            all_labels.append(item['bias'])
                elif isinstance(data, dict):
                    if 'content' in data and 'bias' in data:
                        all_texts.append(data['content'])
                        all_labels.append(data['bias'])
    
    # Randomly sample if we have more than sample_size
    if len(all_texts) > sample_size:
        indices = random.sample(range(len(all_texts)), sample_size)
        texts = [all_texts[i] for i in indices]
        labels = [all_labels[i] for i in indices]
    else:
        texts = all_texts
        labels = all_labels
    
    return texts, labels

def evaluate_model():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference")

    # Load model and tokenizer
    model_path = "/Users/daman/Downloads/Article-Bias-Prediction-main/output/model2"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load random sample of data
    data_dir = "/Users/daman/Downloads/Article-Bias-Prediction-main/data/jsons"
    texts, true_labels = load_random_sample(data_dir, sample_size=1000)
    
    # Create dataset and dataloader
    dataset = BiasDataset(texts, true_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

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

    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=['Left', 'Center', 'Right']))

    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_df = pd.DataFrame(cm, 
                        index=['True Left', 'True Center', 'True Right'],
                        columns=['Pred Left', 'Pred Center', 'Pred Right'])
    
    print("\nConfusion Matrix:")
    print(cm_df)

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
        print(f"True: {['Left', 'Center', 'Right'][true_labels[i]]}")
        print(f"Predicted: {['Left', 'Center', 'Right'][all_predictions[i]]}")

if __name__ == "__main__":
    evaluate_model() 