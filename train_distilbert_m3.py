import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
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

def load_data(data_dir):
    texts = []
    labels = []
    
    # Load all JSON files from the directory
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

def train_model():
    # Set up device for M3 Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    else:

        
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Load data
    data_dir = "/Users/daman/Downloads/Article-Bias-Prediction-main/data/jsons"
    texts, labels = load_data(data_dir)
    
    # Split data into train, validation, and test sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    config = DistilBertConfig.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3,  # 0: left, 1: center, 2: right
        dropout=0.2,  # Increased dropout
        attention_dropout=0.2  # Increased attention dropout
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        config=config
    )
    model.to(device)

    # Create datasets
    train_dataset = BiasDataset(train_texts, train_labels, tokenizer)
    val_dataset = BiasDataset(val_texts, val_labels, tokenizer)
    test_dataset = BiasDataset(test_texts, test_labels, tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,  # Smaller batch size for M3 GPU
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8
    )

    # Set up optimizer with weight decay
    optimizer = AdamW(model.parameters(), 
                     lr=1e-5,  # Reduced learning rate
                     weight_decay=0.01)  # Added weight decay
    
    num_epochs = 5  # Increased epochs
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps * 0.1,  # 10% warmup
        num_training_steps=num_training_steps
    )

    # Training loop
    best_val_loss = float('inf')
    patience = 3  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_val_loss += loss.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += len(labels)

        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct_predictions / total_predictions
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            output_dir = "/Users/daman/Downloads/Article-Bias-Prediction-main/output/model2"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved best model to {output_dir}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            test_correct += (predictions == labels).sum().item()
            test_total += len(labels)

    test_accuracy = test_correct / test_total
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    train_model() 