import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import gc

class BiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Path is not a directory: {data_dir}")
    
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory: {data_dir}")
    
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    
    texts = []
    labels = []
    error_count = 0
    
    for filename in tqdm(json_files, desc="Loading JSON files"):
        try:
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
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {filename}: {str(e)}")
            error_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_count += 1
    
    if not texts:
        raise ValueError("No valid articles found in the JSON files")
    
    print(f"Successfully loaded {len(texts)} articles")
    print(f"Encountered {error_count} errors during loading")
    
    # Print class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} articles")
    
    return texts, labels

# Add M3 GPU detection and configuration
def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple M3 GPU (Metal)")
        # Get available memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"Available system memory: {memory.available / (1024**3):.2f} GB")
        except ImportError:
            print("Install psutil for memory information")
    else:
        device = torch.device("cpu")
        print("Metal GPU not available, using CPU")
    return device

def train_model():
    # Set up device with M3 GPU detection
    device = setup_device()

    try:
        # Load data
        data_dir = "/Users/daman/Downloads/Article-Bias-Prediction-main/data/jsons"
        print(f"Loading data from: {data_dir}")
        texts, labels = load_data(data_dir)
        
        # Split data into train, validation, and test sets
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        print(f"\nDataset split sizes:")
        print(f"Training set: {len(train_texts)} articles")
        print(f"Validation set: {len(val_texts)} articles")
        print(f"Test set: {len(test_texts)} articles")

        # Initialize tokenizer and model
        model_name = "albert-base-v2"  # Changed to ALBERT which is very memory efficient
        print(f"\nLoading model: {model_name}")
        
        # Clear memory before loading model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load model with M3 optimizations
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        ).to(device)
        
        # Update dropout settings for better training
        model.config.dropout = 0.1  # ALBERT uses less dropout
        model.config.attention_dropout = 0.1
        model.config.hidden_dropout = 0.1

        # Create datasets
        print("\nCreating datasets...")
        train_dataset = BiasDataset(train_texts, train_labels, tokenizer)
        val_dataset = BiasDataset(val_texts, val_labels, tokenizer)
        test_dataset = BiasDataset(test_texts, test_labels, tokenizer)
        print("Datasets created successfully")

        # Create datasets with batch size optimized for ALBERT
        gradient_accumulation_steps = 2  # ALBERT can handle smaller accumulation
        effective_batch_size = 32  # ALBERT is very memory efficient
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=2
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,
            num_workers=2
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            num_workers=2
        )

        # Set up optimizer with learning rate for ALBERT
        optimizer = AdamW(
            model.parameters(),
            lr=1e-5,  # Standard learning rate for ALBERT
            weight_decay=0.01
        )
        
        num_epochs = 3
        num_training_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps
        )

        # Training loop
        best_val_loss = float('inf')
        patience = 2
        patience_counter = 0
        
        try:
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                
                # Training
                model.train()
                total_train_loss = 0
                optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                    # Clear memory periodically
                    if batch_idx % 50 == 0:  # More frequent memory clearing
                        gc.collect()
                        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss / gradient_accumulation_steps
                    total_train_loss += loss.item() * gradient_accumulation_steps

                    loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

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
                        predictions = torch.argmax(logits, dim=-1)
                        correct_predictions += (predictions == labels).sum().item()
                        total_predictions += labels.size(0)

                avg_val_loss = total_val_loss / len(val_dataloader)
                val_accuracy = correct_predictions / total_predictions
                print(f"Validation loss: {avg_val_loss:.4f}")
                print(f"Validation accuracy: {val_accuracy:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save the best model to output directory
                    output_dir = "/Users/daman/Downloads/Article-Bias-Prediction-main/output/albert_model"
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    print(f"Saved best model to {output_dir}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break

        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            raise

    except Exception as e:
        print(f"An error occurred during training setup: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()