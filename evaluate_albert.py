import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import json
import os
from train_albert_m3 import BiasDataset, setup_device

def load_test_data(data_dir):
    """Load test data from JSON files"""
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    texts = []
    labels = []
    
    for filename in tqdm(json_files, desc="Loading test data"):
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
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return texts, labels

def evaluate_model():
    # Set up device
    device = setup_device()
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_path = "/Users/daman/Downloads/Article-Bias-Prediction-main/output/albert_model"
    print(f"\nLoading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Load test data
    data_dir = "/Users/daman/Downloads/Article-Bias-Prediction-main/data/jsons"
    print(f"Loading test data from: {data_dir}")
    test_texts, test_labels = load_test_data(data_dir)
    
    # Create test dataset and dataloader
    test_dataset = BiasDataset(test_texts, test_labels, tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=2
    )

    # Evaluation
    print("\nStarting evaluation...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    print("\nEvaluation Results:")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print(conf_matrix)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'classification_report': classification_report(all_labels, all_predictions, output_dict=True),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    output_file = "/Users/daman/Downloads/Article-Bias-Prediction-main/output/albert_model/evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    evaluate_model() 