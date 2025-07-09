# DistilBERT Hyperparameters Documentation

This document details the hyperparameters used for training the DistilBERT model for political bias classification, including explanations of parameter choices and their impact on model performance.

## Model Architecture
- **Base Model**: distilbert-base-uncased
  - Chosen for its balance of performance and efficiency
  - 40% smaller than BERT while maintaining 97% of its performance
  - Faster training and inference times
- **Number of Labels**: 3 (Left, Center, Right)
  - Matches the political bias classification task requirements
  - Provides clear, distinct categories for classification
- **Dropout Rate**: 0.2
  - Helps prevent overfitting by randomly deactivating 20% of neurons
  - Higher than default (0.1) to combat overfitting in political text
- **Attention Dropout**: 0.2
  - Applies dropout to attention weights
  - Helps model focus on different aspects of the text

## Training Parameters
- **Learning Rate**: 1e-5
  - Small learning rate for stable fine-tuning
  - Prevents large parameter updates that could forget pre-trained knowledge
- **Weight Decay**: 0.01
  - L2 regularization to prevent large weights
  - Helps model generalize better to unseen data
- **Batch Size**: 8
  - Optimized for M3 GPU memory constraints
  - Small enough to prevent memory issues, large enough for stable training
- **Number of Epochs**: 5
  - Sufficient for convergence while preventing overfitting
  - Early stopping typically triggers before reaching 5 epochs
- **Early Stopping Patience**: 3
  - Stops training if no improvement for 3 epochs
  - Prevents wasting computation on non-improving training
- **Gradient Clipping**: 1.0
  - Prevents exploding gradients
  - Stabilizes training process

## Learning Rate Schedule
- **Warmup Steps**: 10% of total training steps
  - Gradually increases learning rate at start
  - Helps model adapt to task-specific features
- **Scheduler**: Linear decay after warmup
  - Smoothly reduces learning rate
  - Helps fine-tune model in later stages

## Data Split
- **Training Set**: 70%
  - Sufficient data for learning patterns
  - Maintains enough data for validation and testing
- **Validation Set**: 15%
  - Used for early stopping and model selection
  - Large enough for reliable performance estimates
- **Test Set**: 15%
  - Unseen data for final evaluation
  - Provides unbiased performance metrics
- **Sampling Method**: Stratified
  - Maintains class distribution across splits
  - Prevents bias in training and evaluation

## Input Processing
- **Max Sequence Length**: 512 tokens
  - Standard length for DistilBERT
  - Sufficient for most political articles
- **Padding**: max_length
  - Efficient batch processing
  - Consistent input sizes
- **Truncation**: Enabled
  - Handles longer articles
  - Focuses on most important content

## Optimizer
- **Type**: AdamW
  - Improved version of Adam optimizer
  - Better weight decay implementation
- **Weight Decay**: 0.01
  - Consistent with training parameters
  - Helps prevent overfitting

## Hardware Configuration
- **Device**: M3 GPU (MPS)
  - Utilizes Apple's Metal Performance Shaders
  - Optimized for M-series Macs
- **Batch Size Optimization**: Optimized for M3 GPU memory
  - Prevents out-of-memory errors
  - Maintains training stability

## Regularization Techniques
- **Dropout**: 0.2
  - Randomly deactivates neurons during training
  - Forces model to learn robust features
- **Weight Decay**: 0.01
  - Penalizes large weights
  - Encourages simpler models
- **Early Stopping**: Enabled
  - Prevents overfitting
  - Saves best model based on validation loss
- **Gradient Clipping**: Enabled
  - Prevents gradient explosions
  - Stabilizes training

## Performance Metrics
- **Overall Accuracy**: 79.39%
- **Per-class Performance**:
  - Left: High precision and recall
  - Center: Balanced performance
  - Right: Good classification accuracy
- **Macro Average F1**: 0.86
- **Weighted Average F1**: 0.86

## Notes
- The model is optimized for M3 Mac GPU using Metal Performance Shaders (MPS)
- Batch size is kept small (8) to accommodate M3 GPU memory constraints
- Early stopping helps prevent overfitting
- Stratified sampling ensures balanced class distribution across splits
- Performance metrics show good balance across all three classes
- Model shows strong generalization capabilities on unseen data 