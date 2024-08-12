This repository contains the source code and resources for an LLM-based mathematical problem solver. It combines the generative capabilities of large language models with the discriminative power of classical machine learning techniques.

## Project Structure

- **`inference.py`**: Contains the main inference pipeline for solving mathematical problems using the two-stage approach.
- **`qlora_training.py`**: Script for fine-tuning the base language model using QLoRA (Quantized Low-Rank Adaptation).
- **`statistical_features_classifier.py`**: Trains the second-stage logistic regression classifier using aggregated statistics from the QLoRA model's outputs.
- **`model/`**: Directory containing the base language model and fine-tuned QLoRA adapters.
- **`data/`**: Contains training, validation, and test datasets.

## Setup

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/AIMO.git
   cd AIMO

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt

3. **Download the Base Model**:
   ```sh
   Download the base language model (e.g., DeepSeek-Math-7B-RL) and place it in the model/ directory.

## Running the Project

1. **Fine-tune the QLoRA Model**:
   ```sh
   python qlora_training.py
2. **Train the statistical features classifier**:
   ```sh
   python aggregated_statistics_classifier.py
3. **Run inference**:
   ```sh
   python inference.py
