# AI Generated Text Detection

This repository contains the source code and resources for a binary classification project aimed at detecting AI-generated texts. The project is based on the [Kaggle competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) and utilizes a variety of classical machine learning models as well as a fine-tuned DistilRoBERTa model to achieve its goal.

## Project Structure

- **`data/`**: Contains pre-processed and post-processed training and test datasets in CSV format. Training datasets can be augmented with custom-generated synthetic data. The test set provided is a placeholder and should be replaced for actual use.
- **`model_checkpoints/`**: Stores trained models' checkpoints.
- **`EDA.ipynb`**: Jupyter notebook for exploratory data analysis on the training set.
- **`generate_synthetic_essays.ipynb`**: Notebook for generating synthetic training data using Mistral-7b instruct.
- **`data_processing.py`**: Processes the training and test sets, tokenizes and vectorizes texts, and saves the resulting sparse matrices as NPZ files in the `data/` folder.
- **`optuna.ipynb`**: Contains hyperparameter optimization for classical ML models (Ridge, Multinomial Naive Bayes, SVM, and XGBoost) and visualizations of optimization history and parameter importance.
- **`classical_models_training.py`**: Trains the four classical ML models and saves them as `.pkl` files in the `model_checkpoints/` folder.
- **`distilroberta_training.py`**: Fine-tunes the pre-trained DistilRoBERTa-base model on the training set and saves the checkpoint to the `model_checkpoints/` folder.
- **`inference.py`**: Loads trained classical ML models and DistilRoBERTa, ensembles them using weights to make predictions on the test set.
- **`pseudo_labeling.py`**: Implements advanced pseudo-labeling techniques to leverage accurate predictions for accuracy improvement.

## Setup

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/q-xZzz/ai-text-detection.git

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt

## Running the Project

To get the project up and running, follow these steps:

- **(optional)Exploratory Data Analysis**: Open `EDA.ipynb` with Jupyter Notebook or JupyterLab to explore the training dataset.
- **(optional)Generating Synthetic Training Data**: Use `generate_synthetic_essays.ipynb` to create additional synthetic data for training.
- **Data Processing**: Run `python data_processing.py` to tokenize and vectorize the datasets, and save the processed data for training and testing.
- **(optional)Hyperparameter Optimization**: Launch `optuna.ipynb` to find the optimal hyperparameters for the classical ML models.
- **Model Training**:
  - For classical ML models, execute `python classical_models_training.py`.
  - For DistilRoBERTa, run `python distilroberta_training.py`.
- **Inference**: Use `python inference.py` to load the trained models, ensemble them, and make predictions on the test set.
- **Pseudo Labeling**: Advanced pseudo-labeling techniques can be applied using `pseudo_labeling.py` to further refine the model's accuracy by leveraging confident predictions.
