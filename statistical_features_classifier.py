import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import joblib

# Load the QLoRA model and tokenizer
MODEL_PATH = "/path/to/base/model"
ADAPTER_PATH = "/path/to/qlora/adapter"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Function to make predictions using the QLoRA model
def predict_qlora(text, max_length=1024):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    return probs[0, 1].item()  # Return probability of positive class

# Function to extract statistical features
def extract_features(probabilities):
    if len(probabilities) == 1:
        return [1, probabilities[0], probabilities[0], probabilities[0], probabilities[0], 0, 0]
    else:
        return [
            len(probabilities),
            min(probabilities),
            max(probabilities),
            np.mean(probabilities),
            np.median(probabilities),
            np.std(probabilities),
            np.percentile(probabilities, 75) - np.percentile(probabilities, 25)  # IQR
        ]

# Load and preprocess the synthetic data
df = pd.read_csv("/path/to/synthetic_solutions.csv")

# Group solutions by problem and predicted answer
grouped_solutions = defaultdict(lambda: defaultdict(list))
for _, row in df.iterrows():
    prob = predict_qlora(row['solution'])
    grouped_solutions[row['problem']][row['answer']].append(prob)

# Extract features for each group
features = []
labels = []
for problem, answers in grouped_solutions.items():
    for answer, probs in answers.items():
        features.append(extract_features(probs))
        labels.append(1 if answer == df[df['problem'] == problem]['correct_answer'].iloc[0] else 0)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

print(f"Train AUC: {train_auc}")
print(f"Test AUC: {test_auc}")

# Save the model
joblib.dump(clf, 'logistic_regression_classifier.joblib')

print("Logistic Regression Classifier saved.")