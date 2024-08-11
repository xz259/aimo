import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import vstack

train = pd.read_csv('/data/train_essays.csv')
test = pd.read_csv('/data/test_essays.csv')
Y_train = train['label'].values

from scipy.sparse import load_npz

X_train = load_npz('/data/processed_train.npz')
X_test = load_npz('/data/processed_test.npz')

from scipy.special import expit as sigmoid

from inference import final_preds

def pseudo_label_1(preds, iterations, pct):

    current_X_train = X_train.copy() 
    current_Y_train = Y_train.copy() 

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")

        num_samples = len(preds)
        sorted_indices = np.argsort(preds)


        top_indices = sorted_indices[-int(num_samples * pct):]
        bottom_indices = sorted_indices[:int(num_samples * pct)]


        added_indices = set()

        top_indices = [idx for idx in top_indices if idx not in added_indices]
        bottom_indices = [idx for idx in bottom_indices if idx not in added_indices]

        added_indices.update(top_indices)
        added_indices.update(bottom_indices)


        top_vectors = tf_test[top_indices]
        bottom_vectors = tf_test[bottom_indices]
        top_labels = np.ones(len(top_indices))
        bottom_labels = np.zeros(len(bottom_indices))

        current_X_train = vstack([current_X_train, top_vectors, bottom_vectors])
        current_Y_train = np.concatenate([current_Y_train, top_labels, bottom_labels])

        ridge = Ridge(solver='sag', max_iter=10000, tol=1e-4, alpha=1)
        ridge.fit(current_X_train, current_Y_train)

        ridge_preds = ridge.predict(tf_test)    
        ridge_probs = sigmoid(ridge_preds)
        current_probs = 0.9*current_probs + 0.1*ridge_probs
        
    final_preds = current_probs 
    return final_preds


import torch
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tokenizer = AutoTokenizer.from_pretrained('distilroberta-base"')
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base"', num_labels=num_labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['text'], max_length=512, padding=True, truncation=True)
    return tokenized_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits[:, 1])
    auc = roc_auc_score(labels, probs)
    return {"roc_auc": auc}


def pseudo_label_2(preds, epochs, pct):
    probs_array = np.array(preds)
    num_samples = len(probs_array)
    sorted_indices = np.argsort(probs_array)
    num_top_bottom = int(num_samples * pct)

    top_indices = sorted_indices[-num_top_bottom:]  
    bottom_indices = sorted_indices[:num_top_bottom] 


    top_labels = np.ones(len(top_indices))
    bottom_labels = np.zeros(len(bottom_indices))


    test_texts = test['text'].tolist()
    top_texts = [test_texts[i] for i in top_indices]
    bottom_texts = [test_texts[i] for i in bottom_indices]

    temp_texts = top_texts + bottom_texts
    temp_labels = np.concatenate([top_labels, bottom_labels])
    temp_labels = temp_labels.astype(int)


    temp_data = {'text': temp_texts, 'label': temp_labels}
    temp_df = pd.DataFrame(temp_data)


    augmented_df = pd.concat([train, temp_df]).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(augmented_df['text'], augmented_df['label']):
        train_df = augmented_df.iloc[train_index]
        val_df = augmented_df.iloc[val_index]
        break
        

    val_ds = Dataset.from_pandas(val_df)
    val_ds_enc = val_ds.map(preprocess_function, batched=True)

    train_ds = Dataset.from_pandas(train_df)
    train_ds_enc = train_ds.map(preprocess_function, batched=True)

    from transformers import EarlyStoppingCallback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    num_train_epochs = epochs
    metric_name = "roc_auc"
    model_name = "distilroberta"
    batch_size = 2

    args = TrainingArguments(
        f"{model_name}-finetuned_v5",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        lr_scheduler_type = "cosine",
        save_safetensors = False,
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        report_to='none', 
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds_enc,
        eval_dataset=val_ds_enc,  
        tokenizer=tokenizer,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
    )

    test_ds = Dataset.from_pandas(test)
    test_ds_enc = test_ds.map(preprocess_function, batched=True)
    test_preds = trainer.predict(test_ds_enc)
    test_logits = test_preds.predictions
    probs = sigmoid(test_logits[:,1])

    return 0.8*preds + 0.2*probs


final_preds = pseudo_label_1(final_preds, 5, 0.2)
final_preds = pseudo_label_2(final_preds, 4, 0.2)
