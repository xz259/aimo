import pandas as pd
import torch
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("/data/train_essays.csv")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_index, val_index in skf.split(train['text'], train['label']):
    train_df = augmented_df.iloc[train_index]
    val_df = augmented_df.iloc[val_index]
    break


tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=num_labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['text'], max_length=512, padding=True, truncation=True)
    return tokenized_inputs

val_ds = Dataset.from_pandas(val_df)
val_ds_enc = val_ds.map(preprocess_function, batched=True)

train_ds = Dataset.from_pandas(train_df)
train_ds_enc = train_ds.map(preprocess_function, batched=True)




def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits[:, 1])
    auc = roc_auc_score(labels, probs)
    return {"roc_auc": auc}

from transformers import EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

num_train_epochs = 5.0
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

import wandb
wandb.init(mode="disabled")

trainer.train()

trainer.save_model("/model_checkpoints/distilroberta/")
