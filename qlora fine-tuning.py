import os
import re
import sys
import gc
import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, set_seed
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import expit as sigmoid
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer

# Constants
MODEL_PATH = "/path/to/model"
DEBUG = False

# Set up CUDA and random seed
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cudnn.benchmark = True

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# Load and prepare data
def load_data(train_path, val_path, sample_size=None):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    if sample_size:
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size)
    
    train_df["Match"] = train_df["Match"].astype(int)
    val_df["Match"] = val_df["Match"].astype(int)
    
    train_df = train_df.rename(columns={"Match": "label", "Solution": "text"})
    val_df = val_df.rename(columns={"Match": "label", "Solution": "text"})
    
    cols = ["text", "label"]
    return train_df[cols], val_df[cols]

train_df, val_df = load_data("/path/to/train.csv", "/path/to/val.csv", sample_size=500 if DEBUG else None)

# Prepare datasets
def prepare_datasets(train_df, val_df):
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    
    def preprocess_function(examples, max_length=2048):
        tokenized_inputs = tokenizer(examples['text'], max_length=max_length, truncation=True)
        tokenized_inputs['labels'] = examples['label']
        return tokenized_inputs
    
    train_tokenized_ds = train_ds.map(preprocess_function, batched=True)
    val_tokenized_ds = val_ds.map(preprocess_function, batched=True)
    
    return train_tokenized_ds, val_tokenized_ds

# Model setup
config = AutoConfig.from_pretrained(MODEL_PATH)
config.gradient_checkpointing = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    device_map="auto",
    quantization_config=quantization_config,
)

base_model.config.pretraining_tp = 1
base_model.config.pad_token_id = tokenizer.pad_token_id

# Setup LoRA configuration
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']

peft_config = LoraConfig(
    r=256,
    lora_alpha=128,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    target_modules=target_modules,
    rank_pattern={'gate_proj': 256, 'down_proj': 128, 'up_proj': 128}
)

# Get PEFT model
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# Prepare datasets
train_tokenized_ds, val_tokenized_ds = prepare_datasets(train_df, val_df)

# Evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = sigmoid(logits[:, 1])
    auc = roc_auc_score(labels, preds)
    return {"roc_auc": auc}

# Training arguments
training_args = TrainingArguments(
    output_dir="/path/to/output",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    max_grad_norm=0.5,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    weight_decay=0.2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    warmup_steps=30,
    logging_steps=500,
    metric_for_best_model="roc_auc",
    save_total_limit=2,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_ds,
    eval_dataset=val_tokenized_ds,
    tokenizer=tokenizer,
    data_collator=None,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Training
def train():
    trainer.train()

if __name__ == "__main__":
    train()