import asyncio
import nest_asyncio
import time
import pandas as pd
import numpy as np
import os
import re
import sys
import gc
import subprocess
import math
import random
import sympy
from tqdm import tqdm
from collections import defaultdict, Counter
import traceback
import joblib
import pickle
from sklearn.linear_model import LogisticRegression
import torch
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from vllm.utils import random_uuid
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, AutoTokenizer
from peft import PeftModelForSequenceClassification

# Define constants
MODEL_PATH = "/path/to/model"
N_REPETITIONS = 64
ALLOCATED_TIME = 600  # 10 minutes

# Set up CUDA settings
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cudnn.benchmark = True

# Function to clear GPU memory
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.2)

# Set random seed for reproducibility
def seed_everything(seed=None):
    if seed is None:
        seed = random.randint(0, 9999)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Using seed: {seed}")
    return seed

# Define ScoreHead class for model output
class ScoreHead(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, device):
        super().__init__()
        self.out_proj = torch.nn.Linear(hidden_size, num_labels, dtype=torch.bfloat16, device=device)

    def forward(self, x):
        return self.out_proj(x)

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path)
        weight = state_dict['weight'].to(torch.bfloat16)
        self.out_proj.weight.data.copy_(weight)

# Function to predict score using the model
def score_predict(input_text, model, tokenizer, score_head, max_length=1024):
    # Tokenize input
    tokenized_text = tokenizer(
        input_text, 
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(model.device)
    
    attention_mask = tokenized_text["attention_mask"]
    
    # Get model output
    with torch.no_grad():
        outputs = model(**tokenized_text, output_hidden_states=True)
    
    # Process hidden states
    hidden_states = outputs.hidden_states[-1]
    last_token_position = torch.sum(attention_mask, dim=1) - 1
    pooled_output = hidden_states[torch.arange(hidden_states.size(0)), last_token_position]
    
    del outputs, hidden_states
    clean_memory()
    
    # Get prediction
    logits = score_head(pooled_output)
    pred = torch.sigmoid(logits[:, 1])
    return pred.item()

# Function to get best prediction from classifier
def predict_best(features):
    predictions = AIMO_clf.predict_proba(features)
    return predictions[0][1]

# Extract Python code from text
def extract_code(text):
    pattern = re.compile(r"```python\n([\s\S]*?)```")
    matches = pattern.findall(text)
    code = matches[-1]
    if matches:
        return code, True
    else:
        return "", False

# Process and execute extracted code
def process_code(code):
    # Replace symbols function calls
    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
    code = re.sub(r"symbols\([^)]+\)", repl, code)
    
    # Write code to file
    with open('code.py', 'w') as fout:
        fout.write(code)

    # Execute code with timeout
    batcmd = 'timeout 3 ' + sys.executable + ' code.py'
    try:
        shell_output = subprocess.check_output(
            batcmd, shell=True, stderr=subprocess.STDOUT, text=True
        )
        try:
            code_output = float(eval(shell_output))
        except Exception as e:
            print(e)
            traceback_str = traceback.format_exc()
            print("An error occurred. Here is the traceback:")
            print(traceback_str)
            code_output = "The result must be a number but the code returned:\n" + shell_output.strip()
        result = "SUCCESS"
    except Exception as e:
        print("Command failed with return code", e.returncode)
        print("Error output:", e.output)
        code_output = e.output
        result = "FAIL"
        if "not defined" in code_output:
            code_output +='\\nCheck the formatting and imports.'
    return code_output, result

# Parse answer from text
def naive_parse(text):
    out = []
    start = False
    end = False
    
    answer_index = text.rfind("answer is")

    if answer_index != -1:
        text = text[answer_index:]

        for s in list(text):
            if s in '0123456789' and not end:
                start = True
                out.append(s)
            elif start:
                end = True
    return ''.join(out)

# Check if generation is complete and extract answer
def check_finish_generation(text):
    try:
        match = re.findall(r'boxed\{(\d+)\}', text)
        if not len(match):
            answer = naive_parse(text)
        else:
            answer = match[-1]
        answer = float(eval(answer))
        print("TEXT ANSWER", answer)
        return answer, True
    except:
        pass
    return -1, False

# Validate if answer is acceptable
def valid_answer(num):
    if isinstance(num, (int, float)):
        if isinstance(num, float) and not num.is_integer():
            return False
        return 0 <= num 
    return False

# Main function to solve a problem
async def solve_problem(problem, PROBLEM_START_TIME, EXTRA_TIME):
    # ... (rest of the function remains the same)

# Main execution function
async def main():
    # Set up engine arguments
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        dtype="half",
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        gpu_memory_utilization=0.65,
        swap_space=4,
        max_model_len=2048,
        kv_cache_dtype="auto",
        disable_custom_all_reduce=False,
    )

    # Initialize language model
    llm = AsyncLLMEngine.from_engine_args(engine_args)

    # Load classifier
    AIMO_clf = joblib.load('/path/to/clf.pkl')

    # Set up sampling parameters
    sampling_parameters = SamplingParams(
        max_tokens=2048,
        temperature=0.9,
        top_p=1.0,
        stop=[],
        n=1,
        include_stop_str_in_output=True,
        skip_special_tokens=True,
    )

    # Define prompts
    tool = """Write a Python program to solve the following program. \n\"{}\"\nPut the final numerical answer within \\boxed{}.\n\n"""
    cot = """Think step-by-step to solve the following problem.\n\"{}\"\nPut the final numerical answer within \\boxed{}.\n\n"""
    prompt_options = [tool, cot]

    results = []  # List to store best answers for all problems

    # Process each problem
    for problem in problems:  
        PROBLEM_START_TIME = time.time()
        EXTRA_TIME = 0
        
        # Solve problem and get predictions
        pred_list, solution_list, freq_dict = await solve_problem(problem, PROBLEM_START_TIME, EXTRA_TIME)
        
        clean_memory()
        
        # Process predictions
        num_preds = len(pred_list)
        unique_preds = list(set(pred_list))
        prob_list = []
        prob_dict = defaultdict(list)    
        pred_feats = defaultdict(list)
        best = -1

        if num_preds > 0:
            # Calculate probabilities for each prediction
            for i, pred in enumerate(pred_list):
                prob = score_predict(solution_list[i], model, tokenizer, score_head)
                clean_memory()
                prob_list.append(prob)
                prob_dict[pred].append(prob)

            # Calculate features for each unique prediction
            for pred in unique_preds:
                probs = prob_dict[pred]
                n_samples = len(probs)
                freq = n_samples / num_preds

                if n_samples == 1:
                    single_prob = probs[0]
                    pred_feats[pred] = [freq, single_prob, single_prob, single_prob, single_prob, 0, 0]
                else:
                    min_prob = min(probs)
                    max_prob = max(probs)
                    mean_prob = np.mean(probs)
                    median_prob = np.median(probs)
                    std_prob = np.std(probs)
                    q25, q75 = np.percentile(probs, [25, 75])
                    iqr = q75 - q25
                    pred_feats[pred] = [freq, min_prob, max_prob, mean_prob, median_prob, std_prob, iqr]

            # Get final probabilities for each prediction
            final_prob_list = []
            for pred in unique_preds:
                feats = np.array(pred_feats[pred]).reshape(1, -1)  
                print(f"The features for {pred} are {pred_feats[pred]}")
                final_prob = predict_best(feats)
                final_prob_list.append(final_prob)
            
            # Select best prediction
            max_prob_index = np.argmax(final_prob_list)
            best = unique_preds[max_prob_index]
            best_prob = final_prob_list[max_prob_index]
            
            print(f"Best prediction {best} was chosen with probability {best_prob}")

        results.append(best)  # Add the best prediction for this problem to the results list

    return results  # Return all results at once

# To run the main function
if __name__ == "__main__":
    results = asyncio.run(main())
    print("Final results:", results)