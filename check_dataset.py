
import os
from datasets import concatenate_datasets, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import time


model_name = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

LOCAL_DIR = "./data/calc_bench/"
config = {
    "arithmetic": {
        "tool": True,
        "train": {
            "n": 1000,
            "max_digits": 5,
            "max_ops": 4,
            "nlp_p": 0.1,
        },
        "test": {
            "n": 2000,
            "max_digits": 5,
            "max_ops": 6,
            "nlp_p": 0.7
        },
    },
    "countdown": {
        "tool": True,
        "train": {
            "n": 20000,
            "max_digits": 4,
            "max_ops": 3,
        },
        "test": {
            "n": 2000,
            "max_digits": 4,
            "max_ops": 3,
        },
    },
    "modulo": {
        "tool": True,
        "train": {
            "n": 2000,
            "max_n": 10000000,
            "max_m": 100000,
        },
        "test": {
            "n": 2000,
            "max_n": 1000000000,
            "max_m": 10000000,
        }
    },
    "repeat": {
        "tool": False,
        "train": {
            "n": 1000,
            "max_digits": 10,
        },
        "test": {
            "n": 2000,
            "max_digits": 10,
        }
    },
    # "approximate": {
    #     "fn": generate_approximate,
    #     "tool": True,
    #     "train": {
    #         "n": 2000,
    #         "deg_max": 3,
    #     },
    #     "test": {
    #         "n": 500,
    #         "deg_max": 1,
    #     }
    # }, #model not using calculator
    "count": {
        "tool": False,
        "train": {
            "n": 1000,
            "max_len": 20,
        },
        "test": {
            "n": 2000,
            "max_len": 20,
        }
    },
    "sudoku": {
        "tool": False,
        "train": {
            "n": 400,
            "min_difficulty": 0.1,
            "d1": 3,
            "d2": 2
        },
        "test": {
            "n": 200,
            "min_difficulty": 0.5,
            "d1": 2,
            "d2": 2
        }
    },
    "gsm8kr": {
        "train": {
            "split": "train",
        },
        "test": {
            "split": "test",
        }
    }
}

def check_prompt_length(data):
    return len(tok(data['prompt'][0]['content'])['input_ids']) + len(tok(data['prompt'][1]['content'])['input_ids']) + len(tok(data['prompt'][2]['content'])['input_ids'])

if __name__ == "__main__":
    for is_baseline in [True, False]:
        for d_name, d_config in config.items():
            d_name = d_name + ("_baseline" if is_baseline else "_earl")
            dir = os.path.join(LOCAL_DIR, d_name)
            use_tool = d_config.get("tool", False)
            # if exists
            if os.path.exists(f"{dir}/train.parquet") and os.path.exists(f"{dir}/test.parquet"):
                # breakpoint()
                print(f"Dataset {d_name} already exists, skipping generation.")
                # load train and test parquet and print the max length of prompt
                train_loaded = Dataset.from_parquet(os.path.join(dir, "train.parquet"))
                test_loaded = Dataset.from_parquet(os.path.join(dir, "test.parquet"))

                train_prompt_length = [check_prompt_length(x) for x in train_loaded]
                test_prompt_length = [check_prompt_length(x) for x in test_loaded]
                print("Max length of prompt in train dataset:", max(train_prompt_length))
                print("Max length of prompt in test dataset:", max(test_prompt_length))