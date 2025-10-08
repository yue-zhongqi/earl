import random
import operator
import ast
import os
from calc_utils import generate_random_expression
from datasets import Dataset


MAX_RESPONSE_LENGTH = 128
OTHER_TOKENS_LENGTH = 30
BASELINE_DIR = "./data/arithmetic_baseline"
EARL_DIR = "./data/arithmetic_earl"

target_datasets = {
    "digit": {
        "train": 20000,
        "test": 1000,
        "train_digits": 4,
        "train_ops": 3,
        "test_digits": 4,
        "test_ops": 5
    },
    "lang1": {
        "train": 20000,
        "test": 2000,
        "train_digits": 4,
        "train_ops": 4,
        "test_digits": 4,
        "test_ops": 6,
        "train_nlp_p": 0.5,
        "test_nlp_p": 0.5,
    },
    "lang2": {
        "train": 20000,
        "test": 2000,
        "train_digits": 4,
        "train_ops": 3,
        "test_digits": 4,
        "test_ops": 6,
        "train_nlp_p": 0.05,
        "test_nlp_p": 0.7,
    },
    "lang3": {
        "train": 20000,
        "test": 2000,
        "train_digits": 3,
        "train_ops": 5,
        "test_digits": 4,
        "test_ops": 5,
        "train_nlp_p": 0.05,
        "test_nlp_p": 0.7,
    },
}

def generate_equations(num=1000, max_digits=2, max_ops=2, nlp_p=0.0):
    for _ in range(num):
        expr, result, desc = generate_random_expression(max_digits, max_ops, nlp_p=nlp_p)
        if result is not None and len(expr + str(result)) < MAX_RESPONSE_LENGTH - OTHER_TOKENS_LENGTH:
            yield {"input": desc, "label": str(result), "expr": expr}

def make_map_fn(split, variant):
    def process_fn(example, idx):
        question_raw = f"What is {example.pop("input")}?"
        answer_raw = example.pop("label")
        expr = example.pop("expr")
        
        if variant == "earl":
            system_prompt = (
                "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. You are allowed to use calculator with the ' calculate' keyword. For example, you may calculate (12 + 6) / 3 = 6. You can use calculator multiple times in your reasoning process."
            )
        elif variant == "baseline":
            system_prompt = (
                "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. You are allowed to use calculator by wrapping the expression with <calculator> </calculator> tags. The calculator output will be given inside <result> </result> tags. For example, <calculator> (12 + 6) / 3 </calculator> will produce <result> 6 </result>. You can use calculator multiple times in your reasoning process."
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        data = {
            "data_source": f"basic_calculator_tutorial",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": question_raw +' Output the final answer after "####". If the result is a decimal, keep three digits after the decimal point. For example, #### 0.333 or #### 2.',
                },
                {
                    "role": "assistant",
                    "content": "Let me solve this step by step.\n"
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer_raw},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
            "expr": expr + '\n#### ' +answer_raw,
        }
        return data

    return process_fn

if __name__ == "__main__":
    for idx, config in target_datasets.items():
        print(f"Generating dataset {idx} with config: {config}")

        train_nlp_p = config.get("train_nlp_p", 0.0)
        test_nlp_p = config.get("test_nlp_p", 0.0)
        def train_generator_fn():
            return generate_equations(num=config['train'], max_digits=config['train_digits'], max_ops=config['train_ops'], nlp_p=train_nlp_p)

        def test_generator_fn():
            return generate_equations(num=config['test'], max_digits=config['test_digits'], max_ops=config['test_ops'], nlp_p=test_nlp_p)
        
        train_dataset_base = Dataset.from_generator(train_generator_fn, split='train')
        test_dataset_base = Dataset.from_generator(test_generator_fn, split='test')

        for variant, out_dir in [("baseline", BASELINE_DIR), ("earl", EARL_DIR)]:
            variant_dir = f"{out_dir}_{idx}"
            train_path = os.path.join(variant_dir, "train.parquet")
            test_path = os.path.join(variant_dir, "test.parquet")

            if os.path.exists(train_path) and os.path.exists(test_path):
                print(f"[{variant.upper()}] dataset for [{idx}] already exists, skipping.")
                train_loaded = Dataset.from_parquet(train_path)
                test_loaded = Dataset.from_parquet(test_path)
                train_prompt_length = [len(x['prompt'][0]['content']) for x in train_loaded]
                test_prompt_length = [len(x['prompt'][0]['content']) for x in test_loaded]
                print("Max length of prompt in train dataset:", max(train_prompt_length))
                print("Max length of prompt in test dataset:", max(test_prompt_length))
                continue

            os.makedirs(variant_dir, exist_ok=True)
            print(f"Packaging variant: {variant} at {variant_dir}")

            # Apply prompt formatting
            train_dataset = train_dataset_base.map(function=make_map_fn("train", variant), with_indices=True)
            test_dataset = test_dataset_base.map(function=make_map_fn("test", variant), with_indices=True)

            # Save to parquet
            train_dataset.to_parquet(train_path)
            test_dataset.to_parquet(test_path)

            # Quick check
            print(f"{variant.upper()} - First train example:", train_dataset[0])
            print(f"{variant.upper()} - First test example:", test_dataset[0])