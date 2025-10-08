import os
import re
from calc_utils import generate_random_expression
from datasets import Dataset

MAX_RESPONSE_LENGTH = 128
OTHER_TOKENS_LENGTH = 30
BASELINE_DIR = "./data/countdown_baseline"
EARL_DIR = "./data/countdown_earl"

target_datasets = {
    "0": {
        "train": 20000,
        "test": 2000,
        "train_digits": 2,
        "train_ops": 3,
        "test_digits": 2,
        "test_ops": 3
    }, # IID dataset
    "1": {
        "train": 20000,
        "test": 2000,
        "train_digits": 4,
        "train_ops": 3,
        "test_digits": 4,
        "test_ops": 3
    }, # IID dataset
    "2": {
        "train": 20000,
        "test": 2000,
        "train_digits": 3,
        "train_ops": 4,
        "test_digits": 3,
        "test_ops": 4
    }, # IID dataset
}

def is_integer(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False

def generate_countdowns(num=1000, max_digits=2, max_ops=2):
    for _ in range(num):
        expr, result, desc = generate_random_expression(max_digits, max_ops, nlp_p=0)
        if result is not None and len(expr + str(result)) < MAX_RESPONSE_LENGTH - OTHER_TOKENS_LENGTH and is_integer(result):
            numbers = list(map(int, re.findall(r'\d+', expr)))
            # shuffe numbers
            import random
            random.shuffle(numbers)
            print(expr)
            yield {"numbers": numbers, "label": str(result), "expr": expr}

def make_map_fn(split, variant):
    def process_fn(example, idx):
        nums = example.pop('numbers')
        target = example.pop('label')
        expr = example.pop('expr')

        if variant == "earl":
            system_prompt = (
                "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. You are allowed to use calculator with the ' calculate' keyword. For example, you may calculate (12 + 6) / 3 = 6. You can use calculator multiple times in your reasoning process."
            )
        elif variant == "baseline":
            system_prompt = (
                "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. You are allowed to use calculator by wrapping the expression with <calculator> </calculator> tags. The calculator output will follow inside <result> </result> tags. For example, <calculator> (12 + 6) / 3 </calculator> will produce <result> 6 </result>. You can use calculator multiple times in your reasoning process."
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        data = {
            "data_source": 'countdown',
            "prompt": [
                {"role": "system", "content": system_prompt},
                { 
                    "role": "user",
                    "content": f"Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."
                },
                {
                    "role": "assistant",
                    "content": "Let me solve this step by step.\n<think>"
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": '{}->{}'.format(str(nums), str(target))
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'nums': nums,
                'target': target,
                'expr': expr
            }
        }
        return data

    return process_fn

if __name__ == "__main__":
    for idx, config in target_datasets.items():
        print(f"Generating dataset {idx} with config: {config}")

        def train_generator_fn():
            return generate_countdowns(num=config['train'], max_digits=config['train_digits'], max_ops=config['train_ops'])

        def test_generator_fn():
            return generate_countdowns(num=config['test'], max_digits=config['test_digits'], max_ops=config['test_ops'])
        
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
                print("First row in train dataset:", train_loaded[0])
                print("First row in test dataset:", test_loaded[0])
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
    