import random
import operator
import ast
import os
from calc_utils import generate_random_expression, gen_with_factor, random_number, rand_num2words
from datasets import concatenate_datasets, Dataset
import re
LOCAL_DIR = "./data/calc_bench/"

# arithmetic
def generate_equations(n=1000, max_digits=2, max_ops=2, nlp_p=0.0):
    for _ in range(n):
        expr, result, desc = generate_random_expression(max_digits, max_ops, nlp_p=nlp_p)
        yield {"question": f"What is {desc}? Output your answer after `####`.", "answer": str(result), "expr": expr, "task": "arithmetic"}

# countdown
def generate_countdowns(n=1000, max_digits=2, max_ops=2):
    for _ in range(n):
        expr, result, desc = generate_random_expression(max_digits, max_ops, nlp_p=0)
        numbers = list(map(int, re.findall(r'\d+', expr)))
        # shuffe numbers
        import random
        random.shuffle(numbers)
        question = f"Using the numbers {numbers}, create an equation that equals {result}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Output the final answer after `####`. For example, given numbers [1, 2, 3] and target number 1, output #### (1 + 2) / 3."
        yield {"question": question, "answer": str(result), "expr": expr, "task": "countdown"}

# gsm8k rewrite
def generate_gsm8k2(split='train'):
    gsm_dir = "./data/gsm_8k_rewrite/"
    train_loaded = Dataset.from_parquet(os.path.join(gsm_dir, "train.parquet"))
    test_loaded = Dataset.from_parquet(os.path.join(gsm_dir, "test.parquet"))
    dataset = train_loaded if split == 'train' else test_loaded
    for item in dataset:
        if item['extra_info'].get('rewrite_tag', 'no') == 'yes':
            question = item['extra_info']['question_rewrite']
            question += " Output your final answer after `####`."
            answer = item['extra_info']['answer_rewrite'].split('\n#### ')[1]
            expr = item['extra_info']['answer_rewrite'].split('\n#### ')[0]
            yield {"question": question, "answer": answer, "expr": expr, "task": "gsm8kr"}

def generate_gsm8k(split='train'):
    gsm_dir = "./data/gsm_8k_rewrite/"
    train_loaded = Dataset.from_parquet(os.path.join(gsm_dir, "train.parquet"))
    test_loaded = Dataset.from_parquet(os.path.join(gsm_dir, "test.parquet"))
    dataset = train_loaded if split == 'train' else test_loaded
    for item in dataset:
        question = item['prompt'][0]['content']
        answer = item['reward_model']['ground_truth']
        expr = item['extra_info']['answer']
        yield {"question": question, "answer": answer, "expr": expr, "task": "gsm8kr"}

# train, test = generate_gsm8k()

# n mod m
def generate_n_mod_m(n=1000, max_n=1000, max_m=10):
    for _ in range(n):
        m = random.randint(1, max_m)
        n = random.randint(m, max_n)
        result = n % m
        question = f"What is {n} mod {m}? Output your answer after `####`."
        expr = f"{n} % {m}"
        yield {"question": question, "answer": str(result), "expr": expr, "task": "modulo"}

# approximate
def generate_approximate(n=1000, deg_max=2):
    for _ in range(n):
        deg = random.randint(1, deg_max)
        eq, answer = gen_with_factor(deg_max=deg, coeff_range=(-50,50), round_to=1, r_range=(1,50))
        eq_str = str(eq.expr).replace("**", "^")
        question = f"Solve the equation:\n{eq_str}=0\nFind an approximate solution x in the range [1, 50]. Be as accurate as possible. Hint: 1) Use the calculator tool to evaluate the polynomial at different x values to narrow down the interval where the root is located.\n2) As the calculator tool has no power operator ^, expand powers with multiplication, e.g., use x * x to compute x^2.\n3) Continue until the root is accurate to at least 2-3 decimal places. Output your final answer after `####`. For example, given 4.8*x^3 - 37.18*x^2 - 39.96^x + 40.42=0, output #### 8.6."
        yield {"question": question, "answer": str(answer), "expr": eq_str, "task": "approximate"}
# generate_approximate(deg_max=2)

# repeat numbers
def generate_repeat_number(n=1000, max_digits=20):
    for _ in range(n):
        rand_number = random_number(max_digits=max_digits)
        rand_desc = rand_num2words(rand_number, p=0.5)
        yield {"question": f"Write {rand_desc} in digits after `####`. For example, given 100, outputs #### 100.", "answer": str(rand_number), "expr": str(rand_number), "task": "repeat"}

# count numbers
def generate_count_number(n=1000, max_len=20):
    def generate_counting_problem(max_digits: int):
        # Step 1. Random integer with <= max_digits digits
        num_digits = random.randint(1, max_digits)
        lower = 1 if num_digits == 1 else 10 ** (num_digits - 1)
        upper = 10 ** num_digits - 1
        rand_int = random.randint(lower, upper)
        # Step 2. Pick a digit
        chosen_digit = str(random.randint(0, 9))
        # Step 3. Count occurrences
        count = str(rand_int).count(chosen_digit)
        return rand_int, int(chosen_digit), count
    for _ in range(n):
        rand_int, chosen_digit, count = generate_counting_problem(max_len)
        rand_desc = rand_num2words(rand_int, p=0.9)
        question = f"How many times does the digit {chosen_digit} appear in the number {rand_desc}? Output your answer after `####`. For example, given the number 121 and digit 1, output #### 2."
        yield {"question": question, "answer": str(count), "expr": f"str({rand_int}).count('{chosen_digit}')", "task": "count"}
# generate_count_number()

# 4*4 sudoku
def generate_sudoku(n=1000, min_difficulty=0.1, d1=3, d2=2):
    from sudoku import Sudoku
    example_puzzle = Sudoku(d1, d2).difficulty(0.5)
    example_solution = example_puzzle.solve().board
    example_board = example_puzzle.board
    problems = []
    for _ in range(n):
        difficulty = random.uniform(min_difficulty, 0.7)
        puzzle = Sudoku(d1, d2, seed=random.randint(1, 100000)).difficulty(difficulty)
        solution = puzzle.solve().board
        puzzle_str = str(puzzle.board)
        success = False
        while not success:
            difficulty = random.uniform(min_difficulty, 0.7)
            puzzle = Sudoku(d1, d2, seed=random.randint(1, 100000)).difficulty(difficulty)
            solution = puzzle.solve().board
            puzzle_str = str(puzzle.board)
            if puzzle_str in problems:
                retry_count += 1
                if retry_count > 500:
                    print("Too many retries, stopping generation.")
                    print(len(problems))
                    break
            else:
                problems.append(puzzle_str)
                retry_count = 0
                success = True
        prod = d1 * d2
        system_prompt = "You are a Sudoku solver. Fill in all None entries correctly.\n"
        question = f"Input format:\n- A {prod}x{prod} square JSON array of arrays.\n- Each cell is either an integer or None.\n- Valid digits are 1..{prod}.\nRules:\n- Each row must contain all digits 1..{prod} exactly once.\n- Each column must contain all digits 1..{prod} exactly once.\n- Each {d2}x{d1} subgrid must contain all digits 1..{prod} exactly once.\nOutput format:\n- A {prod}x{prod} square JSON array of arrays.\n- Always put the solved grid immediately after the delimiter `####`\n- Do not include any text before or after the JSON. No code fences.\nExample input:\n{example_board}\nExample output:\n#### {example_solution}\nNow solve the following puzzle:\n{puzzle.board}."
        answer = f"{solution}"
        yield {"question": question, "answer": answer, "expr": str(solution), "system_prompt": system_prompt, "task": "sudoku"}
#print(generate_sudoku())

config = {
    "arithmetic": {
        "fn": generate_equations,
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
        "fn": generate_countdowns,
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
        "fn": generate_n_mod_m,
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
        "fn": generate_repeat_number,
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
        "fn": generate_count_number,
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
        "fn": generate_sudoku,
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
        "fn": generate_gsm8k2,
        'tool': True,
        "train": {
            "split": "train",
        },
        "test": {
            "split": "test",
        }
    }
}


def make_map_fn(split, baseline=False, use_tool=False):
    baseline_tool_hint = "You are allowed to use calculator by wrapping the expression with <calculator> </calculator> tags. The calculator output will follow inside <result> </result> tags. For example, <calculator> (12 + 6) / 3 </calculator> will produce <result> 6 </result>. You can use calculator multiple times in your reasoning process."
    earl_tool_hint = "You are allowed to use calculator with the ' calculate' keyword. For example, you may calculate (12 + 6) / 3 = 6. You can use calculator multiple times in your reasoning process."
    tool_hint = baseline_tool_hint if baseline else earl_tool_hint
    def process_fn(example, idx):
        question = example.pop("question")
        answer = example.pop("answer")
        expr = example.pop("expr")
        task = example.pop("task")
        system_prompt = example.pop("system_prompt", "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n")
        if use_tool:
            system_prompt += tool_hint
        data = {
            "data_source": 'calc_bench',
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": "Let me solve this step by step.\n"
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,
                "task": task,
                "expr": expr,
            },
            "expr": expr,
        }
        return data
    return process_fn

if __name__ == "__main__":
    data_configs = [
        {
            "is_baseline": True,
            "use_tool": True,
        },
        {
            "is_baseline": False,
            "use_tool": True,
        },
        {
            "is_baseline": False,   # does not matter if not use tool
            "use_tool": False,
        },
        
    ]
    for data_config in data_configs:
        is_baseline = data_config['is_baseline']
        global_enable_tool = data_config['use_tool']
        train_dataset_list = []
        test_dataset_list = []
        for d_name, d_config in config.items():
            if not global_enable_tool:
                suffix = ""
            elif is_baseline:
                suffix = "_baseline"
            else:
                suffix = "_earl"
            d_name = d_name + suffix
            dir = os.path.join(LOCAL_DIR, d_name)
            use_tool = d_config.get("tool", False)
            use_tool = use_tool and global_enable_tool
            # if exists
            if os.path.exists(f"{dir}/train.parquet") and os.path.exists(f"{dir}/test.parquet"):
                print(f"Dataset {d_name} already exists, skipping generation.")
                # load train and test parquet and print the max length of prompt
                train_loaded = Dataset.from_parquet(os.path.join(dir, "train.parquet"))
                test_loaded = Dataset.from_parquet(os.path.join(dir, "test.parquet"))

                train_prompt_length = [len(x['prompt'][0]['content']) for x in train_loaded]
                test_prompt_length = [len(x['prompt'][0]['content']) for x in test_loaded]
                print("Max length of prompt in train dataset:", max(train_prompt_length))
                print("Max length of prompt in test dataset:", max(test_prompt_length))
                train_dataset_list.append(train_loaded)
                test_dataset_list.append(test_loaded)
            else:
                print(f"Generating dataset {d_name} with config: {d_config}")

                def train_generator_fn():
                    return d_config['fn'](**d_config['train'])

                def test_generator_fn():
                    return d_config['fn'](**d_config['test'])

                train_dataset = Dataset.from_generator(
                    train_generator_fn, split='train', keep_in_memory=True, cache_dir=None,
                )
                test_dataset = Dataset.from_generator(
                    test_generator_fn, split='test', keep_in_memory=True, cache_dir=None,
                )

                train_dataset = train_dataset.map(function=make_map_fn("train", baseline=is_baseline, use_tool=use_tool), with_indices=True)
                test_dataset = test_dataset.map(function=make_map_fn("test", baseline=is_baseline, use_tool=use_tool), with_indices=True)

                os.makedirs(LOCAL_DIR, exist_ok=True)
                train_dataset.to_parquet(os.path.join(dir, "train.parquet"))
                test_dataset.to_parquet(os.path.join(dir, "test.parquet"))
                train_dataset_list.append(train_dataset)
                test_dataset_list.append(test_dataset)

                # Load the saved parquet files
                train_loaded = Dataset.from_parquet(os.path.join(dir, "train.parquet"))
                test_loaded = Dataset.from_parquet(os.path.join(dir, "test.parquet"))

                # Print the first row of each dataset
                print("First row in train dataset:", train_loaded[0])
                print("First row in test dataset:", test_loaded[0])
        # combine dataset
        d_name = "combined"
        d_name = d_name + ("_baseline" if is_baseline else "_earl")
        dir = os.path.join(LOCAL_DIR, d_name)
        # merge list of train_dataset_list
        merged_train = concatenate_datasets(train_dataset_list)
        merged_test = concatenate_datasets(test_dataset_list)
        # get a random subset of merged_test with 2000 samples
        if len(merged_test) > 2000:
            merged_test = merged_test.shuffle(seed=42).select(range(2000))
        merged_train.to_parquet(os.path.join(dir, "train.parquet"))
        merged_test.to_parquet(os.path.join(dir, "test.parquet"))