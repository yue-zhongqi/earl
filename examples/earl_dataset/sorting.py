import os
import re
from calc_utils import generate_random_expression
from datasets import Dataset
import random

LOCAL_DIR = "./data"
MAX_SYMBOLS = 5

target_datasets = {
    "sort_debug": {
        "n": 2000,
        "is_train": True,
        "tasks": {"sort": 0.95, "compare": 0.05},
        "n_items": {2: 0.3, 3: 0.3, 4: 0.2, 5: 0.2}
    },
    "compare_train": {
        "n": 10000,
        "is_train": True,
        "tasks": {"compare": 1.},
        "n_items": {2: 1.}
    },
    "compare_test": {
        "n": 1000,
        "is_train": False,
        "tasks": {"compare": 1.},
        "n_items": {2: 1.}
    },
    "order_train": {
        "n": 20000,
        "is_train": True,
        "tasks": {"order": 0.95, "compare": 0.05},
        "n_items": {2: 0.3, 3: 0.3, 4: 0.2, 5: 0.2}
    },
    "order_test": {
        "n": 2000,
        "is_train": False,
        "tasks": {"order": 1.,},
        "n_items": {2: 0.3, 3: 0.3, 4: 0.2, 5: 0.2}
    },
    "sort_train": {
        "n": 80000,
        "is_train": True,
        "tasks": {"sort": 1.},
        "n_items": {2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4}
    },
    "sort_4_5": {
        "n": 2000,
        "is_train": False,
        "tasks": {"sort": 1.,},
        "n_items": {4: 0.5, 5: 0.5}
    },
    "sort_4": {
        "n": 2000,
        "is_train": False,
        "tasks": {"sort": 1.,},
        "n_items": {4: 1.}
    }
}

def random_sufficient_comparisons(sorted_symbols, extra_comparisons=0):
    comparisons = []
    # sufficient comparisons
    for i in range(len(sorted_symbols)-1):
        comparisons.append([sorted_symbols[i], sorted_symbols[i+1]])
    # random extra comparisons
    import random
    num_extra = extra_comparisons
    while num_extra > 0:
        a, b = random.sample(sorted_symbols, 2)
        if [a, b] not in comparisons and [b, a] not in comparisons:
            comparisons.append([a, b] if sorted_symbols.index(a) < sorted_symbols.index(b) else [b, a])
            num_extra -= 1
    # shuffle comparisons
    random.shuffle(comparisons)
    return comparisons

def generate_question(n_items, task="sort"):
    assert n_items <= MAX_SYMBOLS

    if task == "compare":
        n_items = 2
        items = list(range(n_items))
        import random
        random.shuffle(items)
        # get two random symbols from A, B, C, ...
        symbols = random.sample([chr(i) for i in range(65, 65 + MAX_SYMBOLS)], 2)  # A, B, C, ...
        interaction_kwargs = {
            "task": task,
            "tools": ["compare"],
            "items": items,
            "symbols": symbols
        }
        answer = f"{symbols[0]} > {symbols[1]}" if items[0] > items[1] \
            else f"{symbols[0]} < {symbols[1]}" if items[0] < items[1] \
            else f"{symbols[0]} = {symbols[1]}"
        return interaction_kwargs, answer
    elif task == 'sort':
        items = list(range(n_items))
        symbols = [chr(i) for i in range(65, 65 + n_items)]  # A, B, C, ...
        import random
        random.shuffle(items)
        # random should_reverse
        should_reverse = random.choice([True, False])
        interaction_kwargs = {
            "task": task,
            "tools": ["compare", "swap"],
            "items": items,
            "reverse": should_reverse,
            "symbols": symbols,
        }
        paired = list(zip(symbols, items))
        sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=should_reverse)
        sorted_symbols = [sym for sym, _ in sorted_pairs]
        return interaction_kwargs, ','.join(sorted_symbols)
    elif task == 'order':
        completed = False
        while not completed:
            items = list(range(n_items))
            symbols = [chr(i) for i in range(65, 65 + n_items)]  # A, B, C, ...
            import random
            random.shuffle(items)
            should_reverse = random.choice([True, False])
            max_extra_edges = {
                2: 0,
                3: 1,
                4: 2,
                5: 3,
            }
            paired = list(zip(symbols, items))
            sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=should_reverse)
            sorted_symbols = [sym for sym, _ in sorted_pairs]
            interaction_kwargs = {
                "task": task,
                "tools": ["swap"],
                "items": items,
                "reverse": should_reverse,
                "symbols": symbols,
                "conditions": random_sufficient_comparisons(
                    sorted_symbols, extra_comparisons=random.randint(0, max_extra_edges[n_items]
                )),
            }
            completed = (','.join(sorted_symbols) != ','.join(symbols))
        return interaction_kwargs, ','.join(sorted_symbols)
    elif task == 'search':
        raise NotImplementedError

def generate_data(num, p_tasks, p_n_items):
    data = []
    for _ in range(num):
        # sample a task from p_tasks
        import random
        task = random.choices(list(p_tasks.keys()), weights=list(p_tasks.values()))[0]
        n_items = random.choices(list(p_n_items.keys()), weights=list( p_n_items.values() ) )[0]
        interaction_kwargs, answer = generate_question(n_items, task)
        data.append({"interaction_kwargs": interaction_kwargs, "answer": answer})
    return data

def prompt(interaction_kwargs, is_baseline):
    baseline_start = "You have access to the following tools:\n"
    earl_start = "You have access to the following tools:\n"
    prompt_out = baseline_start if is_baseline else earl_start
    if 'compare' in interaction_kwargs['tools']:
        baseline_tool_desc = "- compare tool: use <compare> A, B </compare> to compare two items A and B. The output will be enclosed in <result> </result> tags, e.g., <result> A > B </result>.\n"
        earl_tool_desc = "- compare tool: compare A and B: A > B\n"
        prompt_out += baseline_tool_desc if is_baseline else earl_tool_desc
    if 'swap' in interaction_kwargs['tools']:
        baseline_tool_desc = "- swap tool: use <swap> A, B </swap> to swap two items A and B. The resulting symbol sequence will be enclosed in <result> </result> tags, e.g., <result> B, A </result>.\n"
        earl_tool_desc = "- swap tool: swap A and B => B, A\n"
        prompt_out += baseline_tool_desc if is_baseline else earl_tool_desc
    return prompt_out

def describe(interaction_kwargs):
    if interaction_kwargs['task'] == 'sort':
        symbols = interaction_kwargs['symbols']
        should_reverse = interaction_kwargs['reverse']
        order = "descending" if should_reverse else "ascending"
        item_str = ', '.join(symbols)
        q = f"Sort the following items in {order} order: {item_str}. While you do not know the values of the items, you can compare any two items using the compare tool. Once you know the order of them, you can use the swap tool multiple times to complete the task."
        example = "For example, to sort A, B in descending order, if you find A < B with the compare tool, you can use the swap tool on A and B to complete the task."
        req = "Use fewest possible comparisons and swaps to complete the task. Stop when the sequence is sorted and do not output any answer."
        return f"{q} {example} {req}"
    elif interaction_kwargs['task'] == 'compare':
        symbols = interaction_kwargs['symbols']
        item_str = ' and '.join(symbols)
        q = f"Compare the two items: {item_str}. While you do not know the values of the items, you can compare any two items using the compare tool."
        format = "Enclose your answer within <answer> and </answer> tags."
        example = "For example, given A and B, if you find A > B using the compare tool, output <answer> A > B </answer>."
    elif interaction_kwargs['task'] == 'order':
        symbols = interaction_kwargs['symbols']
        should_reverse = interaction_kwargs['reverse']
        order = "descending" if should_reverse else "ascending"
        item_str = ', '.join(symbols)
        conditions = []
        rel1 = ">" if should_reverse else "<"
        rel2 = "<" if should_reverse else ">"
        for a, b in interaction_kwargs['conditions']:
            conditions.append( random.choice([ f"{a} {rel1} {b}", f"{b} {rel2} {a}" ]) )
        conditions_str = ', '.join(conditions)
        q = f"Sort the following items in {order} order: {item_str}. You know the ordering of some pairs of items: {conditions_str}. Based on this, use swap tool to change the order of two items."
        format = "Use fewest possible swaps to complete the task. Stop when the sequence is ordered and do not output any answer."
        example = "For example, to sort A, B, C in descending order, if the condition says A < B and B < C, you can swap A and C to complete the task."
    return f"{q} {format} {example}"

def earl_generator(data):
    for d in data:
        yield {
            "interaction_kwargs": d['interaction_kwargs'],
            "answer": d['answer'],
            "system_prompt": prompt(d['interaction_kwargs'], is_baseline=False),
            "user_request": describe(d['interaction_kwargs']),
        }

def baseline_generator(data):
    for d in data:
        yield {
            "interaction_kwargs": d['interaction_kwargs'],
            "answer": d['answer'],
            "system_prompt": prompt(d['interaction_kwargs'], is_baseline=True),
            "user_request": describe(d['interaction_kwargs']),
        }

def make_map_fn(split):
    def process_fn(example, idx):
        user_request = example.pop('user_request')
        interaction_kwargs = example.pop('interaction_kwargs')
        answer = example.pop('answer')
        prompt = example.pop('system_prompt')
        # print(f"{split}: {question_raw}={answer_raw}")
        data = {
            "data_source": 'sorting',
            "prompt": [{
                "role": "system",
                "content": f"You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. {prompt}"
            },
            { 
                "role": "user",
                "content": user_request,
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n"
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'interaction_kwargs': interaction_kwargs,
                "task": interaction_kwargs['task'],
            },
        }
        return data

    return process_fn

if __name__ == "__main__":
    for name, config in target_datasets.items():
        parts = name.split('_')
        baseline_name = '_'.join(parts[:1] + ['baseline'] + parts[1:])
        baseline_dir = f"{LOCAL_DIR}/{baseline_name}"
        earl_name = '_'.join(parts[:1] + ['earl'] + parts[1:])
        earl_dir = f"{LOCAL_DIR}/{earl_name}"
        if os.path.exists(f"{dir}/train.parquet") and os.path.exists(f"{dir}/test.parquet"):
            print(f"Dataset {name} already exists, skipping generation.")
            # load train and test parquet and print the max length of prompt
            train_loaded = Dataset.from_parquet(os.path.join(dir, "train.parquet"))
            test_loaded = Dataset.from_parquet(os.path.join(dir, "test.parquet"))

            train_prompt_length = [len(x['prompt'][0]['content']) for x in train_loaded]
            test_prompt_length = [len(x['prompt'][0]['content']) for x in test_loaded]
            print("Max length of prompt in train dataset:", max(train_prompt_length))
            print("Max length of prompt in test dataset:", max(test_prompt_length))
            continue
        data = generate_data(num=config['n'], p_tasks=config['tasks'], p_n_items=config['n_items'])

        print(f"Generating dataset {name} with config: {config}")

        def baseline_generator_fn():
            return baseline_generator(data)
        
        def earl_generator_fn():
            return earl_generator(data)
        
        target_file = "train.parquet" if config['is_train'] else "test.parquet"
        split = "train" if config['is_train'] else "test"
        
        baseline_dataset = Dataset.from_generator(baseline_generator_fn, split=split)
        earl_dataset = Dataset.from_generator(earl_generator_fn, split=split)

        baseline_dataset = baseline_dataset.map(function=make_map_fn(split), with_indices=True)
        earl_dataset = earl_dataset.map(function=make_map_fn(split), with_indices=True)

        os.makedirs(LOCAL_DIR, exist_ok=True)
        baseline_dataset.to_parquet(os.path.join(baseline_dir, target_file))
        earl_dataset.to_parquet(os.path.join(earl_dir, target_file))

        # Load the saved parquet files
        baseline_loaded = Dataset.from_parquet(os.path.join(baseline_dir, target_file))
        earl_loaded = Dataset.from_parquet(os.path.join(earl_dir, target_file))

        # Print the first row of each dataset
        print("First row in train dataset:", baseline_loaded[0])
        print("First row in test dataset:", earl_loaded[0])
    