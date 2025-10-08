import os
import json
from tabulate import tabulate
import copy

RESULTS_DIR = os.path.join("val_results", "0_autorunner")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.jsonl")
CHECKPOINT_DIR = "checkpoints/earl"
DATASET_DIR = "data"
EARL_DEFAULT_CONFIG = "tool_configs/basic_calculator.yaml"
BASELINE_DEFAULT_CONFIG = "tool_configs/basic_calculator_baseline_fast.yaml"
PRINT_FILE = os.path.join(RESULTS_DIR, "0_summary.txt")

VAL_DATASETS = {
    "countdown": {
        "countdown_0": {"has_separate_prompts": True},
        "countdown_1": {"has_separate_prompts": True},
        "countdown_2": {"has_separate_prompts": True},
    },
    "arithmetic": {
        "arithmetic_digit": {"has_separate_prompts": True},
        "arithmetic_lang3": {"has_separate_prompts": True},
    },
    "recall_numbers": {
        "numbers": {"has_separate_prompts": False},
    },
    "calc_bench": {
        "calc_bench/arithmetic": {"has_separate_prompts": True},
        "calc_bench/count": {"has_separate_prompts": True},
        "calc_bench/countdown": {"has_separate_prompts": True},
        "calc_bench/gsm8kr": {"has_separate_prompts": True},
        "calc_bench/modulo": {"has_separate_prompts": True},
        "calc_bench/repeat": {"has_separate_prompts": True},
    }
}

VAL_CONFIGS = {
    # key: training dataset
    "countdown": {
        "configs": {
            "trainer.n_gpus_per_node": 4,
            "data.max_prompt_length": 256,
            "data.max_response_length": 1024,
            "actor_rollout_ref.model.path": "Qwen/Qwen2.5-3B-Instruct",
        },
        "models": {
            "earl-cpo": {"model_path": "qwen2.5-3b/earl-cpo"},
            "prompt-grpo": {"model_path": "qwen2.5-3b/prompt-grpo-try2"},
            "prompt-cpo": {"model_path": "qwen2.5-3b/prompt-cpo-try2"},
            "earl-cpo-base": {"model_path": "qwen2.5-3b-base/earl-cpo"},
            "prompt-grpo-base": {"model_path": "qwen2.5-3b-base/prompt-grpo"},
            "prompt-cpo-base": {"model_path": "qwen2.5-3b-base/prompt-cpo"},
        },
        "val_datasets": ["countdown", "arithmetic", "recall_numbers"],
    },
    "calc_bench": {
        "configs": {
            "trainer.n_gpus_per_node": 4,
            "data.max_prompt_length": 384,
            "data.max_response_length": 1024,
            "actor_rollout_ref.model.path": "Qwen/Qwen2.5-3B-Instruct",
        },
        "models": {
            "earl-cpo": {"model_path": "qwen2.5-3b/earl-cpo", "tool_config": "tool_configs/combined_calculator.yaml"},
            "prompt-grpo": {"model_path": "qwen2.5-3b/prompt-grpo", "tool_config": "tool_configs/combined_calculator_baseline.yaml"},           # " <calculator>"
            "prompt-cpo": {"model_path": "qwen2.5-3b/prompt-cpo", "tool_config": "tool_configs/combined_calculator_baseline.yaml"},           # " <calculator>"
            "prompt-grpo-cfg1": {"model_path": "qwen2.5-3b/prompt-grpo2", "tool_config": "tool_configs/combined_calculator_baseline2.yaml"},    # " <calculator> "
            "prompt-cpo-cfg1": {"model_path": "qwen2.5-3b/prompt-cpo2", "tool_config": "tool_configs/combined_calculator_baseline2.yaml"},    # " <calculator> "
        },
        "val_datasets": ["calc_bench"],
    },
    "arithmetic-lang": {
        "configs": {
            "trainer.n_gpus_per_node": 2,
            "data.max_prompt_length": 256,
            "data.max_response_length": 128,
            "actor_rollout_ref.model.path": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "models": {
            "lang2-earl-cpo": {"model_path": "qwen2.5-0.5b/lang2-earl-cpo"},
            "lang2-prompt-grpo": {"model_path": "qwen2.5-0.5b/lang2-prompt-grpo"},
            "lang2-prompt-cpo": {"model_path": "qwen2.5-0.5b/lang2-prompt-cpo"},
            "lang3-earl-cpo": {"model_path": "qwen2.5-0.5b/lang3-earl-cpo"},
            "lang3-prompt-grpo": {"model_path": "qwen2.5-0.5b/lang3-prompt-grpo"},
            "lang3-prompt-cpo": {"model_path": "qwen2.5-0.5b/lang3-prompt-cpo"},
        },
        "val_datasets": ["arithmetic"],
    },
}

def read_results():
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                results.append(json.loads(line))
    return results

# Load existing results if available
results = read_results()

def validate_train_dataset_config(config):
    return "models" in config and "val_datasets" in config and len(config["models"].keys()) > 0 and len(config["val_datasets"]) > 0

def get_val_dataset_dir(base_dir, dataset_name, has_separate_prompts, is_earl):
    if not has_separate_prompts:
        val_dir = os.path.join(base_dir, dataset_name)
    else:
        paths = dataset_name.split("/")
        names = paths[-1].split("_")
        if len(names) == 1:
            d_name = f"{dataset_name}_earl" if is_earl else f"{dataset_name}_baseline"
            val_dir = os.path.join(base_dir, d_name)
        elif len(names) == 2:
            p_name = "earl" if is_earl else "baseline"
            names.insert(1, p_name)
            d_name = "_".join(names)
            paths[-1] = d_name
            val_dir = os.path.join(base_dir, '/'.join(paths))
        else:
            raise ValueError(f"Unexpected dataset name format: {dataset_name}")
    files = os.listdir(val_dir)
    if "test.parquet" in files:
        return os.path.join(val_dir, "test.parquet")
    else:
        raise FileNotFoundError(f"test.parquet not found in {val_dir}")

def check_experiment_done(results, train_dataset, model_name, ckpt_num, dataset_name):
    # check if results list have a record with the same train_dataset, model_name, ckpt_num, dataset_name
    for record in results:
        if (record["train_dataset"] == train_dataset and
            record["model_name"] == model_name and
            int(record["ckpt_num"]) == int(ckpt_num) and
            record["val_dataset"] == dataset_name):
            return True
    return False

def record_results(experiment, results):
    val_data_file = experiment["val_data_file"]
    # check if val_data_file exists
    assert os.path.exists(val_data_file), f"Validation data file {val_data_file} does not exist."
    # read the val_data_file
    data = []
    with open(val_data_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    excelude_keys = ["input", "output", "step"]
    filtered_data = [{k: v for k, v in item.items() if k not in excelude_keys} for item in data]
    # remove keys whose values are not int or float
    final_data = []
    for item in filtered_data:
        new_item = {}
        for k, v in item.items():
            if isinstance(v, (int, float)):
                new_item[k] = v
        final_data.append(new_item)
    # average the value for each key
    if len(final_data) == 0:
        print(f"Warning: No valid results found in {val_data_file}.")
        return
    avg_data = {}
    for k in final_data[0].keys():
        avg_data[k] = sum(item[k] for item in final_data if k in item) / len(final_data)
    # record the results
    meta_info = experiment["meta_info"]
    train_dataset = meta_info["train_dataset"]
    model_name = meta_info["model_name"]
    ckpt_num = meta_info["ckpt_num"]
    dataset_name = meta_info["val_dataset"]
    new_data = {
        "train_dataset": train_dataset,
        "model_name": model_name,
        "ckpt_num": int(ckpt_num),
        "val_dataset": dataset_name,
    }
    for k, v in avg_data.items():
        new_data[k] = v
    results.append(new_data)

def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for record in results:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

def write_results(results):
    with open(PRINT_FILE, "w", encoding="utf-8") as f:
        f.write("Validation Results Summary\n\n")

    all_train_datasets = set(record["train_dataset"] for record in results)
    all_val_datasets = set(record["val_dataset"] for record in results)
    # sort all_train_datasets and all_val_datasets
    all_train_datasets = sorted(list(all_train_datasets))
    all_val_datasets = sorted(list(all_val_datasets))
    for train_d in all_train_datasets:
        for val_d in all_val_datasets:
            # filter results for this train and val dataset
            filtered = [r for r in results if r["train_dataset"] == train_d and r["val_dataset"] == val_d]

            # remove "train_dataset" and "val_dataset" key from filtered
            omit_cols = {"train_dataset", "val_dataset"}
            filtered = [{k: v for k, v in row.items() if k not in omit_cols} for row in filtered]
            
            # sort according to model_name and ckpy_num
            filtered = sorted(
                filtered,
                key=lambda x: (x["model_name"],  x["ckpt_num"])
            )
            if len(filtered) == 0:
                continue
            # write into PRINT_FILE
            with open(PRINT_FILE, "a", encoding="utf-8") as f:
                f.write(f"Results for Train Dataset: {train_d} => Val Dataset: {val_d}\n")
                f.write(tabulate(filtered, headers="keys", tablefmt="grid"))
                f.write("\n\n")

def run_experiment(experiment):
    configs = experiment["configs"]
    cmd = 'python3 -m verl.trainer.main_earl \
    trainer.val_only=True \
    actor_rollout_ref.earl.model.freeze_base_model=False \
    actor_rollout_ref.earl.model.init_from_base=True \
    actor_rollout_ref.earl.training.tools=["calculator"] \
    data.train_files=./data/countdown_1/train.parquet \
    data.train_batch_size=256 \
    data.filter_overlong_prompts=True \
    data.truncation="error" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=["console"] \
    trainer.project_name="earl" \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.total_epochs=4'
    for key, value in configs.items():
        cmd += f" {key}={value}"
    os.system(cmd)

# generate required experiments
required_experiments = []
# for training dataset
for train_dataset in VAL_CONFIGS.keys():
    train_dataset_config = VAL_CONFIGS[train_dataset]
    if validate_train_dataset_config(train_dataset_config):

        # for model name
        for model_name, model_info in train_dataset_config["models"].items():
            if "model_path" not in model_info:
                print(f"Warning: Skipping model {model_name} as it lacks 'model_path'.")
                continue
            # get available checkpoints
            ckpt_path = os.path.join(CHECKPOINT_DIR, train_dataset, model_info["model_path"])
            # get folders in ckpt_path
            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint path {ckpt_path} does not exist. Skipping.")
                continue
            ckpt_folders = [f for f in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, f))]
            # get info about this model
            is_earl = "earl" in model_name.lower()
            adv_estimator = "grpo" if "grpo" in model_name.lower() else "cpo"

            # for each checkpoint
            for ckpt_name in ckpt_folders:
                # get ckpt number from "global_step_xxx"
                if not ckpt_name.startswith("global_step_"):
                    print(f"Warning: Skipping folder {ckpt_name} as it does not start with 'global_step_'.")
                    continue
                ckpt_num = ckpt_name.split("_")[-1]

                # for each val dataset group
                for val_dataset_group in train_dataset_config["val_datasets"]:

                    # for each val dataset
                    for dataset_name in VAL_DATASETS[val_dataset_group].keys():
                        val_dataset_config = VAL_DATASETS[val_dataset_group][dataset_name]
                        try:
                            val_dir = get_val_dataset_dir(DATASET_DIR, dataset_name, val_dataset_config["has_separate_prompts"], is_earl)
                        except Exception as e:
                            print(f"Warning: {str(e)}. Skipping dataset {dataset_name}.")
                            continue
                        # check if this experiment is already done
                        if check_experiment_done(results, train_dataset, model_name, ckpt_num, dataset_name):
                            continue
                        #print(f"Scheduling validation: Train Dataset={train_dataset}, Model={model_name}, Ckpt={ckpt_num}, Val Dataset={dataset_name}")
                        experiment_config = train_dataset_config.get("configs", {}).copy()
                        val_data_dir = os.path.join(RESULTS_DIR, train_dataset, model_name, ckpt_num, dataset_name)
                        val_data_file = os.path.join(val_data_dir, "0.jsonl")
                        tool_config_default = EARL_DEFAULT_CONFIG if is_earl else BASELINE_DEFAULT_CONFIG
                        tool_config = model_info.get("tool_config", tool_config_default)
                        # print(f"val_{train_dataset}_{model_name}_{ckpt_num}_{dataset_name} using {tool_config}.")
                        experiment_config.update({
                            "algorithm.adv_estimator": adv_estimator,
                            "trainer.finetune_from_path": os.path.join(ckpt_path, ckpt_name),
                            "tool_config_path": tool_config,
                            "data.val_files": val_dir,
                            "trainer.validation_data_dir": val_data_dir,
                            "trainer.experiment_name": f"val_{train_dataset}_{model_name}_{ckpt_num}_{dataset_name}",
                        })
                        experiment = {
                            "configs": experiment_config,
                            "val_data_file": val_data_file,
                            "meta_info": {
                                "train_dataset": train_dataset,
                                "model_name": model_name,
                                "ckpt_num": ckpt_num,
                                "val_dataset": dataset_name,
                                "is_earl": is_earl,
                                "ckpt_path": os.path.join(ckpt_path, ckpt_name),
                            }
                        }
                        required_experiments.append(experiment)


# filtered_exp = [exp for exp in required_experiments if 'calc_bench' in exp['configs']['trainer.experiment_name'] ]
# print( f"Total required experiments: {len( filtered_exp)}" )


for experiment in required_experiments:
    # if 'calc_bench' in experiment["configs"]["trainer.experiment_name"] and int(experiment["meta_info"]["ckpt_num"]) > 25:
    print(experiment["configs"]["trainer.experiment_name"])
    retries = 0
    success = False
    while retries < 5 and not success:
        try:
            run_experiment(experiment)
            results = read_results()
            record_results(experiment, results)
            success = True
        except Exception as e:
            print(f"Exception: {e}")
            retries += 1
    if not success:
        raise RuntimeError(f"Experiment {experiment['configs']['trainer.experiment_name']} failed after 5 retries.")
    save_results(results)
    write_results(results)
    
