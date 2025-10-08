import os
import json

data_config = {
    # "calc_bench/arithmetic": {"has_separate_prompts": True},
    # "calc_bench/count": {"has_separate_prompts": True},
    # "calc_bench/countdown": {"has_separate_prompts": True},
    # "calc_bench/gsm8kr": {"has_separate_prompts": True},
    # "calc_bench/sudoku": {"has_separate_prompts": True},
    "arithmetic_digit": {"has_separate_prompts": True},
    "arithmetic_lang2": {"has_separate_prompts": True},
    "arithmetic_lang3": {"has_separate_prompts": True},
}

model_configs = [
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct"
]



def run_experiment(config):
    cmd = 'python3 -m verl.trainer.main_earl \
    trainer.val_only=True \
    actor_rollout_ref.earl.model.freeze_base_model=False \
    actor_rollout_ref.earl.model.init_from_base=True \
    actor_rollout_ref.earl.training.tools=["calculator"] \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/countdown_1/train.parquet \
    data.train_batch_size=256 \
    data.filter_overlong_prompts=True \
    data.truncation="error" \
    tool_config_path=tool_configs/basic_calculator_baseline_fast.yaml \
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
    for key, value in config.items():
        cmd += f" {key}={value}"
    os.system(cmd)

results = []
for model_name in model_configs:
    for dataset_name, dataset_config in data_config.items():
        config = {}
        model_name_short = model_name.split('/')[-1]
        config.update({
            "data.val_files": os.path.join("./data", dataset_name, "test.parquet"),
            "trainer.validation_data_dir": os.path.join("./val_results", "0_autorunner", model_name_short, dataset_name),
            "trainer.experiment_name": f"val_zeroshot_{model_name_short}_{0}_{dataset_name}",
        })
        config.update(
            {
                "trainer.n_gpus_per_node": 1,
                "data.max_prompt_length": 256,
                "data.max_response_length": 384,
                "actor_rollout_ref.model.path": model_name,
            }
        )
        run_experiment(config)
        
        val_data_file = os.path.join("./val_results", "0_autorunner", model_name_short, dataset_name, "0.jsonl")
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
        else:
            avg_data = {}
            for k in final_data[0].keys():
                avg_data[k] = sum(item[k] for item in final_data if k in item) / len(final_data)
            # record the results
            train_dataset = "zero_shot"
            model_name = model_name
            ckpt_num = '0'
            new_data = {
                "train_dataset": train_dataset,
                "model_name": model_name,
                "ckpt_num": int(ckpt_num),
                "val_dataset": dataset_name,
            }
            for k, v in avg_data.items():
                new_data[k] = v
            results.append(new_data)
print(results)