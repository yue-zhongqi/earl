# Codebase for ExpA and EARL

ExpA is a new paradigm for Large Language Models to interact with external environments, such as using calculators, or external APIs. It decouples environment interactions from language by internalizing them in an Expanded Action space (ExpA), beyond the vocabulary of LLMs. Please find the paper [here]() and its implementation in this repository.

## 1. Setup



### Option 1: Using Apptainer

```bash
apptainer build verl.sif \
  docker://whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3
```


### Option 2: Local Installation

Follow the [official VeRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html), or use the quick setup below:

```bash
# Create and activate a conda environment
conda create -n verl python=3.10 -y
conda activate verl

# Install dependencies
bash scripts/install_vllm_sglang_mcore.sh

# Clone and install VeRL
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .

# Install PyTorch (choose CUDA version matching your NVIDIA driver)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124

# Resolve potential dependency issues based on error messages
pip install tensordict==0.6.2
pip install torchdata msgspec vllm==0.8.5.post1 tensorboard==2.16.2
```

> *If you encounter version conflicts, update dependencies according to the error messages.*

---

## 2. Data Generation

We use four datasets: arithmetic for 0.5B experiments, countdown for ablation, CalcBench and Sorting for our main experiments. CalcBench statistics is given below. Please refer to our paper for additional details.

| **Task** | **Max Number (10^x)** |        | **#Operands** |        | **Lang. Portion** |        | **#Instances** |        |
|-----------|-----------------------|--------|----------------|--------|-------------------|--------|----------------|--------|
|           | **Train** | **Test** | **Train** | **Test** | **Train** | **Test** | **Train** | **Test** |
| Arithmetic | 5 | 5 | 5 | 7 | 10% | 70% | 1,000 | 2,000 |
| Countdown | 4 | 4 | 4 | 4 | N/A | N/A | 20,000 | 2,000 |
| GSM8K* | 6 | 6 | N/A | N/A | N/A | N/A | 5,317 | 579 |
| Count | 20 | 20 | N/A | N/A | 90% | 90% | 1,000 | 2,000 |

Dataset generation script is given under examples/earl_dataset/. SFT data can be found [here]().

---

## 3. Implementation
Our code is implemented based on the [VeRL](https://github.com/volcengine/verl) library. Main modifications include:
### Trainer
- Add main_earl.py under verl.trainer as the entry point for training with external environments (for both EARL, Prompt+GRPO, Prompt+CPO and SFT+GRPO);
- Add verl.trainer.earl.earl_trainer to implement training loop with external environments.

### Workers
- We add worker implementations under verl.workers.earl;
- Files that begin with "vllm" include Customizations to the vLLM library for rollout. This is to enable Algorithm 1 in the paper, as well as action masking for baselines, e.g., when \<calculator\> tag is detected, LLMs can only output calculator-related tokens such as digits and operations.


---



## 4. Training

Before starting, ensure that **setup** and **data generation** are completed.

### Quick Start
For quick debugging, run:

```bash
bash examples/earl_trainer/arithmetic_lang_earl.sh
```

For debugging with breakpoints, please use [Ray Distributed Debugger](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html).

Full training scripts can be found under:

  ```
  exps/train/
  ```

### Customizing Training Configuration

Below are common configuration options you may wish to adjust:

| Key                                             | Description                                                                                                                                        |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `algorithm.adv_estimator`                       | Choose between `CPO` or `GRPO`.                                                                                                                    |
| `actor_rollout_ref.earl.training.tools`         | Provide a list of strings specifying environment names to be intervened by **CPO**. Only relevant when `algorithm.adv_estimator = CPO`.            |
| `tool_config_path`                              | Path to a config file in `tool_configs/`. Use `*_baseline` for baselines, or the corresponding ExpA config otherwise.                              |
| `data.train_files`, `data.val_files`            | Paths to training and validation data. Note that **baselines** and **ExpA** use different prompts and dataset names (see data generation scripts). |
| `actor_rollout_ref.earl.training.loss`          | For **SFT**, set to `"cross_entropy_reg"`.                                                                                                         |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | For **SFT**, set to `"ce"`.                                                                                                                        |
| —                                               | For all other configurations, please refer to the [VeRL documentation](https://verl.readthedocs.io/en/latest/examples/config.html).                |

---

## 5. Validation

For quick validation, add the flag:

```bash
trainer.val_only=True
```

to your training command or configuration file.

We also provide:

```
exps/run_validation.py
```

to evaluate **all saved checkpoints** on each validation dataset.
Modify the script as needed to match your experimental setup.

