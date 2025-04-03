# TAMAC: Time-Aware Multimodal Action Control for Imitation Learning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

## Overview
This repository contains the code for our research work **"TAMAC: Time-Aware Multimodal Action Control for Imitation Learning"**. The paper has been submitted to *ARM 2025*, and we will update the status accordingly.

<!-- ## Paper Information
**Title:** TAMAC: Time-Aware Multimodal Action Control for Imitation Learning  
**Authors:** Author1, Author2, Author3  
**Submitted to:** ARM 2025  
**Status:** Submitted (Under Review)  
[arXiv Link (if available)](https://arxiv.org/abs/example)   -->

## Installation
To set up the environment using Conda, use the following steps:

```bash
# Clone the repository
git clone https://github.com/Loshawn/TAMAC
cd TAMAC

# Create a new conda environment with a specific Python version
conda create --name tamac python=3.9
conda activate tamac

# Install dependencies
pip install -r requirements.txt
```

### Recommended Environment

- **Python**: >= 3.9
- **PyTorch**: >= 2.0.1
- **CUDA**: 11.8 (or appropriate version based on your system)

> **Note**: We used an NVIDIA 4090 GPU for experiments. If you encounter out-of-memory (OOM) issues due to GPU memory limitations when using the provided parameters (such as the batch size, model size, etc.), we recommend upgrading PyTorch to version 2.2 or higher. The newer versions include optimizations for memory usage and performance improvements, which can help mitigate OOM errors.

## Dataset
To prepare the dataset, run the dataset preparation script `record_sim_episodes.sh`:

The following 5 tasks are available for dataset preparation: `sim_transfer_cube_scripted`, `sim_insertion_scripted`, `sim_insertion_scripted_rev`, `sim_stack_cube_scripted` and `sim_storage_cube_scripted`

```bash
# Define the task and dataset directory
task="your_task"  # Replace with your chosen task
dataset_dir="./dataset/$task"

# Step 1: Record simulation episodes for the task
# This command generates the dataset by recording 100 episodes
python3 record_sim_episodes_our.py --task_name $task --dataset_dir $dataset_dir --num_episodes 100

# Step 2: Visualize the recorded episodes
# This command visualizes the dataset and shows the episodes
python3 visualize_episodes.py --dataset_dir $dataset_dir --episode_idx -1
```

## Usage
To run the main training script, use the provided shell script `train.sh` :

```bash
policy_class="TAMAC"
task_name="your_task"
log_name="Demo"

python3 imitate_episodes_our.py \
--task_name $task_name \
--ckpt_dir "ckpt/$policy_class/$task_name/$log_name" \
--policy_class $policy_class --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 4 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0
```

If you want to evaluate the model, simply add the `--eval` flag. And if you wish to use temporal ensemble during evaluation, also add the `--temporal_agg` flag.

<!-- ## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{yourpaper2024,
  author    = {Author1 and Author2 and Author3},
  title     = {TAMAC: Time-Aware Multimodal Action Control for Imitation Learning},
  journal   = {ARM 2025},
  year      = {2024},
  note      = {Under Review}
}
``` -->

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions, feel free to reach out to [zhanghb77@mail2.sysu.edu.cn](mailto:zhanghb77@mail2.sysu.edu.cn).
