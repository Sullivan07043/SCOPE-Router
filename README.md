# SCOPE: Scalable and Controllable Routing via Pre-hoc Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2601.22323-b31b1b.svg)](https://arxiv.org/abs/2601.22323)
[![HuggingFace](https://img.shields.io/badge/🤗-Models%20%26%20Datasets-yellow)](https://huggingface.co/Cooolder)
[![GitHub](https://img.shields.io/github/license/Sullivan07043/SCOPE-Router)](https://github.com/Sullivan07043/SCOPE-Router)

SCOPE is an intelligent LLM routing framework that predicts how accurate and how expensive each model will be before running it, allowing users to control cost-accuracy trade-offs and naturally handle new models.

## Overview

Model routing chooses which language model to use for each query. By sending easy queries to cheaper models and hard queries to stronger ones, it can significantly reduce inference cost while maintaining high accuracy.

SCOPE goes beyond simple model selection by **predicting model performance**. Trained with reinforcement learning, SCOPE makes reasoning-based predictions by retrieving how models behave on similar problems, enabling it to:

- **Boost accuracy by up to 25.7%** when performance is the priority
- **Cut costs by up to 95.1%** when efficiency matters most
- **Adapt to new, unseen models** without retraining

The framework uses a two-stage routing approach:

- **Stage I (Offline Calibration)**: Build performance statistics from anchor questions using semantic similarity
- **Stage II (Online Routing)**: For each query, combine anchor-based calibration with SCOPE model predictions to select the optimal model

## Key Features

- **Intelligent Model Selection**: Automatically select the best LLM for each query
- **Cost-Accuracy Trade-off**: Control the balance between accuracy and cost with a single parameter
- **Budget-Aware Routing**: Given a budget, automatically find the optimal configuration
- **Custom Model Pool Support**: Use your own set of models
- **Custom Query Set Support**: Route your own questions

## Installation

```bash
# Clone the repository
git clone https://github.com/Sullivan07043/SCOPE-Router.git
cd SCOPE-Router

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- CUDA (recommended for VLLM inference)
- ~4GB GPU memory for embedding model
- ~16GB GPU memory for SCOPE-CoT-RL-v3 model

## Quick Start

### Using Default Dataset and Model Pool

```bash
cd scripts

# Step 1: Run SCOPE inference (generates predictions for each query-model pair)
python scope_inference.py \
    --dataset id \
    --output results/selection.json \
    --similarity_output results/similarity.json

# Step 2: Run two-stage routing with budget constraint
python two_stage_routing.py \
    --selection results/selection.json \
    --similarity results/similarity.json \
    --budget 10.0 \
    --output results/routing_results.json
```

### Output

The routing script outputs:
- **Optimal alpha**: The automatically selected trade-off parameter
- **Accuracy**: Percentage of correctly answered questions
- **Total cost**: Total API cost in USD
- **Model distribution**: How often each model was selected
- **Per-question decisions**: Which model was selected for each query

## Detailed Usage

### 1. SCOPE Inference (`scope_inference.py`)

Runs the SCOPE-CoT-RL-v3 model to predict performance for each query-model pair.

```bash
# Basic usage with HuggingFace dataset
python scope_inference.py --dataset id --output selection.json

# With custom query file
python scope_inference.py \
    --query_file ../examples/custom_queries.json \
    --output selection.json \
    --similarity_output similarity.json

# With custom model pool (requires anchor inference first)
python scope_inference.py \
    --dataset id \
    --model_pool ../examples/custom_model_pool.txt \
    --anchor_dir data/anchor_results/ \
    --output selection.json
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset type: `id` (in-distribution) or `ood` (out-of-distribution) | `id` |
| `--query_file` | Custom query JSON file | None |
| `--model_pool` | Custom model pool file (txt) | None (uses DEFAULT_POOL) |
| `--anchor_dir` | Directory with custom anchor results | None |
| `--output` | Output selection JSON file | `results/selection.json` |
| `--similarity_output` | Save similarity results | None |
| `--tensor_parallel` | Number of GPUs for VLLM | 1 |
| `--limit` | Limit questions for testing | None |

### 2. Two-Stage Routing (`two_stage_routing.py`)

Performs the two-stage routing algorithm to select optimal models.

```bash
# With budget constraint (recommended)
python two_stage_routing.py \
    --selection selection.json \
    --similarity similarity.json \
    --budget 10.0

# With specific alpha value
python two_stage_routing.py \
    --selection selection.json \
    --similarity similarity.json \
    --alpha 0.5

# Generate full Pareto curve
python two_stage_routing.py \
    --selection selection.json \
    --similarity similarity.json
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--selection` | Selection JSON from scope_inference.py | Required |
| `--similarity` | Similarity JSON from scope_inference.py | Required |
| `--budget` | Total budget in USD (auto-selects optimal alpha) | None |
| `--alpha` | Trade-off parameter: 0=cost-focused, 1=accuracy-focused | None |
| `--dataset` | Dataset type for anchor data | `id` |
| `--output` | Output routing results JSON | `routing_results.json` |
| `--alpha_steps` | Number of alpha values to search | 101 |

### 3. Custom Model Pool (`inference_anchor.py`)

When using a custom model pool, first run inference on the anchor set.

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY=your_api_key_here

# Run anchor inference
python inference_anchor.py \
    --model_pool ../examples/custom_model_pool.txt \
    --output data/anchor_results/ \
    --dataset id
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_pool` | Custom model pool file | Required |
| `--output` | Output directory for results | `data/anchor_results` |
| `--dataset` | Dataset type | `id` |
| `--api_key` | OpenRouter API key (or use env var) | None |
| `--limit` | Limit anchors for testing | None |

## File Formats

### Custom Query File (`custom_queries.json`)

```json
[
  {
    "id": "q001",
    "prompt": "What is the capital of France?\n\nA. London\nB. Paris\nC. Berlin\nD. Madrid",
    "gt": "B",
    "category": "geography"
  },
  {
    "id": "q002",
    "prompt": "What is 2 + 2?\n\nA. 3\nB. 4\nC. 5\nD. 6",
    "gt": "B",
    "category": "math"
  }
]
```

### Custom Model Pool File (`custom_model_pool.txt`)

```
# One OpenRouter model ID per line
# Lines starting with # are comments
google/gemma-3-27b-it
meta-llama/llama-3.3-70b-instruct
qwen/qwen3-14b
tngtech/deepseek-r1t2-chimera
```

### Selection Output Format

```json
{
  "method": "SCOPE-CoT-RL-v3",
  "questions": [
    {
      "question_id": "q001",
      "all_models": [
        {
          "model": "gemma-3-27b",
          "predicted": {
            "correctness": "yes",
            "token_length": 150,
            "cost": 0.0023,
            "confidence": 1.0
          },
          "ground_truth": {
            "correctness": "yes",
            "token_length": 142,
            "cost": 0.0021
          }
        }
      ]
    }
  ]
}
```

### Routing Results Format

```json
{
  "method": "SCOPE-CoT-RL-v3",
  "budget": 10.0,
  "optimal_alpha": 0.42,
  "result": {
    "accuracy": 0.6521,
    "total_cost": 9.87,
    "average_cost": 0.00415,
    "model_distribution": {
      "deepseek-r1t2-chimera": 856,
      "gemma-3-27b": 524,
      "qwen3-14b": 412
    },
    "routing_decisions": [
      {
        "question_id": "q001",
        "selected_model": "deepseek-r1t2-chimera",
        "ground_truth_correct": true,
        "cost": 0.0056
      }
    ]
  }
}
```

## Project Structure

```
SCOPE-Router/
├── config/
│   ├── __init__.py
│   └── model_pools.py          # Model pool definitions and pricing
├── scripts/
│   ├── __init__.py
│   ├── compute_similarity.py    # Query-to-anchor similarity computation
│   ├── inference_anchor.py      # Anchor set inference for custom pools
│   ├── scope_inference.py       # SCOPE model prediction pipeline
│   └── two_stage_routing.py     # Two-stage routing algorithm
├── examples/
│   ├── custom_model_pool.txt    # Example custom model pool
│   └── custom_queries.json      # Example custom query set
├── requirements.txt
└── README.md
```

## Default Model Pool

The default model pool includes 7 models balanced across cost tiers:

| Model | Type | Accuracy | Avg Cost |
|-------|------|----------|----------|
| deepseek-r1t2-chimera | High-cost | 73.60% | $0.719 |
| nova-2-lite-v1 | High-cost | 60.71% | $2.386 |
| qwen3-235b-a22b | High-cost | 58.96% | $0.619 |
| qwen3-14b | Mid-cost | 58.73% | $0.294 |
| gpt-oss-20b | Low-cost | 59.13% | $0.082 |
| llama-3-3-70b | Low-cost | 54.76% | $0.050 |
| gemma-3-27b | Low-cost | 53.17% | $0.027 |

## Datasets

SCOPE uses the following HuggingFace datasets:

- **In-Distribution**: [Cooolder/SCOPE-60K-final](https://huggingface.co/datasets/Cooolder/SCOPE-60K-final)
- **Out-of-Distribution**: [Cooolder/SCOPE-60K-OOD-final](https://huggingface.co/datasets/Cooolder/SCOPE-60K-OOD-final)

## Models

- **SCOPE Predictor**: [Cooolder/SCOPE-CoT-RL-v3](https://huggingface.co/Cooolder/SCOPE-CoT-RL-v3) - Pre-trained model, ready to use
- **Embedding Model**: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## How It Works

### Two-Stage Routing Algorithm

**Stage I (Offline Calibration):**
1. For each query, find top-K most similar anchor questions using semantic embeddings
2. Aggregate model performance statistics from these anchors with similarity weighting
3. Normalize accuracy and cost within each query's context

**Stage II (Online Routing):**
1. Compute Stage I score from anchor statistics: `score_1 = α × norm_acc + (1-α) × (1-norm_cost)^sensitivity`
2. Compute Stage II score from SCOPE predictions: `score_2 = α × pred_correct + (1-α) × (1-pred_cost)^sensitivity`
3. Combine scores: `final_score = w × score_1 + (1-w) × score_2`
4. Select model with highest final score

### Budget-Aware Alpha Selection

Given a budget constraint:
1. Scan alpha values from 0 (cost-focused) to 1 (accuracy-focused)
2. For each alpha, compute routing result and total cost
3. Select alpha that maximizes accuracy while staying within budget

## Citation

If you use SCOPE in your research, please cite:

```bibtex
@article{cao2026scope,
  title={Models Under SCOPE: Scalable and Controllable Routing via Pre-hoc Reasoning},
  author={Cao, Qi and Zhang, Shuhao and Zhou, Ruizhe and Zhang, Ruiyi and Qin, Peijia and Xie, Pengtao},
  journal={arXiv preprint arXiv:2601.22323},
  year={2026}
}
```

Paper: [https://arxiv.org/abs/2601.22323](https://arxiv.org/abs/2601.22323)

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
