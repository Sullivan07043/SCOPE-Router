#!/usr/bin/env python3
"""
SCOPE Router Inference Pipeline

This script performs end-to-end inference using the SCOPE-CoT-RL-v3 model:
1. Load test/anchor data (from HuggingFace or custom files)
2. Compute query-to-anchor similarity (using compute_similarity.py)
3. Build SCOPE-RL-data-v3 format prompts
4. Run inference with VLLM
5. Parse results and output selection format for two-stage routing

Usage:
    # Default: use HuggingFace dataset and DEFAULT_POOL
    python scope_inference.py --dataset id --output results/selection.json
    
    # Custom query set with default model pool
    python scope_inference.py --query_file queries.json --output results/selection.json
    
    # Custom model pool (requires anchor inference first)
    python scope_inference.py --dataset id --model_pool custom_pool.txt \
                              --anchor_dir data/anchor_results/ --output results/selection.json

Requirements:
    - vllm
    - transformers
    - datasets
    - sentence-transformers
"""

import os
import re
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Import from local modules
from compute_similarity import (
    load_anchor_data,
    load_embedding_model,
    generate_embeddings,
    compute_similarities,
    load_cached_embeddings,
    save_embeddings,
    HF_DATASET_ID,
    HF_DATASET_OOD,
)

# Add parent directory to path for config imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.model_pools import (
    DEFAULT_POOL,
    PRICING,
    OPENROUTER_ID_TO_MODEL_NAME,
    MODEL_NAME_TO_OPENROUTER_ID,
    load_custom_pool,
)

# Default Configuration
DEFAULT_SCOPE_MODEL = "Cooolder/SCOPE-CoT-RL-v3"
DEFAULT_NUM_ANCHOR_EXAMPLES = 5  # Number of anchor examples in prompt
DEFAULT_TOP_K_SIMILARITY = 10    # Top-K similar anchors to consider
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7


def load_custom_queries(filepath: str) -> List[Dict]:
    """
    Load custom query set from JSON file.
    
    Expected format:
    [
        {"id": "q1", "prompt": "Question text...", "gt": "A", "category": "..."},
        ...
    ]
    """
    print(f"Loading custom queries from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    # Ensure required fields
    for i, q in enumerate(queries):
        if 'id' not in q:
            q['id'] = f"custom_{i:04d}"
        if 'prompt' not in q and 'question' in q:
            q['prompt'] = q['question']
    
    print(f"Loaded {len(queries)} custom queries")
    return queries


def load_anchor_results_from_local(anchor_dir: str, models: List[str]) -> Dict[str, Dict[str, Dict]]:
    """
    Load anchor performance data from local inference results.
    
    Args:
        anchor_dir: Directory containing anchor inference results
        models: List of model IDs to load
    
    Returns:
        {anchor_id: {model_name: {'is_correct': bool, 'token_count': int}}}
    """
    print(f"Loading anchor results from {anchor_dir}...")
    
    anchor_performance = defaultdict(dict)
    loaded_models = 0
    
    for model_id in models:
        model_dir_name = model_id.replace('/', '_')
        result_file = Path(anchor_dir) / model_dir_name / "anchor_results.json"
        
        if not result_file.exists():
            print(f"  Warning: No results for {model_id}")
            continue
        
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        model_short = normalize_model_name(model_id)
        
        for item in results:
            anchor_id = item['id']
            usage = item.get('usage', {})
            token_count = usage.get('completion_tokens', 0) if isinstance(usage, dict) else 0
            
            anchor_performance[anchor_id][model_short] = {
                'is_correct': item.get('is_correct', False),
                'token_count': token_count or 0,
            }
        
        loaded_models += 1
    
    print(f"Loaded results for {loaded_models} models, {len(anchor_performance)} anchors")
    return dict(anchor_performance)


def load_test_data(dataset_type: str = "id") -> List[Dict]:
    """
    Load test data from HuggingFace dataset.
    
    Args:
        dataset_type: 'id' for in-distribution, 'ood' for out-of-distribution
    
    Returns:
        List of test records with 'id', 'prompt', 'model_name', 'is_correct', etc.
    """
    from datasets import load_dataset
    
    dataset_name = HF_DATASET_OOD if dataset_type == "ood" else HF_DATASET_ID
    print(f"Loading test data from {dataset_name}...")
    
    dataset = load_dataset(dataset_name, split="test")
    
    test_data = []
    for item in dataset:
        test_data.append({
            'id': item['id'],
            'prompt': item['prompt'],
            'model_name': item['model_name'],
            'is_correct': item['is_correct'],
            'gt': item.get('gt', ''),
            'category': item.get('category', ''),
            'usage_prompt_tokens': item.get('usage_prompt_tokens', 0),
            'usage_completion_tokens': item.get('usage_completion_tokens', 0),
        })
    
    print(f"Loaded {len(test_data)} test records")
    return test_data


def load_anchor_performance_data(dataset_type: str = "id") -> Dict[str, Dict[str, Dict]]:
    """
    Load anchor performance data (model results on anchor questions).
    
    Args:
        dataset_type: 'id' for in-distribution, 'ood' for out-of-distribution
    
    Returns:
        {anchor_id: {model_name: {'is_correct': bool, 'token_count': int}}}
    """
    from datasets import load_dataset
    
    dataset_name = HF_DATASET_OOD if dataset_type == "ood" else HF_DATASET_ID
    print(f"Loading anchor performance data from {dataset_name}...")
    
    dataset = load_dataset(dataset_name, split="anchor")
    
    anchor_performance = defaultdict(dict)
    for item in dataset:
        anchor_id = item['id']
        model_name = normalize_model_name(item['model_name'])
        
        anchor_performance[anchor_id][model_name] = {
            'is_correct': item['is_correct'],
            'token_count': item.get('usage_completion_tokens', 0),
        }
    
    print(f"Loaded performance data for {len(anchor_performance)} anchors")
    return dict(anchor_performance)


def normalize_model_name(name: str) -> str:
    """Normalize model name by removing prefixes like 'tngtech/', ':free', etc."""
    name = name.replace(':free', '')
    prefixes = ['tngtech/', 'google/', 'meta-llama/', 'mistralai/', 'qwen/', 'nvidia/', 'amazon/', 'openai/']
    for prefix in prefixes:
        name = name.replace(prefix, '')
    
    # Map common variations
    name_map = {
        'gemma-3-12b-it': 'gemma-3-12b',
        'gemma-3-27b-it': 'gemma-3-27b',
        'gemma-3-4b-it': 'gemma-3-4b',
        'llama-3.1-8b-instruct': 'llama-3-1-8b',
        'llama-3.3-70b-instruct': 'llama-3-3-70b',
    }
    
    return name_map.get(name, name)


def get_model_pool_short_names() -> List[str]:
    """Get short model names from DEFAULT_POOL."""
    return [OPENROUTER_ID_TO_MODEL_NAME.get(m, normalize_model_name(m)) for m in DEFAULT_POOL]


def build_scope_prompt(
    target_question: str,
    target_model: str,
    anchor_examples: List[Dict],
    anchor_performance: Dict[str, Dict[str, Dict]],
    num_anchor_examples: int = DEFAULT_NUM_ANCHOR_EXAMPLES
) -> str:
    """
    Build SCOPE-RL-data-v3 format prompt.
    
    This format matches the training data of Cooolder/SCOPE-CoT-RL-v3 model.
    
    Args:
        target_question: The question to predict performance for
        target_model: The target model name (OpenRouter ID format)
        anchor_examples: List of similar anchor questions with metadata
        anchor_performance: {anchor_id: {model_name: {'is_correct': bool, 'token_count': int}}}
        num_anchor_examples: Number of anchor examples to include in prompt
    
    Returns:
        Formatted prompt string
    """
    model_short_name = normalize_model_name(target_model)
    
    # Build anchor examples section
    examples_text = ""
    for i, anchor in enumerate(anchor_examples[:num_anchor_examples], 1):
        anchor_id = anchor['anchor_id']
        anchor_question = anchor.get('anchor_question', '')
        
        # Get this model's performance on this anchor
        model_perf = anchor_performance.get(anchor_id, {}).get(model_short_name, {})
        is_correct = model_perf.get('is_correct', False)
        token_count = model_perf.get('token_count', 0)
        
        correct_str = "yes" if is_correct else "no"
        
        # Handle None token_count
        if token_count is None:
            token_count = 0
        
        examples_text += f"""Example {i}:
Question: {anchor_question}
Performance: {{len: {token_count}, correct: {correct_str}}}

"""
    
    # Build the full prompt - matches SCOPE-RL-data-v3 format exactly
    prompt = f"""### Task
You are a performance prediction expert. Given a target question, {num_anchor_examples} anchor questions with their performance results, and a target AI model, predict how the model will perform on the target question, specifically the output length and correctness after related reasoning analysis.

### Target Model
{target_model}

{examples_text}### Target Question
{target_question}

### Output Format (STRICT)
Analysis: [Your comprehensive analysis covering anchor patterns, target question characteristics, and reasoning.]
Predicted Performance: {{len: [integer], correct: [yes/no]}}

### Output:"""
    
    return prompt


def parse_prediction(output: str) -> Dict:
    """
    Parse model output to extract predicted performance.
    
    Args:
        output: Raw model output string
    
    Returns:
        {'correctness': 'yes'/'no', 'token_length': int, 'confidence': float}
    """
    result = {
        'correctness': 'unknown',
        'token_length': 0,
        'confidence': 0.5
    }
    
    # Try to find "Predicted Performance: {len: X, correct: yes/no}"
    pattern = r'Predicted Performance:\s*\{?\s*len:\s*(\d+)\s*,\s*correct:\s*(yes|no)\s*\}?'
    match = re.search(pattern, output, re.IGNORECASE)
    
    if match:
        result['token_length'] = int(match.group(1))
        result['correctness'] = match.group(2).lower()
        result['confidence'] = 1.0
    else:
        # Fallback: try to find any correct: yes/no pattern
        correct_match = re.search(r'correct:\s*(yes|no)', output, re.IGNORECASE)
        if correct_match:
            result['correctness'] = correct_match.group(1).lower()
            result['confidence'] = 0.8
        
        # Try to find any len: X pattern
        len_match = re.search(r'len:\s*(\d+)', output, re.IGNORECASE)
        if len_match:
            result['token_length'] = int(len_match.group(1))
    
    return result


def calculate_cost(prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
    """Calculate cost in USD."""
    # Get pricing from model name
    short_name = normalize_model_name(model_name)
    openrouter_id = MODEL_NAME_TO_OPENROUTER_ID.get(short_name, '')
    
    pricing = PRICING.get(openrouter_id, {"input": 0.0, "output": 0.0})
    
    cost = (prompt_tokens / 1_000_000) * pricing["input"] + \
           (completion_tokens / 1_000_000) * pricing["output"]
    
    return cost


def run_vllm_inference(
    prompts: List[str],
    model_name: str = DEFAULT_SCOPE_MODEL,
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    tensor_parallel_size: int = 1
) -> List[str]:
    """
    Run batch inference using VLLM.
    
    Args:
        prompts: List of prompt strings
        model_name: HuggingFace model ID
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        tensor_parallel_size: Number of GPUs for tensor parallelism
    
    Returns:
        List of generated output strings
    """
    from vllm import LLM, SamplingParams
    
    print(f"\n{'='*70}")
    print(f"Loading VLLM model: {model_name}")
    print(f"{'='*70}")
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)
    
    return results


def build_selection_format(
    predictions: Dict[str, Dict[str, Dict]],
    test_data: List[Dict],
    method_name: str = "SCOPE-CoT-RL"
) -> Dict:
    """
    Build selection format output for two-stage routing.
    
    Args:
        predictions: {question_id: {model_name: {'predicted': {...}, 'ground_truth': {...}}}}
        test_data: Original test data for ground truth
        method_name: Name of the method
    
    Returns:
        Selection format dict
    """
    # Build ground truth lookup
    gt_lookup = {}
    for item in test_data:
        qid = item['id']
        model = normalize_model_name(item['model_name'])
        
        if qid not in gt_lookup:
            gt_lookup[qid] = {}
        
        prompt_tokens = item.get('usage_prompt_tokens', 0) or 0
        completion_tokens = item.get('usage_completion_tokens', 0) or 0
        
        gt_lookup[qid][model] = {
            'correctness': 'yes' if item['is_correct'] else 'no',
            'token_length': completion_tokens,
            'cost': calculate_cost(prompt_tokens, completion_tokens, model),
        }
    
    # Build questions list
    questions = []
    for qid, models in predictions.items():
        all_models = []
        for model_name, data in models.items():
            # Get ground truth
            gt = gt_lookup.get(qid, {}).get(model_name, {
                'correctness': 'unknown',
                'token_length': 0,
                'cost': 0.0
            })
            
            # Get prediction
            pred = data.get('predicted', {})
            
            # Calculate predicted cost
            pred_cost = calculate_cost(
                gt_lookup.get(qid, {}).get(model_name, {}).get('prompt_tokens', 0),
                pred.get('token_length', 0),
                model_name
            )
            pred['cost'] = pred_cost
            
            all_models.append({
                'model': model_name,
                'predicted': pred,
                'ground_truth': gt
            })
        
        questions.append({
            'question_id': qid,
            'all_models': all_models
        })
    
    return {
        'method': method_name,
        'questions': questions
    }


def build_selection_format_custom(
    predictions: Dict[str, Dict[str, Dict]],
    queries: List[Dict],
    model_pool_ids: List[str],
    method_name: str = "SCOPE-CoT-RL"
) -> Dict:
    """
    Build selection format output for custom queries (no ground truth).
    
    Args:
        predictions: {question_id: {model_name: {'predicted': {...}}}}
        queries: List of custom query dicts
        model_pool_ids: List of model OpenRouter IDs
        method_name: Name of the method
    
    Returns:
        Selection format dict (with predicted only, no ground_truth)
    """
    questions = []
    
    for query in queries:
        qid = query['id']
        
        if qid not in predictions:
            continue
        
        all_models = []
        for model_id in model_pool_ids:
            model_short = OPENROUTER_ID_TO_MODEL_NAME.get(model_id, normalize_model_name(model_id))
            
            pred = predictions.get(qid, {}).get(model_short, {}).get('predicted', {})
            
            # Calculate predicted cost
            pred_cost = calculate_cost(0, pred.get('token_length', 0), model_short)
            pred['cost'] = pred_cost
            
            all_models.append({
                'model': model_short,
                'predicted': pred,
            })
        
        questions.append({
            'question_id': qid,
            'prompt': query.get('prompt', '')[:200] + '...' if len(query.get('prompt', '')) > 200 else query.get('prompt', ''),
            'all_models': all_models
        })
    
    return {
        'method': method_name,
        'is_custom_queries': True,
        'questions': questions
    }


def main():
    parser = argparse.ArgumentParser(
        description="SCOPE Router Inference Pipeline"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["id", "ood"],
        default="id",
        help="Dataset type: 'id' for in-distribution, 'ood' for out-of-distribution"
    )
    parser.add_argument(
        "--query_file", "-q",
        type=str,
        default=None,
        help="Custom query file (JSON). If not provided, uses HuggingFace test set."
    )
    parser.add_argument(
        "--model_pool", "-m",
        type=str,
        default=None,
        help="Custom model pool file (txt). If not provided, uses DEFAULT_POOL."
    )
    parser.add_argument(
        "--anchor_dir",
        type=str,
        default=None,
        help="Directory with custom anchor results (required for custom model pool)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/selection.json",
        help="Output file path for selection format results"
    )
    parser.add_argument(
        "--similarity_output",
        type=str,
        default=None,
        help="Save similarity results to this file (for two_stage_routing.py)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory for caching embeddings"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for VLLM inference"
    )
    parser.add_argument(
        "--tensor_parallel",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process (for testing)"
    )
    parser.add_argument(
        "--cpu_embedding",
        action="store_true",
        help="Use CPU for embedding model"
    )
    
    # SCOPE model and prompt parameters
    parser.add_argument(
        "--scope_model",
        type=str,
        default=DEFAULT_SCOPE_MODEL,
        help=f"SCOPE model to use (default: {DEFAULT_SCOPE_MODEL})"
    )
    parser.add_argument(
        "--num_anchor_examples",
        type=int,
        default=DEFAULT_NUM_ANCHOR_EXAMPLES,
        help=f"Number of anchor examples in prompt (default: {DEFAULT_NUM_ANCHOR_EXAMPLES})"
    )
    parser.add_argument(
        "--top_k_similarity",
        type=int,
        default=DEFAULT_TOP_K_SIMILARITY,
        help=f"Top-K similar anchors to consider (default: {DEFAULT_TOP_K_SIMILARITY})"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum new tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SCOPE Router Inference Pipeline")
    print("="*70)
    print(f"Output: {args.output}")
    print(f"\nConfiguration:")
    print(f"  SCOPE model: {args.scope_model}")
    print(f"  Num anchor examples: {args.num_anchor_examples}")
    print(f"  Top-K similarity: {args.top_k_similarity}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    
    # Determine model pool
    if args.model_pool:
        model_pool_ids = list(load_custom_pool(args.model_pool))
        model_pool = [normalize_model_name(m) for m in model_pool_ids]
        print(f"Model pool: Custom ({len(model_pool)} models)")
        
        # Check anchor_dir is provided for custom pool
        if not args.anchor_dir:
            print("\n❌ Error: --anchor_dir required for custom model pool")
            print("   Run inference_anchor.py first to generate anchor results")
            return
    else:
        model_pool_ids = list(DEFAULT_POOL)
        model_pool = get_model_pool_short_names()
        print(f"Model pool: DEFAULT_POOL ({len(model_pool)} models)")
    
    print(f"Models: {model_pool}")
    print()
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("Step 1: Loading data")
    print("="*70)
    
    # Load anchor performance data
    if args.anchor_dir:
        anchor_performance = load_anchor_results_from_local(args.anchor_dir, model_pool_ids)
    else:
        anchor_performance = load_anchor_performance_data(args.dataset)
    
    # Load query data
    if args.query_file:
        # Custom query set
        custom_queries = load_custom_queries(args.query_file)
        test_data_filtered = []
        unique_questions = custom_queries
        use_custom_queries = True
        print(f"Using custom query set: {len(custom_queries)} questions")
    else:
        # HuggingFace test set
        test_data = load_test_data(args.dataset)
        
        # Filter test data to only include models in pool
        test_data_filtered = [
            item for item in test_data 
            if normalize_model_name(item['model_name']) in model_pool
        ]
        print(f"Filtered test data: {len(test_data_filtered)} records (models in pool)")
        use_custom_queries = False
    
    # Step 2: Load/compute anchor embeddings and similarity
    print("\n" + "="*70)
    print("Step 2: Computing query-to-anchor similarity")
    print("="*70)
    
    # Load embedding model
    embedding_model = load_embedding_model(use_gpu=not args.cpu_embedding)
    
    # Load anchor data and embeddings
    cache_prefix = f"anchor_{args.dataset}"
    anchor_embeddings, anchor_data = load_cached_embeddings(args.cache_dir, cache_prefix)
    
    if anchor_embeddings is None:
        anchor_data = load_anchor_data(args.dataset)
        anchor_embeddings = generate_embeddings(embedding_model, anchor_data, "anchor")
        save_embeddings(anchor_embeddings, anchor_data, args.cache_dir, cache_prefix)
    
    # Get unique questions for similarity computation
    if not use_custom_queries:
        # Group test data by question ID
        questions_by_id = defaultdict(list)
        for item in test_data_filtered:
            questions_by_id[item['id']].append(item)
        
        unique_questions = []
        seen_ids = set()
        for item in test_data_filtered:
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                unique_questions.append({
                    'id': item['id'],
                    'prompt': item['prompt'],
                    'category': item.get('category', ''),
                    'gt': item.get('gt', ''),
                })
    # else: unique_questions already set from custom_queries
    
    if args.limit:
        unique_questions = unique_questions[:args.limit]
        print(f"Limited to {len(unique_questions)} questions for testing")
    
    # Generate query embeddings
    query_embeddings = generate_embeddings(embedding_model, unique_questions, "query")
    
    # Compute similarities
    similarity_results = compute_similarities(
        query_embeddings, anchor_embeddings,
        unique_questions, anchor_data,
        top_k=args.top_k_similarity
    )
    
    # Build similarity lookup
    similarity_lookup = {item['router_id']: item['similar_anchors'] for item in similarity_results}
    
    # Save similarity results if requested
    if args.similarity_output:
        os.makedirs(os.path.dirname(args.similarity_output) if os.path.dirname(args.similarity_output) else '.', exist_ok=True)
        with open(args.similarity_output, 'w', encoding='utf-8') as f:
            json.dump(similarity_results, f, indent=2, ensure_ascii=False)
        print(f"Saved similarity results to: {args.similarity_output}")
    
    # Step 3: Build prompts for SCOPE model
    print("\n" + "="*70)
    print("Step 3: Building SCOPE prompts")
    print("="*70)
    
    prompts = []
    prompt_metadata = []  # Track (question_id, model_name) for each prompt
    
    for question in tqdm(unique_questions, desc="Building prompts"):
        qid = question['id']
        similar_anchors = similarity_lookup.get(qid, [])
        
        if not similar_anchors:
            continue
        
        # Build prompt for each model in pool
        for model_id in model_pool_ids:
            model_short = OPENROUTER_ID_TO_MODEL_NAME.get(model_id, normalize_model_name(model_id))
            
            prompt = build_scope_prompt(
                target_question=question['prompt'],
                target_model=model_id,
                anchor_examples=similar_anchors,
                anchor_performance=anchor_performance,
                num_anchor_examples=args.num_anchor_examples
            )
            
            prompts.append(prompt)
            prompt_metadata.append((qid, model_short))
    
    print(f"Built {len(prompts)} prompts ({len(unique_questions)} questions × {len(model_pool_ids)} models)")
    
    # Step 4: Run VLLM inference
    print("\n" + "="*70)
    print("Step 4: Running VLLM inference")
    print("="*70)
    
    outputs = run_vllm_inference(
        prompts,
        model_name=args.scope_model,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel
    )
    
    # Step 5: Parse predictions
    print("\n" + "="*70)
    print("Step 5: Parsing predictions")
    print("="*70)
    
    predictions = defaultdict(dict)
    
    for i, output in enumerate(tqdm(outputs, desc="Parsing")):
        qid, model_name = prompt_metadata[i]
        parsed = parse_prediction(output)
        
        predictions[qid][model_name] = {
            'predicted': parsed
        }
    
    # Step 6: Build selection format output
    print("\n" + "="*70)
    print("Step 6: Building selection format output")
    print("="*70)
    
    if use_custom_queries:
        # For custom queries, build simplified output (no ground truth)
        selection_output = build_selection_format_custom(
            predictions,
            unique_questions,
            model_pool_ids,
            method_name="SCOPE-CoT-RL-v3"
        )
    else:
        selection_output = build_selection_format(
            predictions,
            test_data_filtered,
            method_name="SCOPE-CoT-RL-v3"
        )
    
    # Save output
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(selection_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved selection format output to: {args.output}")
    print(f"   Total questions: {len(selection_output['questions'])}")
    print(f"   Models per question: {len(model_pool_ids)}")
    
    # Print summary statistics
    if not use_custom_queries:
        print("\n" + "="*70)
        print("Summary Statistics")
        print("="*70)
        
        total_correct_predictions = 0
        total_predictions = 0
        
        for q in selection_output['questions']:
            for m in q['all_models']:
                pred = m['predicted'].get('correctness', 'unknown')
                gt = m['ground_truth'].get('correctness', 'unknown')
                
                if pred != 'unknown' and gt != 'unknown':
                    if pred == gt:
                        total_correct_predictions += 1
                    total_predictions += 1
        
        if total_predictions > 0:
            accuracy = total_correct_predictions / total_predictions
            print(f"Prediction accuracy: {accuracy*100:.2f}% ({total_correct_predictions}/{total_predictions})")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
