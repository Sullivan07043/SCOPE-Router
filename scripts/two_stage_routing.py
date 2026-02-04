#!/usr/bin/env python3
"""
SCOPE Two-Stage Routing

This script implements the two-stage routing algorithm:
- Stage I (Offline): Build anchor statistics with similarity weighting
- Stage II (Online): Route each query to optimal model combining anchor stats and predictions

Input: Selection format JSON (output from scope_inference.py)
Output: Routing results with accuracy, cost, and optimal model selection

Usage:
    # Auto-select optimal alpha given budget (recommended)
    python two_stage_routing.py --selection selection.json --similarity sim.json --budget 10.0
    
    # Specify alpha manually
    python two_stage_routing.py --selection selection.json --similarity sim.json --alpha 0.5
    
    # Compute full Pareto curve (no alpha or budget)
    python two_stage_routing.py --selection selection.json --similarity sim.json
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_pools import (
    DEFAULT_POOL,
    AVAILABLE_POOLS,
    get_pool,
    PRICING,
    OPENROUTER_ID_TO_MODEL_NAME,
    MODEL_NAME_TO_OPENROUTER_ID,
    calculate_cost,
    load_pricing,
)

# Import from compute_similarity
from compute_similarity import (
    load_anchor_data as load_anchor_questions,
    load_embedding_model,
    generate_embeddings,
    compute_similarities,
    load_cached_embeddings,
    save_embeddings,
    HF_DATASET_ID,
    HF_DATASET_OOD,
    TOP_K,
)

# Default Configuration
DEFAULT_TOP_K = 6            # Number of anchor questions to use
DEFAULT_SIMILARITY_POWER = 2.0   # Power for similarity weighting
DEFAULT_COST_SENSITIVITY = 2.0   # Cost sensitivity parameter
DEFAULT_STAGE1_WEIGHT = 0.3      # Weight for Stage I (anchor calibration)


def load_selection_data(filepath: str) -> Dict:
    """Load selection format data."""
    print(f"Loading selection data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data.get('questions', []))} questions")
    return data


def load_similarity_data(filepath: str) -> List[Dict]:
    """Load similarity data from file."""
    print(f"Loading similarity data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded similarities for {len(data)} questions")
    return data


def load_anchor_performance(dataset_type: str = "id") -> Dict[str, Dict[str, Dict]]:
    """
    Load anchor performance data from HuggingFace.
    
    Returns:
        {anchor_id: {model_name: {'is_correct': bool, 'cost': float}}}
    """
    from datasets import load_dataset
    
    dataset_name = HF_DATASET_OOD if dataset_type == "ood" else HF_DATASET_ID
    print(f"Loading anchor performance from {dataset_name}...")
    
    dataset = load_dataset(dataset_name, split="anchor")
    
    anchor_performance = defaultdict(dict)
    
    for item in dataset:
        anchor_id = item['id']
        model_name = normalize_model_name(item['model_name'])
        
        prompt_tokens = item.get('usage_prompt_tokens', 0) or 0
        completion_tokens = item.get('usage_completion_tokens', 0) or 0
        
        # Calculate cost
        openrouter_id = MODEL_NAME_TO_OPENROUTER_ID.get(model_name, "")
        cost = calculate_cost(openrouter_id, prompt_tokens, completion_tokens)
        
        anchor_performance[anchor_id][model_name] = {
            'is_correct': item.get('is_correct', False),
            'cost': cost,
        }
    
    print(f"Loaded performance for {len(anchor_performance)} anchors")
    return dict(anchor_performance)


def normalize_model_name(name: str) -> str:
    """Normalize model name."""
    name = name.replace(':free', '')
    prefixes = ['tngtech/', 'google/', 'meta-llama/', 'mistralai/', 'qwen/', 'nvidia/', 'amazon/', 'openai/']
    for prefix in prefixes:
        name = name.replace(prefix, '')
    
    name_map = {
        'gemma-3-12b-it': 'gemma-3-12b',
        'gemma-3-27b-it': 'gemma-3-27b',
        'gemma-3-4b-it': 'gemma-3-4b',
        'llama-3.1-8b-instruct': 'llama-3-1-8b',
        'llama-3.3-70b-instruct': 'llama-3-3-70b',
    }
    return name_map.get(name, name)


def compute_anchor_statistics(
    similarity_data: List[Dict],
    anchor_performance: Dict[str, Dict[str, Dict]],
    top_k: int = DEFAULT_TOP_K,
    similarity_power: float = DEFAULT_SIMILARITY_POWER
) -> Dict[str, Dict[str, Dict]]:
    """
    Stage I: Compute weighted anchor statistics for each question.
    
    Returns:
        {question_id: {model_name: {'norm_acc': float, 'norm_cost': float, ...}}}
    """
    print("\n" + "="*70)
    print("Stage I: Computing anchor statistics")
    print("="*70)
    
    normalized_stats = {}
    epsilon = 1e-9
    
    for item in similarity_data:
        question_id = item['router_id']
        top_anchors = item['similar_anchors'][:top_k]
        
        # Collect weighted model performance
        model_performances = defaultdict(lambda: {'correct': 0.0, 'total_weight': 0.0, 'cost': 0.0})
        
        for anchor_info in top_anchors:
            anchor_id = anchor_info['anchor_id']
            similarity = anchor_info['similarity']
            
            if anchor_id not in anchor_performance:
                continue
            
            weight = similarity ** similarity_power
            
            for model_name, perf in anchor_performance[anchor_id].items():
                model_performances[model_name]['total_weight'] += weight
                model_performances[model_name]['correct'] += weight * (1 if perf['is_correct'] else 0)
                model_performances[model_name]['cost'] += weight * perf['cost']
        
        # Calculate weighted average accuracy and cost
        model_stats = {}
        for model_name, perf in model_performances.items():
            if perf['total_weight'] > 0:
                accuracy = perf['correct'] / perf['total_weight']
                avg_cost = perf['cost'] / perf['total_weight']
                model_stats[model_name] = {'accuracy': accuracy, 'cost': avg_cost}
        
        if not model_stats:
            continue
        
        # Normalize within question context
        accuracies = [s['accuracy'] for s in model_stats.values()]
        costs = [s['cost'] for s in model_stats.values()]
        
        acc_min, acc_max = min(accuracies), max(accuracies)
        log_costs = [np.log(c + epsilon) for c in costs]
        cost_min, cost_max = min(log_costs), max(log_costs)
        
        normalized_models = {}
        for model_name, stats in model_stats.items():
            # Normalize accuracy
            if acc_max - acc_min > epsilon:
                norm_acc = (stats['accuracy'] - acc_min) / (acc_max - acc_min)
            else:
                norm_acc = 0.5
            
            # Normalize cost (log scale)
            log_cost = np.log(stats['cost'] + epsilon)
            if cost_max - cost_min > epsilon:
                norm_cost = (log_cost - cost_min) / (cost_max - cost_min)
            else:
                norm_cost = 0.5
            
            normalized_models[model_name] = {
                'norm_acc': norm_acc,
                'norm_cost': norm_cost,
                'raw_acc': stats['accuracy'],
                'raw_cost': stats['cost']
            }
        
        normalized_stats[question_id] = normalized_models
    
    print(f"Computed statistics for {len(normalized_stats)} questions")
    return normalized_stats


def two_stage_routing(
    selection_data: Dict,
    normalized_stats: Dict[str, Dict[str, Dict]],
    alpha: float = 0.5,
    cost_sensitivity: float = DEFAULT_COST_SENSITIVITY,
    stage1_weight: float = DEFAULT_STAGE1_WEIGHT
) -> Dict:
    """
    Stage II: Online routing combining anchor stats and predictions.
    
    Returns:
        Dictionary with routing results
    """
    print("\n" + "="*70)
    print(f"Stage II: Online routing (alpha={alpha})")
    print("="*70)
    
    # Collect all predicted costs for normalization
    all_pred_costs = []
    for q in selection_data['questions']:
        for model_data in q.get('all_models', []):
            pred = model_data.get('predicted', {})
            if pred and 'cost' in pred:
                all_pred_costs.append(pred['cost'])
    
    if all_pred_costs:
        pred_cost_min, pred_cost_max = min(all_pred_costs), max(all_pred_costs)
    else:
        pred_cost_min, pred_cost_max = 0.0, 1.0
    
    # Dynamic adjustments based on alpha
    dynamic_cost_sensitivity = cost_sensitivity * (1.0 + (1.0 - alpha) * 2.0)
    dynamic_stage1_weight = stage1_weight * (0.5 + 0.5 * alpha)
    
    results = []
    total_cost = 0.0
    total_correct = 0
    total_questions = 0
    model_selection_count = defaultdict(int)
    
    # Prediction quality stats
    prediction_correct = 0
    prediction_total = 0
    
    for q in selection_data['questions']:
        question_id = q['question_id']
        all_models = q.get('all_models', [])
        
        if not all_models or question_id not in normalized_stats:
            continue
        
        question_stats = normalized_stats[question_id]
        
        best_model = None
        best_score = -float('inf')
        
        for model_data in all_models:
            model_name = model_data['model']
            pred = model_data.get('predicted', {})
            gt = model_data.get('ground_truth', {})
            
            # Count prediction quality
            if pred and gt:
                if pred.get('correctness') == gt.get('correctness'):
                    prediction_correct += 1
                prediction_total += 1
            
            # Stage I score (anchor statistics)
            stage1_score = 0.0
            if model_name in question_stats:
                stats = question_stats[model_name]
                norm_acc = stats['norm_acc']
                norm_cost = stats['norm_cost']
                
                cost_term = (1 - norm_cost) ** dynamic_cost_sensitivity
                stage1_score = alpha * norm_acc + (1 - alpha) * cost_term
            
            # Stage II score (predictions)
            stage2_score = 0.0
            if pred:
                pred_correctness = 1.0 if pred.get('correctness') == 'yes' else 0.0
                pred_cost = pred.get('cost', pred_cost_max)
                
                if pred_cost_max > pred_cost_min:
                    norm_pred_cost = (pred_cost - pred_cost_min) / (pred_cost_max - pred_cost_min)
                else:
                    norm_pred_cost = 0.5
                
                pred_cost_term = (1 - norm_pred_cost) ** dynamic_cost_sensitivity
                stage2_score = alpha * pred_correctness + (1 - alpha) * pred_cost_term
            
            # Combined score
            final_score = dynamic_stage1_weight * stage1_score + (1 - dynamic_stage1_weight) * stage2_score
            
            if final_score > best_score:
                best_score = final_score
                best_model = model_data
        
        # Fallback
        if best_model is None and all_models:
            best_model = all_models[0]
        
        if best_model:
            selected_model = best_model['model']
            gt = best_model.get('ground_truth', {})
            
            # Get actual cost
            actual_cost = gt.get('cost', 0.0)
            total_cost += actual_cost
            
            # Check if correct
            is_correct = gt.get('correctness') == 'yes'
            if is_correct:
                total_correct += 1
            
            total_questions += 1
            model_selection_count[selected_model] += 1
            
            results.append({
                'question_id': question_id,
                'selected_model': selected_model,
                'score': best_score,
                'ground_truth_correct': is_correct,
                'cost': actual_cost,
            })
    
    # Compute metrics
    accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_cost = total_cost / total_questions if total_questions > 0 else 0
    prediction_accuracy = prediction_correct / prediction_total if prediction_total > 0 else 0
    
    print(f"\nRouting Results:")
    print(f"  Total questions: {total_questions}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Average cost per question: ${avg_cost:.6f}")
    print(f"  Prediction accuracy: {prediction_accuracy*100:.2f}%")
    
    print(f"\nModel selection distribution:")
    for model, count in sorted(model_selection_count.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_questions if total_questions > 0 else 0
        print(f"  {model}: {count} ({pct:.1f}%)")
    
    return {
        'alpha': alpha,
        'total_questions': total_questions,
        'accuracy': accuracy,
        'total_cost': total_cost,
        'average_cost': avg_cost,
        'prediction_accuracy': prediction_accuracy,
        'model_distribution': dict(model_selection_count),
        'routing_decisions': results,
    }


def compute_pareto_curve(
    selection_data: Dict,
    normalized_stats: Dict[str, Dict[str, Dict]],
    alpha_values: np.ndarray = None,
    cost_sensitivity: float = DEFAULT_COST_SENSITIVITY,
    stage1_weight: float = DEFAULT_STAGE1_WEIGHT
) -> List[Tuple[float, float, float]]:
    """
    Compute Pareto frontier curve by varying alpha.
    
    Returns:
        List of (total_cost, accuracy, alpha) tuples
    """
    if alpha_values is None:
        alpha_values = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.10, ..., 1.0
    
    print("\n" + "="*70)
    print("Computing Pareto curve")
    print("="*70)
    
    curve_points = []
    
    for alpha in alpha_values:
        result = two_stage_routing(
            selection_data, normalized_stats,
            alpha=alpha,
            cost_sensitivity=cost_sensitivity,
            stage1_weight=stage1_weight
        )
        
        curve_points.append((
            result['total_cost'],
            result['accuracy'],
            alpha
        ))
    
    return curve_points


def main():
    parser = argparse.ArgumentParser(
        description="SCOPE Two-Stage Routing"
    )
    parser.add_argument(
        "--selection", "-s",
        type=str,
        required=True,
        help="Path to selection format JSON file (from scope_inference.py)"
    )
    parser.add_argument(
        "--similarity",
        type=str,
        help="Path to similarity JSON file. If not provided, will compute."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="routing_results.json",
        help="Output file for routing results"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["id", "ood"],
        default="id",
        help="Dataset type for anchor data"
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=None,
        help="Alpha value (0=cost-focused, 1=accuracy-focused). If not set, auto-select based on budget."
    )
    parser.add_argument(
        "--budget", "-b",
        type=float,
        default=None,
        help="Total budget in USD. Will auto-select optimal alpha to maximize accuracy within budget."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory for caching embeddings"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for embeddings"
    )
    parser.add_argument(
        "--alpha_steps",
        type=int,
        default=101,
        help="Number of alpha values to search (default: 101 for 0.01 precision)"
    )
    
    # Routing algorithm parameters
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of anchor questions to use in Stage I (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--similarity_power",
        type=float,
        default=DEFAULT_SIMILARITY_POWER,
        help=f"Power for similarity weighting in Stage I (default: {DEFAULT_SIMILARITY_POWER})"
    )
    parser.add_argument(
        "--cost_sensitivity",
        type=float,
        default=DEFAULT_COST_SENSITIVITY,
        help=f"Cost sensitivity parameter for routing (default: {DEFAULT_COST_SENSITIVITY})"
    )
    parser.add_argument(
        "--stage1_weight",
        type=float,
        default=DEFAULT_STAGE1_WEIGHT,
        help=f"Weight for Stage I anchor calibration vs Stage II prediction (default: {DEFAULT_STAGE1_WEIGHT})"
    )
    parser.add_argument(
        "--pricing_file",
        type=str,
        default=None,
        help="Path to custom pricing JSON file. If not provided, uses config/pricing.json or defaults."
    )
    parser.add_argument(
        "--pool_name", "-p",
        type=str,
        default=None,
        choices=["default", "full", "reasoning", "high_budget", "low_budget"],
        help="Filter routing to use only models from this pool. If not set, uses all models in selection file."
    )
    
    args = parser.parse_args()
    
    # Load custom pricing if specified
    global PRICING
    if args.pricing_file:
        PRICING = load_pricing(args.pricing_file)
        print(f"Using custom pricing from: {args.pricing_file}")
    
    print("="*70)
    print("SCOPE Two-Stage Routing")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Load selection data
    selection_data = load_selection_data(args.selection)
    
    # Filter models if pool_name is specified
    if args.pool_name:
        pool = get_pool(args.pool_name)
        pool_short_names = set()
        for model_id in pool:
            short_name = OPENROUTER_ID_TO_MODEL_NAME.get(model_id, model_id.split('/')[-1])
            pool_short_names.add(short_name)
        
        # Filter all_models in each question
        for q in selection_data['questions']:
            q['all_models'] = [m for m in q['all_models'] if m['model'] in pool_short_names]
        
        print(f"Filtering to {args.pool_name.upper()} pool ({len(pool_short_names)} models)")
        print(f"  Models: {sorted(pool_short_names)}")
    
    # Load or compute similarity data
    if args.similarity:
        similarity_data = load_similarity_data(args.similarity)
    else:
        print("\nNo similarity file provided. Computing similarities...")
        
        # Extract unique questions from selection data
        unique_questions = []
        seen_ids = set()
        for q in selection_data['questions']:
            qid = q['question_id']
            if qid not in seen_ids:
                seen_ids.add(qid)
                # We need the prompt - try to get from ground truth data
                # For now, we'll need the user to provide similarity file
                unique_questions.append({'id': qid, 'prompt': ''})
        
        print("  Warning: Cannot compute similarity without query prompts.")
        print("  Please provide --similarity file or use scope_inference.py first.")
        return
    
    # Load anchor performance
    anchor_performance = load_anchor_performance(args.dataset)
    
    # Display configuration
    print(f"\nRouting Configuration:")
    print(f"  Top-K anchors: {args.top_k}")
    print(f"  Similarity power: {args.similarity_power}")
    print(f"  Cost sensitivity: {args.cost_sensitivity}")
    print(f"  Stage I weight: {args.stage1_weight}")
    
    # Compute Stage I statistics
    normalized_stats = compute_anchor_statistics(
        similarity_data, anchor_performance,
        top_k=args.top_k,
        similarity_power=args.similarity_power
    )
    
    # Run routing
    if args.alpha is not None:
        # Single alpha value specified
        result = two_stage_routing(
            selection_data, normalized_stats,
            alpha=args.alpha,
            cost_sensitivity=args.cost_sensitivity,
            stage1_weight=args.stage1_weight
        )
        
        output = {
            'method': selection_data.get('method', 'SCOPE'),
            'dataset': args.dataset,
            'alpha': args.alpha,
            'config': {
                'top_k': args.top_k,
                'similarity_power': args.similarity_power,
                'cost_sensitivity': args.cost_sensitivity,
                'stage1_weight': args.stage1_weight,
            },
            'result': result
        }
        
        print("\n" + "="*70)
        print(f"Routing with alpha={args.alpha}")
        print("="*70)
        print(f"Accuracy: {result['accuracy']*100:.2f}%")
        print(f"Total cost: ${result['total_cost']:.4f}")
        
    elif args.budget is not None:
        # Budget specified - find optimal alpha
        print("\n" + "="*70)
        print(f"Finding optimal alpha for budget ${args.budget:.4f}")
        print("="*70)
        
        alpha_values = np.linspace(0, 1, args.alpha_steps)
        pareto_points = []
        
        print(f"Searching {len(alpha_values)} alpha values...")
        
        for alpha in alpha_values:
            result = two_stage_routing(
                selection_data, normalized_stats,
                alpha=alpha,
                cost_sensitivity=args.cost_sensitivity,
                stage1_weight=args.stage1_weight
            )
            pareto_points.append({
                'alpha': alpha,
                'accuracy': result['accuracy'],
                'total_cost': result['total_cost'],
                'average_cost': result['average_cost'],
                'routing_decisions': result['routing_decisions'],
                'model_distribution': result['model_distribution'],
            })
        
        # Find optimal alpha: maximize accuracy within budget
        valid_points = [p for p in pareto_points if p['total_cost'] <= args.budget]
        
        if not valid_points:
            # No solution within budget, use minimum cost
            best_point = min(pareto_points, key=lambda x: x['total_cost'])
            print(f"\n⚠️ Warning: Budget ${args.budget:.4f} is too low!")
            print(f"   Minimum possible cost: ${best_point['total_cost']:.4f}")
            print(f"   Using lowest cost configuration (alpha={best_point['alpha']:.2f})")
        else:
            # Find highest accuracy within budget
            best_point = max(valid_points, key=lambda x: x['accuracy'])
            print(f"\n✅ Found optimal alpha: {best_point['alpha']:.2f}")
            print(f"   Accuracy: {best_point['accuracy']*100:.2f}%")
            print(f"   Total cost: ${best_point['total_cost']:.4f} (within budget ${args.budget:.4f})")
        
        # Get full result for best alpha
        final_result = two_stage_routing(
            selection_data, normalized_stats,
            alpha=best_point['alpha'],
            cost_sensitivity=args.cost_sensitivity,
            stage1_weight=args.stage1_weight
        )
        
        output = {
            'method': selection_data.get('method', 'SCOPE'),
            'dataset': args.dataset,
            'budget': args.budget,
            'optimal_alpha': best_point['alpha'],
            'config': {
                'top_k': args.top_k,
                'similarity_power': args.similarity_power,
                'cost_sensitivity': args.cost_sensitivity,
                'stage1_weight': args.stage1_weight,
            },
            'result': final_result,
            'pareto_curve': [
                {'alpha': p['alpha'], 'accuracy': p['accuracy'], 
                 'total_cost': p['total_cost'], 'average_cost': p['average_cost']}
                for p in pareto_points
            ],
        }
        
        print(f"\nModel distribution:")
        for model, count in sorted(final_result['model_distribution'].items(), key=lambda x: -x[1]):
            pct = 100 * count / final_result['total_questions']
            print(f"  {model}: {count} ({pct:.1f}%)")
        
    else:
        # No alpha or budget - compute full Pareto curve
        alpha_values = np.linspace(0, 1, args.alpha_steps)
        pareto_points = []
        
        print(f"\nComputing Pareto curve with {len(alpha_values)} alpha values...")
        
        for alpha in alpha_values:
            result = two_stage_routing(
                selection_data, normalized_stats,
                alpha=alpha,
                cost_sensitivity=args.cost_sensitivity,
                stage1_weight=args.stage1_weight
            )
            pareto_points.append({
                'alpha': alpha,
                'accuracy': result['accuracy'],
                'total_cost': result['total_cost'],
                'average_cost': result['average_cost'],
            })
        
        output = {
            'method': selection_data.get('method', 'SCOPE'),
            'dataset': args.dataset,
            'config': {
                'top_k': args.top_k,
                'similarity_power': args.similarity_power,
                'cost_sensitivity': args.cost_sensitivity,
                'stage1_weight': args.stage1_weight,
            },
            'pareto_curve': pareto_points,
            'best_accuracy': max(p['accuracy'] for p in pareto_points),
            'lowest_cost': min(p['total_cost'] for p in pareto_points),
        }
        
        print("\n" + "="*70)
        print("Pareto Curve Summary")
        print("="*70)
        print(f"Best accuracy (alpha=1.0): {output['best_accuracy']*100:.2f}%")
        print(f"Lowest cost (alpha=0.0): ${output['lowest_cost']:.4f}")
        print("\nSample points (alpha → accuracy, cost):")
        for p in pareto_points[::10]:  # Print every 10th point
            print(f"  α={p['alpha']:.2f}: Acc={p['accuracy']*100:.2f}%, Cost=${p['total_cost']:.4f}")
    
    # Save output
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved results to: {args.output}")
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
