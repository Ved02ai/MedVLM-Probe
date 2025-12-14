#!/usr/bin/env python3
"""
Example: Compare multiple VLM models

This example runs MedVLM-Probe on multiple models and compares results.
"""

from medvlm_probe import MedVLMProbe, ProbeConfig
import pandas as pd


MODELS_TO_COMPARE = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    # "llava-hf/llava-1.5-7b-hf",  # Uncomment if you have enough VRAM
    # "Qwen/Qwen2.5-VL-7B-Instruct",
]


def main():
    all_results = []
    
    for model_id in MODELS_TO_COMPARE:
        print(f"\n{'='*60}")
        print(f" Testing: {model_id}")
        print("="*60)
        
        config = ProbeConfig(
            model_id=model_id,
            num_samples_per_class=3,  # Use fewer for speed
        )
        
        probe = MedVLMProbe(config)
        results = probe.run()
        
        # Collect scores
        scores = results.scores
        all_results.append({
            "Model": model_id.split("/")[-1],
            "Overall": f"{scores.overall:.1f}%",
            "Classification": f"{scores.by_type.get('classification', 0):.1f}%",
            "Consistency": f"{scores.by_type.get('consistency', 0):.1f}%",
            "Swap": f"{scores.by_type.get('swap', 0):.1f}%",
            "Grounding": f"{scores.by_type.get('grounding', 0):.1f}%",
        })
        
        # Save individual results
        results.save(f"./results/{model_id.split('/')[-1]}")
    
    # Compare
    print("\n" + "="*60)
    print(" MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame(all_results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv("./results/model_comparison.csv", index=False)
    print("\n Comparison saved to ./results/model_comparison.csv")


if __name__ == "__main__":
    main()
