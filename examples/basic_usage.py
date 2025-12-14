#!/usr/bin/env python3
"""
Example: Basic usage of MedVLM-Probe

This example shows how to:
1. Configure and run the probe
2. Access results
3. Generate visualizations
"""

from medvlm_probe import MedVLMProbe, ProbeConfig
from medvlm_probe.visualizations import ProbeVisualizer


def main():
    # 1. Configure the probe
    config = ProbeConfig(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",  # Vision-language model
        num_samples_per_class=5,                   # Images per class
        probes_to_run=[                            # Tests to run
            "classification",
            "consistency", 
            "swap",
            "grounding"
        ],
        output_dir="./results"
    )
    
    # 2. Create probe instance
    probe = MedVLMProbe(config)
    
    # 3. Run all tests
    print(" Starting MedVLM-Probe...\n")
    results = probe.run()
    
    # 4. Print report
    results.print_report()
    
    # 5. Access scores programmatically
    scores = results.scores
    print(f"\n Programmatic access:")
    print(f"   Overall: {scores.overall:.1f}%")
    print(f"   Classification: {scores.by_type.get('classification', 0):.1f}%")
    print(f"   Consistency: {scores.by_type.get('consistency', 0):.1f}%")
    
    # 6. Save results
    results.save("./results")
    
    # 7. Generate visualizations
    print("\n Generating visualizations...")
    viz = ProbeVisualizer(results)
    viz.save_html("./results")
    
    # 8. Get DataFrame for custom analysis
    df = results.to_dataframe()
    print(f"\n Results DataFrame shape: {df.shape}")
    print(df.head())
    
    return results


if __name__ == "__main__":
    main()
