#!/usr/bin/env python3
"""
Example: Create a custom probe

This example shows how to create your own probe test.
"""

from typing import List, Dict
from medvlm_probe import MedVLMProbe, ProbeConfig
from medvlm_probe.probes.base import BaseProbe, ProbeResult
from medvlm_probe.probes import PROBE_REGISTRY


class SeverityProbe(BaseProbe):
    """
    Custom probe: Test if model can assess severity of findings.
    """
    
    name = "severity"
    description = "Can the model assess the severity of findings?"
    
    PROMPT = """Analyze this chest X-ray and assess the severity of any findings.

Rate the severity as:
- NONE: No significant findings
- MILD: Minor abnormalities
- MODERATE: Significant findings requiring attention  
- SEVERE: Critical findings requiring immediate attention

SEVERITY: [NONE/MILD/MODERATE/SEVERE]
REASONING: [explain your assessment]"""
    
    def run(self, images_by_class: Dict[str, List]) -> List[ProbeResult]:
        results = []
        
        # Test on pneumonia images (should have some severity)
        pneumonia_images = images_by_class.get("pneumonia", [])[:3]
        print(f"\n   Testing severity assessment on {len(pneumonia_images)} pneumonia images...")
        
        for i, med_image in enumerate(pneumonia_images):
            response = self.model.generate(med_image.image, self.PROMPT)
            response_lower = response.lower()
            
            # Check if model gave a severity rating (not NONE for pneumonia)
            severity_keywords = ['mild', 'moderate', 'severe']
            has_severity = any(kw in response_lower for kw in severity_keywords)
            
            # Also check it's not saying "none" for obvious pneumonia
            said_none = 'none' in response_lower and 'severity: none' in response_lower
            
            passed = has_severity and not said_none
            
            result = ProbeResult(
                probe_name=self.name,
                test_type="severity",
                passed=passed,
                details={
                    "has_severity_rating": has_severity,
                    "incorrectly_said_none": said_none
                },
                response=response
            )
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"      {status} Image {i+1}: severity_given={has_severity}")
        
        # Test on normal images (should say NONE)
        normal_images = images_by_class.get("normal", [])[:2]
        print(f"\n   Testing severity assessment on {len(normal_images)} normal images...")
        
        for i, med_image in enumerate(normal_images):
            response = self.model.generate(med_image.image, self.PROMPT)
            response_lower = response.lower()
            
            # For normal images, should say NONE or no significant findings
            correctly_none = 'none' in response_lower or 'no significant' in response_lower
            
            result = ProbeResult(
                probe_name=self.name,
                test_type="severity",
                passed=correctly_none,
                details={
                    "correctly_identified_none": correctly_none
                },
                response=response
            )
            results.append(result)
            
            status = "✅" if correctly_none else "❌"
            print(f"      {status} Normal image {i+1}: correctly_none={correctly_none}")
        
        return results


def main():
    # Register the custom probe
    PROBE_REGISTRY["severity"] = SeverityProbe
    
    # Configure with custom probe
    config = ProbeConfig(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        num_samples_per_class=3,
        probes_to_run=["classification", "severity"]  # Include our custom probe
    )
    
    # Run
    probe = MedVLMProbe(config)
    results = probe.run()
    
    # Report
    results.print_report()
    results.save("./results/custom_probe")


if __name__ == "__main__":
    main()
