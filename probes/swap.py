"""Swap probe - tests if model changes answer for different images"""

from typing import List, Dict
from .base import BaseProbe, ProbeResult


class SwapProbe(BaseProbe):
    """Test if model gives different answers for different images"""
    
    name = "swap"
    description = "Does the model change its answer when shown a different image?"
    
    PROMPT = """Classify this chest X-ray as NORMAL or PNEUMONIA.
CLASSIFICATION: [NORMAL or PNEUMONIA]"""
    
    def run(self, images_by_class: Dict[str, List]) -> List[ProbeResult]:
        """Run swap test - compare responses for different images"""
        results = []
        
        normal_images = images_by_class.get("normal", [])
        pneumonia_images = images_by_class.get("pneumonia", [])
        
        num_tests = min(3, len(normal_images), len(pneumonia_images))
        print(f"\n   Running {num_tests} swap tests...")
        
        for i in range(num_tests):
            # Get prediction for normal image
            normal_response = self.model.generate(
                normal_images[i].image, self.PROMPT
            )
            pred_normal = self.extract_classification(normal_response)
            
            # Get prediction for pneumonia image  
            pneumonia_response = self.model.generate(
                pneumonia_images[i].image, self.PROMPT
            )
            pred_pneumonia = self.extract_classification(pneumonia_response)
            
            # Model should give different answers
            changed = (pred_normal != pred_pneumonia)
            
            result = ProbeResult(
                probe_name=self.name,
                test_type="swap",
                passed=changed,
                details={
                    "pred_normal_image": pred_normal,
                    "pred_pneumonia_image": pred_pneumonia,
                    "changed_appropriately": changed
                },
                response=f"Normal: {normal_response}\n---\nPneumonia: {pneumonia_response}"
            )
            results.append(result)
            
            status = "✅" if changed else "❌"
            print(f"      {status} Pair {i+1}: Normal→{pred_normal}, Pneumonia→{pred_pneumonia}")
        
        return results
