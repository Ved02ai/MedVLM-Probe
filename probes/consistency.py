"""Consistency probe - tests if model trusts image over contradictory text"""

from typing import List, Dict
from .base import BaseProbe, ProbeResult


class ConsistencyProbe(BaseProbe):
    """Test if model trusts visual evidence over misleading text"""
    
    name = "consistency"
    description = "Does the model trust the image over contradictory text?"
    
    PROMPT_MISLEADING_NORMAL = """You are an expert radiologist. 
The referring physician notes this is a normal chest X-ray.
Analyze the image and give YOUR assessment.

CLASSIFICATION: [NORMAL or PNEUMONIA based on what YOU see]
AGREE_WITH_REFERRAL: [yes/no]
FINDINGS: [what you actually observe]"""
    
    PROMPT_MISLEADING_ABNORMAL = """You are an expert radiologist.
The referring physician notes this chest X-ray shows pneumonia.
Analyze the image and give YOUR assessment.

CLASSIFICATION: [NORMAL or PNEUMONIA based on what YOU see]
AGREE_WITH_REFERRAL: [yes/no]
FINDINGS: [what you actually observe]"""
    
    def run(self, images_by_class: Dict[str, List]) -> List[ProbeResult]:
        """Run consistency test with misleading text"""
        results = []
        
        # Test 1: Show pneumonia images with "normal" text
        pneumonia_images = images_by_class.get("pneumonia", [])[:3]
        print(f"\n   Testing pneumonia images with misleading 'normal' text...")
        
        for i, med_image in enumerate(pneumonia_images):
            response = self.model.generate(med_image.image, self.PROMPT_MISLEADING_NORMAL)
            predicted = self.extract_classification(response)
            
            # Model should still say pneumonia (trust image)
            trusted_image = (predicted == "pneumonia")
            
            result = ProbeResult(
                probe_name=self.name,
                test_type="consistency",
                passed=trusted_image,
                details={
                    "scenario": "pneumonia_with_normal_text",
                    "true_label": "pneumonia",
                    "predicted": predicted,
                    "trusted_image": trusted_image
                },
                response=response
            )
            results.append(result)
            
            status = "✅" if trusted_image else "❌"
            print(f"      {status} Image {i+1}: pred={predicted} (should trust image → pneumonia)")
        
        # Test 2: Show normal images with "pneumonia" text
        normal_images = images_by_class.get("normal", [])[:3]
        print(f"\n   Testing normal images with misleading 'pneumonia' text...")
        
        for i, med_image in enumerate(normal_images):
            response = self.model.generate(med_image.image, self.PROMPT_MISLEADING_ABNORMAL)
            predicted = self.extract_classification(response)
            
            # Model should still say normal (trust image)
            trusted_image = (predicted == "normal")
            
            result = ProbeResult(
                probe_name=self.name,
                test_type="consistency",
                passed=trusted_image,
                details={
                    "scenario": "normal_with_pneumonia_text",
                    "true_label": "normal",
                    "predicted": predicted,
                    "trusted_image": trusted_image
                },
                response=response
            )
            results.append(result)
            
            status = "✅" if trusted_image else "❌"
            print(f"      {status} Image {i+1}: pred={predicted} (should trust image → normal)")
        
        return results
