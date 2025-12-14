"""Grounding probe - tests if model can localize findings"""

from typing import List, Dict
from .base import BaseProbe, ProbeResult


class GroundingProbe(BaseProbe):
    """Test if model can describe specific findings with anatomical locations"""
    
    name = "grounding"
    description = "Can the model describe findings with anatomical locations?"
    
    PROMPT = """Describe the specific findings in this chest X-ray.

Include:
- Location of any abnormalities (left/right lung, upper/middle/lower zone)
- Type of opacity (consolidation, infiltrate, ground-glass, etc.)
- Any other relevant findings

FINDINGS:"""
    
    LOCATION_KEYWORDS = [
        'left', 'right', 'bilateral', 'upper', 'lower', 'middle',
        'base', 'apex', 'lobe', 'zone', 'perihilar', 'peripheral'
    ]
    
    FINDING_KEYWORDS = [
        'consolidation', 'infiltrate', 'opacity', 'effusion',
        'ground-glass', 'nodule', 'mass', 'atelectasis',
        'edema', 'cardiomegaly', 'pneumothorax'
    ]
    
    def run(self, images_by_class: Dict[str, List]) -> List[ProbeResult]:
        """Run grounding test on abnormal images"""
        results = []
        
        # Test on pneumonia images (should have findings to describe)
        pneumonia_images = images_by_class.get("pneumonia", [])[:3]
        print(f"\n   Testing visual grounding on {len(pneumonia_images)} pneumonia images...")
        
        for i, med_image in enumerate(pneumonia_images):
            response = self.model.generate(med_image.image, self.PROMPT)
            response_lower = response.lower()
            
            # Check for location mentions
            has_location = any(kw in response_lower for kw in self.LOCATION_KEYWORDS)
            
            # Check for finding type mentions
            has_finding = any(kw in response_lower for kw in self.FINDING_KEYWORDS)
            
            # Grounded = has both location AND finding type
            grounded = has_location and has_finding
            
            result = ProbeResult(
                probe_name=self.name,
                test_type="grounding",
                passed=grounded,
                details={
                    "has_location": has_location,
                    "has_finding_type": has_finding,
                    "grounded": grounded
                },
                response=response
            )
            results.append(result)
            
            status = "✅" if grounded else "❌"
            print(f"      {status} Image {i+1}: location={has_location}, finding={has_finding}")
        
        # Also test on normal images (should say normal/clear)
        normal_images = images_by_class.get("normal", [])[:2]
        print(f"\n   Testing grounding on {len(normal_images)} normal images...")
        
        for i, med_image in enumerate(normal_images):
            response = self.model.generate(med_image.image, self.PROMPT)
            response_lower = response.lower()
            
            # For normal images, should mention "normal", "clear", etc.
            normal_keywords = ['normal', 'clear', 'unremarkable', 'no abnormal', 'negative']
            correctly_normal = any(kw in response_lower for kw in normal_keywords)
            
            result = ProbeResult(
                probe_name=self.name,
                test_type="grounding",
                passed=correctly_normal,
                details={
                    "image_type": "normal",
                    "correctly_identified_normal": correctly_normal
                },
                response=response
            )
            results.append(result)
            
            status = "✅" if correctly_normal else "❌"
            print(f"      {status} Normal image {i+1}: identified as normal={correctly_normal}")
        
        return results
