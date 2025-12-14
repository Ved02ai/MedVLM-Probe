"""Classification probe - tests basic diagnostic ability"""

from typing import List, Dict
from .base import BaseProbe, ProbeResult


class ClassificationProbe(BaseProbe):
    """Test if model can correctly classify images"""
    
    name = "classification"
    description = "Can the model correctly classify normal vs abnormal?"
    
    PROMPT = """You are an expert radiologist. Analyze this chest X-ray.
Classify as NORMAL or PNEUMONIA.

CLASSIFICATION: [NORMAL or PNEUMONIA]
CONFIDENCE: [high/medium/low]
FINDINGS: [what you see]"""
    
    def run(self, images_by_class: Dict[str, List]) -> List[ProbeResult]:
        """Run classification test on all classes"""
        results = []
        
        for class_name, images in images_by_class.items():
            print(f"\n   Testing {class_name} images...")
            
            for i, med_image in enumerate(images):
                response = self.model.generate(med_image.image, self.PROMPT)
                predicted = self.extract_classification(response)
                correct = (predicted == class_name)
                
                result = ProbeResult(
                    probe_name=self.name,
                    test_type="classification",
                    passed=correct,
                    details={
                        "true_label": class_name,
                        "predicted": predicted,
                        "image_index": i
                    },
                    response=response
                )
                results.append(result)
                
                status = "✅" if correct else "❌"
                print(f"      {status} Image {i+1}: true={class_name}, pred={predicted}")
        
        return results
