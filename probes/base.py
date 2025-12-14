"""Base probe class"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Result of a single probe test"""
    probe_name: str
    test_type: str
    passed: bool
    details: Dict[str, Any]
    response: str = ""


class BaseProbe(ABC):
    """Base class for all probing tests"""
    
    name: str = "base"
    description: str = "Base probe"
    
    def __init__(self, model_wrapper, config=None):
        self.model = model_wrapper
        self.config = config or {}
        
    @abstractmethod
    def run(self, images_by_class: Dict[str, List]) -> List[ProbeResult]:
        """
        Run the probe on the provided images.
        
        Args:
            images_by_class: Dict mapping class names to lists of MedicalImage
            
        Returns:
            List of ProbeResult objects
        """
        pass
    
    def extract_classification(self, response: str) -> str:
        """Extract classification from model response"""
        response_lower = response.lower()
        
        pneumonia_keywords = [
            'pneumonia', 'infection', 'infiltrate', 'consolidation', 
            'opacity', 'abnormal', 'pathology'
        ]
        normal_keywords = [
            'normal', 'healthy', 'clear', 'no abnormal', 
            'unremarkable', 'no evidence', 'negative'
        ]
        
        p_score = sum(1 for kw in pneumonia_keywords if kw in response_lower)
        n_score = sum(1 for kw in normal_keywords if kw in response_lower)
        
        if p_score > n_score:
            return 'pneumonia'
        elif n_score > p_score:
            return 'normal'
        return 'uncertain'
