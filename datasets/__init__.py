"""Dataset loading utilities for MedVLM-Probe"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class MedicalImage:
    """A medical image with metadata"""
    image: any  # PIL Image
    label: str
    label_id: int
    source: str
    metadata: Dict = None


class MedicalDataset:
    """Base class for medical image datasets"""
    
    def __init__(
        self,
        dataset_id: str,
        split: str = "test",
        num_samples_per_class: int = 5,
        seed: int = 42
    ):
        self.dataset_id = dataset_id
        self.split = split
        self.num_samples = num_samples_per_class
        self.seed = seed
        
        self.images_by_class: Dict[str, List[MedicalImage]] = {}
        self.class_names: List[str] = []
        
    def load(self) -> "MedicalDataset":
        """Load the dataset - override in subclasses"""
        raise NotImplementedError
        
    def get_samples(self, class_name: str, n: Optional[int] = None) -> List[MedicalImage]:
        """Get n samples from a class"""
        n = n or self.num_samples
        images = self.images_by_class.get(class_name, [])
        if len(images) <= n:
            return images
        random.seed(self.seed)
        return random.sample(images, n)
    
    def get_all_test_samples(self) -> Dict[str, List[MedicalImage]]:
        """Get test samples for all classes"""
        return {
            class_name: self.get_samples(class_name)
            for class_name in self.class_names
        }


class ChestXrayDataset(MedicalDataset):
    """Chest X-ray Pneumonia dataset from HuggingFace"""
    
    def __init__(
        self,
        split: str = "test",
        num_samples_per_class: int = 5,
        seed: int = 42
    ):
        super().__init__(
            dataset_id="hf-vision/chest-xray-pneumonia",
            split=split,
            num_samples_per_class=num_samples_per_class,
            seed=seed
        )
        self.class_names = ["normal", "pneumonia"]
        
    def load(self) -> "ChestXrayDataset":
        """Load chest X-ray dataset"""
        print(f"Loading {self.dataset_id}...")
        
        dataset = load_dataset(self.dataset_id, split=self.split)
        label_names = dataset.features['label'].names
        
        self.images_by_class = {"normal": [], "pneumonia": []}
        
        for item in dataset:
            label_id = item['label']
            label_name = label_names[label_id].lower()
            
            # Map to our standard names
            if label_name in ['normal', '0']:
                class_name = 'normal'
            else:
                class_name = 'pneumonia'
            
            med_image = MedicalImage(
                image=item['image'],
                label=class_name,
                label_id=label_id,
                source=self.dataset_id
            )
            self.images_by_class[class_name].append(med_image)
        
        print(f" Loaded {sum(len(v) for v in self.images_by_class.values())} images")
        for name, images in self.images_by_class.items():
            print(f"   {name}: {len(images)}")
            
        return self


class NIHChestXrayDataset(MedicalDataset):
    """NIH Chest X-ray dataset (14 diseases)"""
    
    def __init__(
        self,
        split: str = "train",
        num_samples_per_class: int = 5,
        seed: int = 42,
        target_labels: List[str] = None
    ):
        super().__init__(
            dataset_id="alkzar90/NIH-Chest-X-ray-dataset",
            split=split,
            num_samples_per_class=num_samples_per_class,
            seed=seed
        )
        self.target_labels = target_labels or ["Pneumonia", "No Finding"]
        self.class_names = [l.lower().replace(" ", "_") for l in self.target_labels]
        
    def load(self) -> "NIHChestXrayDataset":
        """Load NIH Chest X-ray dataset"""
        print(f"Loading {self.dataset_id}...")
        
        dataset = load_dataset(self.dataset_id, split=self.split)
        
        self.images_by_class = {name: [] for name in self.class_names}
        
        for item in dataset:
            labels = item.get('labels', [])
            
            for target in self.target_labels:
                if target in labels:
                    class_name = target.lower().replace(" ", "_")
                    med_image = MedicalImage(
                        image=item['image'],
                        label=class_name,
                        label_id=self.class_names.index(class_name),
                        source=self.dataset_id,
                        metadata={"all_labels": labels}
                    )
                    self.images_by_class[class_name].append(med_image)
                    break
        
        print(f" Loaded images for target labels")
        for name, images in self.images_by_class.items():
            print(f"   {name}: {len(images)}")
            
        return self


def load_medical_dataset(
    dataset_name: str = "chest-xray-pneumonia",
    **kwargs
) -> MedicalDataset:
    """
    Factory function to load medical datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Loaded MedicalDataset instance
    """
    datasets = {
        "chest-xray-pneumonia": ChestXrayDataset,
        "nih-chest-xray": NIHChestXrayDataset,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    
    return datasets[dataset_name](**kwargs).load()
