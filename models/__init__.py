"""Model loading utilities for MedVLM-Probe"""

import torch
from typing import Tuple, Any, Optional
from transformers import AutoProcessor


def load_model(
    model_id: str,
    torch_dtype: str = "float16",
    device_map: str = "auto",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
) -> Tuple[Any, Any]:
    """
    Load a Vision-Language Model and its processor.
    
    Args:
        model_id: HuggingFace model ID
        torch_dtype: Data type for model weights
        device_map: Device placement strategy
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        
    Returns:
        Tuple of (model, processor)
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)
    
    print(f"Loading model: {model_id}")
    
    # Detect model type and load accordingly
    if "qwen" in model_id.lower() and "vl" in model_id.lower():
        model, processor = _load_qwen_vl(model_id, dtype, device_map, min_pixels, max_pixels)
    elif "llava" in model_id.lower():
        model, processor = _load_llava(model_id, dtype, device_map)
    else:
        # Generic loading for other VLMs
        model, processor = _load_generic(model_id, dtype, device_map)
    
    print(f"✅ Model loaded on {next(model.parameters()).device}")
    return model, processor


def _load_qwen_vl(
    model_id: str,
    dtype: torch.dtype,
    device_map: str,
    min_pixels: int,
    max_pixels: int
) -> Tuple[Any, Any]:
    """Load Qwen2.5-VL model"""
    from transformers import Qwen2_5_VLForConditionalGeneration
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    
    return model, processor


def _load_llava(
    model_id: str,
    dtype: torch.dtype,
    device_map: str
) -> Tuple[Any, Any]:
    """Load LLaVA model"""
    from transformers import LlavaForConditionalGeneration
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor


def _load_generic(
    model_id: str,
    dtype: torch.dtype,
    device_map: str
) -> Tuple[Any, Any]:
    """Generic VLM loading"""
    from transformers import AutoModelForVision2Seq
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    return model, processor


class ModelWrapper:
    """Wrapper for unified inference across different VLM architectures"""
    
    def __init__(self, model, processor, model_id: str):
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.device = next(model.parameters()).device
        
    def generate(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.1
    ) -> str:
        """Generate response for image + text input"""
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if "qwen" in self.model_id.lower():
            return self._generate_qwen(image, prompt, max_new_tokens, temperature)
        elif "llava" in self.model_id.lower():
            return self._generate_llava(image, prompt, max_new_tokens, temperature)
        else:
            return self._generate_generic(image, prompt, max_new_tokens, temperature)
    
    def _generate_qwen(self, image, prompt, max_new_tokens, temperature) -> str:
        """Qwen-VL generation"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        response = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return response.strip()
    
    def _generate_llava(self, image, prompt, max_new_tokens, temperature) -> str:
        """LLaVA generation"""
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response.split("ASSISTANT:")[-1].strip()
    
    def _generate_generic(self, image, prompt, max_new_tokens, temperature) -> str:
        """Generic VLM generation"""
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)
