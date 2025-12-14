#  MedVLM-Probe

**Does your Medical Vision-Language Model actually see, or just guess?**

[![PyPI version](https://badge.fury.io/py/medvlm-probe.svg)](https://pypi.org/project/medvlm-probe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 What is MedVLM-Probe?

MedVLM-Probe is a **method for checking visual reasoning** in medical Vision-Language Models. It runs systematic probes to evaluate whether your VLM actually understands medical images or just pattern-matches.

### Probing Checks:

| Probe | What it Checks |
|-------|---------------|
| **Classification** | Can it correctly identify normal vs abnormal? |
| **Consistency** | Does it trust the image over contradictory text? |
| **Swap** | Does it change answers for different images? |
| **Grounding** | Can it localize findings anatomically? |

---

## 📦 Installation

```bash
# From PyPI
pip install medvlm-probe

# From source
git clone https://github.com/Ved02ai/MedVLM-Probe.git
cd MedVLM-Probe
pip install -e .

# With visualizations
pip install medvlm-probe[viz]
```

---

## 🚀 Quick Start

### Python API

```python
from medvlm_probe import MedVLMProbe, ProbeConfig

# Configure
config = ProbeConfig(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    num_samples_per_class=5
)

# Run
probe = MedVLMProbe(config)
results = probe.run()

# Report
results.print_report()
results.save("./results")
```

### CLI

```bash
# Quick evaluation
medvlm-probe run --quick

# Custom model
medvlm-probe run --model Qwen/Qwen2.5-VL-7B-Instruct --samples 10

# Specific probes only
medvlm-probe run --probes classification consistency
```

### One-liner

```python
from medvlm_probe import MedVLMProbe
MedVLMProbe.quick_eval("Qwen/Qwen2.5-VL-3B-Instruct")
```

---

## 📊 Example Output

```
📊 MEDVLM-PROBE REPORT
============================================================

Model: Qwen/Qwen2.5-VL-3B-Instruct
📁 Dataset: hf-vision/chest-xray-pneumonia

🎯 Overall Score: 75.0% (12/16 passed)

📋 Scores by Check Type:
   ✅ Classification: 80.0%
   ⚠️ Consistency: 66.7%
   ✅ Swap: 100.0%
   ⚠️ Grounding: 66.7%

Classification Accuracy:
   Normal: 80.0%
   Pneumonia: 80.0%
```

---

## 🔧 Configuration

```python
from medvlm_probe import ProbeConfig

config = ProbeConfig(
    # Model
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="float16",
    device_map="auto",
    
    # Dataset
    dataset_id="hf-vision/chest-xray-pneumonia",
    num_samples_per_class=5,
    
    # Probes
    probes_to_run=["classification", "consistency", "swap", "grounding"],
    
    # Output
    output_dir="./results",
    save_responses=True,
    
    # Reproducibility
    seed=42
)
```

---

## Supported Models

- ✅ Qwen2.5-VL (3B, 7B, 72B)
- ✅ LLaVA (1.5, 1.6)
- ✅ Any HuggingFace VLM with image input

```python
# Check different models
config = ProbeConfig(model_id="llava-hf/llava-1.5-7b-hf")
```

---

## 📈 Visualizations

```python
from medvlm_probe.visualizations import ProbeVisualizer

viz = ProbeVisualizer(results)
viz.plot_all()           # Show all plots
viz.save_html("./plots") # Save as HTML
```

Available plots:
- Gauge chart (overall score)
- Bar chart (by check type)
- Confusion matrix
- Pass/fail summary

---

## 🔬 Create Custom Probes

```python
from medvlm_probe.probes.base import BaseProbe, ProbeResult
from medvlm_probe.probes import PROBE_REGISTRY

class MyCustomProbe(BaseProbe):
    name = "my_probe"
    description = "My custom check"
    
    def run(self, images_by_class):
        results = []
        # Your logic here
        return results

# Register
PROBE_REGISTRY["my_probe"] = MyCustomProbe

# Use
config = ProbeConfig(probes_to_run=["classification", "my_probe"])
```

---

## 📁 Project Structure

```
MedVLM-Probe/
├── medvlm_probe/
│   ├── __init__.py          # Main exports
│   ├── probe.py             # MedVLMProbe class
│   ├── config.py            # Configuration
│   ├── results.py           # Results handling
│   ├── cli.py               # CLI interface
│   ├── models/              # Model loading
│   ├── datasets/            # Dataset loading
│   ├── probes/              # Probe implementations
│   │   ├── base.py
│   │   ├── classification.py
│   │   ├── consistency.py
│   │   ├── swap.py
│   │   └── grounding.py
│   └── visualizations/      # Plotting
├── examples/
│   ├── basic_usage.py
│   ├── compare_models.py
│   └── custom_probe.py
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 📖 Citation

```bibtex
@software{medvlm_probe,
  author = {Vedant Malik},
  title = {MedVLM-Probe: Checking Visual Reasoning in Medical Vision-Language Models},
  year = {2025},
  publisher = {MedReasoning Lab},
  url = {https://github.com/Ved02ai/MedVLM-Probe}
}
```

---

## 🔗 Links

- **Author**: Vedant Malik, Parkland High School, Allentown, PA
- **HuggingFace MedReasoning-Lab**: [MedReasoning Lab](https://huggingface.co/medreasoning-lab)
---

## Contributing

PRs welcome! Ideas:

- [ ] Add CT/MRI datasets
- [ ] Support more VLM architectures
- [ ] Add localization/bounding box checks
- [ ] Multi-finding detection
- [ ] Report generation probes

---

## 📜 License

MIT License - see [LICENSE](LICENSE)

---
