from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="medvlm-probe",
    version="0.1.0",
    author="Vedant Malik",
    author_email="vedantmac@gmail.com",
    description="Checking Visual Reasoning in Medical Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ved02ai/MedVLM-Probe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.24.0",
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "qwen-vl-utils>=0.0.2",
    ],
    extras_require={
        "viz": ["plotly>=5.18.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "all": [
            "plotly>=5.18.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medvlm-probe=medvlm_probe.cli:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "medical-ai",
        "vision-language-model",
        "radiology",
        "chest-xray",
        "vlm-evaluation",
        "medical-imaging",
    ],
)
