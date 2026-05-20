from setuptools import setup, find_packages

setup(
    name="right-whale-detector",
    version="0.1.0",
    description="CNN + TCN right whale acoustic detector",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchaudio",
        "numpy",
        "pandas",
        "matplotlib",
        "librosa",
        "scikit-learn",
        "tqdm",
        "pyyaml",
        "soundfile",
    ],
)
