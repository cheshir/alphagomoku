from setuptools import setup, find_packages

setup(
    name="alphagomoku",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.26.0",
        "lmdb>=1.4.0",
        "zarr>=2.14.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "pytest>=7.4.0",
        "black>=23.0.0",
        "isort>=5.12.0",
    ],
)