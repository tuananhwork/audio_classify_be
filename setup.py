from setuptools import setup, find_packages

setup(
    name="audio-classifier",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "numpy>=1.19.0",
        "librosa>=0.8.0",
        "tensorflow>=2.5.0",
    ],
) 