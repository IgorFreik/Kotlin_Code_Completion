"""Setup script for the Kotlin Code Completion package."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kotlin-code-completion",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for fine-tuning language models on Kotlin code completion tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kotlin-code-completion",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "kt-parse=kotlin_completion.data_parser:main",
            "kt-train=kotlin_completion.trainer:main",
            "kt-predict=kotlin_completion.predictor:main",
            "kt-evaluate=kotlin_completion.evaluator:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 