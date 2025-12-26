"""Blackjack Reinforcement Learning Project."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="blackjack-rl",
    version="1.0.0",
    author="Blackjack RL Team",
    description="Reinforcement Learning for Blackjack with Card Counting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/blackjack-rl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.4.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "blackjack-train-naive=scripts.train_all_naive:main",
            "blackjack-train-counting=scripts.train_all_counting:main",
            "blackjack-evaluate=scripts.evaluate_all:main",
            "blackjack-report=scripts.generate_report:main",
        ],
    },
)
