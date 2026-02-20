from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nexus-quant",
    version="2.0.0",
    author="Nexus Quantitative Architect",
    author_email="architect@nexus-quant.com",
    description="The world's most elegant, institutional-grade quantitative risk API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nexus-quant/nexus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "yfinance>=0.2.18",
    ],
    extras_require={
        # Highly precise optimization manifold packages
        "enterprise": ["cvxpy>=1.3.0", "mosek>=10.0.0"],
        "dev": ["pytest", "flake8", "black"],
    }
)
