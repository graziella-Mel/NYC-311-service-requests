from setuptools import setup, find_packages

setup(
    name="311_preprocessor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib"
    ],
    author="Graziella Salameh",
    description="A package to preprocess 311 service request data for machine learning.",
    url="https://github.com/graalop/311_preprocessor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
