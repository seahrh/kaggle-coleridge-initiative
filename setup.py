from setuptools import setup, find_packages

__version__ = "1.0"
setup(
    name="kaggle-coleridge-initiative",
    version=__version__,
    python_requires="~=3.7",
    install_requires=[
        "datatable==0.11.1",
        "gcsfs==0.7.2",
        "google-cloud-logging==1.15.1",
        "google-cloud-storage==1.30.0",
        "lightgbm==3.2.0",
        "optuna==2.7.0",
        "pandas==1.2.3",
        "pyarrow==3.0.0",
        "scikit-learn==0.24.1",
        "sentence-transformers==1.0.4",
        "symspellpy==6.7.0",
        "tensorflow==2.4.1",
    ],
    extras_require={
        "tests": [
            "black==20.8b1",
            "mypy==0.812",
            "pytest==6.2.3",
            "pytest-cov==2.11.1",
        ],
        "notebook": ["jupyterlab==1.2.16", "seaborn==0.11.1", "tqdm==4.59.0"],
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="Kaggle Coleridge Initiative competition",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/kaggle-coleridge-initiative",
)
