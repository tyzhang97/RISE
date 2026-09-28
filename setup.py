from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
VERSION_NS = {}
exec((ROOT / "rise" / "_version.py").read_text(encoding="utf-8"), VERSION_NS)

CORE_REQUIREMENTS = [
    "brainspace==0.1.20",
    "h5py==3.11.0",
    "joblib==1.4.2",
    "matplotlib==3.7.5",
    "nibabel==5.2.1",
    "nilearn==0.10.4",
    "nipype==1.8.6",
    "numpy==1.24.4",
    "pandas==1.5.3",
    "Pillow==10.2.0",
    "pyarrow==17.0.0",
    "scikit-learn==1.3.2",
    "scipy==1.10.1",
    "tqdm==4.67.1",
]

NEURAL_REQUIREMENTS = ["tables==3.8.0", "torch==2.4.1"]

ANALYSIS_REQUIREMENTS = [
    "ipython==8.12.3",
    "neuromaps==0.0.5",
    "seaborn==0.13.2",
    "statsmodels==0.14.1",
    "wordcloud==1.9.4",
]

NOTEBOOK_REQUIREMENTS = ["ipykernel==6.29.5", "jupyterlab==3.6.8"]


setup(
    name="risebrain",
    version=VERSION_NS["__version__"],
    description=(
        "Regional Identity Signature Expression (RISE), a quantifiable, "
        "fine-grained, and generalizable framework for mapping individualized "
        "regional identity across the cortex"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=find_packages(include=["rise", "rise.*"]),
    include_package_data=True,
    package_data={
        "rise": [
            "feature_resources/*",
            "hctsa/*",
            "matlab/*.m",
            "vis/resources/*",
            "vis/resources/**/*",
        ]
    },
    exclude_package_data={"rise": ["atlas_repo/*", "atlas_repo/**/*"]},
    install_requires=CORE_REQUIREMENTS + NEURAL_REQUIREMENTS,
    extras_require={
        "neural": NEURAL_REQUIREMENTS,
        "analysis": ANALYSIS_REQUIREMENTS,
        "notebooks": NOTEBOOK_REQUIREMENTS,
        "all": ANALYSIS_REQUIREMENTS + NOTEBOOK_REQUIREMENTS,
    },
)

# MATLAB, MATLAB Engine for Python, and hctsa are external runtime dependencies.
# They must be installed separately and are intentionally absent above.
