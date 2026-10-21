import sys
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
# Read requirements from the file
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

# Check if the user specified the CPU option
if "--cpu" in sys.argv:
    # Remove the gpu packages
    try:
        to_drop = [x for x in install_requires if "nvidia" in x or "cuda" in x]
        for x in to_drop:
            install_requires.remove(x)
    except ValueError:
        pass
    # Remove the --cpu option from sys.argv so setuptools doesn't get confused
    sys.argv.remove("--cpu")

setup(
    name="pifold",
    python_requires=">=3.11",
    version="1.0.0",
    description="a Physics-Informed policy-based co-foding model for biomolecular structure prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Newturing Institute",
    url="https://github.com/newturing/pifold",
    author_email="taitdang@stanford.edu",
    packages=find_packages(
        exclude=(
            "assets",
            "benchmark",
            "*.egg-info",
        )
    ),
    include_package_data=True,
    package_data={
        "pifold": ["model/layer_norm/kernel/*"],
    },
    install_requires=install_requires,
    license="CC BY 4.0",
    platforms="manylinux1",
    entry_points={
        "console_scripts": [
            "pifold = runner.batch_inference:pifold_cli",
        ],
    },
)
