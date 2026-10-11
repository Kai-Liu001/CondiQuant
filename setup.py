from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
setup(
    name="condiquant", version="0.1.0",
    description="Condition Number Based Low-Bit Quantization for Image Super-Resolution",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["basicsr", "basicsr.*"]),
    python_requires=">=3.9", license="Apache-2.0",
    install_requires=[line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
                      if line.strip() and not line.startswith("#")],
)
