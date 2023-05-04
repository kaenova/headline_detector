import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
requirements = (this_directory / "requirements.txt").read_text().split("\n")

setuptools.setup(
    name = "headline_detector",
    version = "1.0.2",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = "Kaenova Mahendra Auditama",
    author_email = "kaenova@gmail.com",
    description = "An Indonesian Headline Detection Python API.",
    classifiers = [
        "Programming Language :: Python :: 3",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.8",
    install_requires=requirements
)