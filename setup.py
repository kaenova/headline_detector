import setuptools

setuptools.setup(
    name = "headline_detector",
    version = "1.0.0",
    author = "Kaenova Mahendra Auditama",
    author_email = "kaenova@gmail.com",
    description = "An Indonesian Headline Detection Python API.",
    classifiers = [
        "Programming Language :: Python :: 3",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.8",
    install_requires=[
        "transformers",
        "torch",
        "lightning",
        "pytorch-nlp",
        "huggingface_hub",
        "numpy",
        "emoji",
        "NDETCStemmer-kaenova"
    ]
)