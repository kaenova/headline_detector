import setuptools

setuptools.setup(
    name = "headline_detector",
    version = "0.0.1",
    author = "Kaenova Mahendra Auditama",
    author_email = "kaenova@gmail.com",
    description = "Indonesian Headline Detection using Fasttext, CNN, or IndoBERTweet models.",
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