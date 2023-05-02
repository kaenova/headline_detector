# headline_detector

_Indonesian Headline Detection Python API_

This is a Python library that provides APIs for detecting headlines in textual data, especially on social media platforms such as Twitter. The library utilizes a model that has been developed and trained on a dataset of Twitter posts containing both headline and non-headline texts, with the assistance of journalism professionals to ensure the data quality.

```sh
$ pip install headline-detector
```

## Available scenario and the performance

| Model        | Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 | Scenario 5 | Scenario 6 |
| ------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Fasttext     | 0.8766     | 0.8714     | 0.8793     | 0.8714     | 0.8714     | 0.8661     |
| CNN          | 0.9081     | 0.9081     | 0.8950     | 0.8898     | 0.8950     | 0.8898     |
| IndoBERTweet | 0.9895     | 0.9921     | 0.9738     | 0.9580     | 0.9843     | 0.9685     |

All meassured in accuracy

### Model Throughput

| Model        | Throughput (± Text/seconds) |
| ------------ | --------------------------- |
| IndoBERTweet | ±1.3                        |
| CNN          | ±281.60                     |
| Fasttext     | ±2048.41                    |

Tested on Intel i7-6700k and 32GB of RAM.

## Usage

Output either 0 (non-headline) and 1 (headline)

```python
from headline_detector import FasttextDetector, IndoBERTweetDetector, CNNDetector

detector = FasttextDetector.load_from_scenario(1)
data = detector.predict_text(
    [
        "nama kamu siapa?",
        "Kapolda Jatim Teddy Minahasa Dikabarkan Ditangkap Terkait Narkoba  https://t.co/LD9X6VFaUR",
    ]
)
print(data)  # output: [0, 1]

detector = CNNDetector.load_from_scenario(3)
data = detector.predict_text(
    [
        "nama kamu siapa?",
        "Kapolda Jatim Teddy Minahasa Dikabarkan Ditangkap Terkait Narkoba  https://t.co/LD9X6VFaUR",
    ]
)
print(data)  # output: [0, 1]

detector = IndoBERTweetDetector.load_from_scenario(5)
data = detector.predict_text(
    [
        "nama kamu siapa?",
        "Kapolda Jatim Teddy Minahasa Dikabarkan Ditangkap Terkait Narkoba  https://t.co/LD9X6VFaUR",
    ]
)
print(data)  # output: [0, 1]

# 0 is non-headline
# 1 is headline
```

## Paper

Coming soon.
