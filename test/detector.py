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
