import emoji
import re

def lowercasing(text: str) -> str:
    return text.lower()


def change_emoji(text: str) -> str:
    return emoji.demojize(text)


def remove_html_tags(text: str) -> str:
    return re.sub("<.*?>", "", text)


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def change_user(text: str) -> str:
    """
    Change a username with '@' at the begining with @USER
    """
    TOKEN = "@USER"
    final_text = text
    results = re.findall(r"(@\S+)", final_text)
    results = set(results)
    results = sorted(results, reverse=True)
    for result in results:
        link = result
        final_text = re.sub(link, TOKEN, final_text, count=0)
    return final_text


def change_web_url(text: str) -> str:
    """
    Change a username with 'http' at the begining with HTTPURL
    """
    TOKEN = "HTTPURL"
    final_text = text
    results = re.findall(r"(http\S+)", final_text)
    results = set(results)
    results = sorted(results, reverse=True)
    for result in results:
        link = result
        final_text = re.sub(link, TOKEN, final_text, count=0)
    return final_text


def remove_username(text: str) -> str:
    """
    Remove username with an @ at front of the username
    """
    return re.sub(r"(@\S+)", "", text)


def remove_url(text: str) -> str:
    """
    Remove an url with a 'http' at front of the url
    """
    return re.sub(r"(http\S+)", "", text)


def remove_emoji(text: str) -> str:
    """
    Remove any emoji
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F1F2-\U0001F1F4"  # Macau flag
        "\U0001F1E6-\U0001F1FF"  # flags
        "\U0001F600-\U0001F64F"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U0001F1F2"
        "\U0001F1F4"
        "\U0001F620"
        "\u200d"
        "\u2640-\u2642"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)
