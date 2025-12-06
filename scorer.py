import re
from mlflow.genai.scorers import scorer


with open("data/History_of_Denmark.md", "r") as file:
    REF = file.read()
REF_LOWER = REF.lower()
REF_LOWER_AZ = re.sub(r"[^a-z]", "", REF_LOWER)


@scorer
def contains_any_quotes(outputs: str, expectations: dict) -> bool:
    return '"' in outputs


@scorer
def contains_single_quote(outputs: str, expectations: dict = {}) -> bool:
    return outputs.count('"') == 2


def verbatim_quote_check(outputs: str, lower=False, az_only=False):
    if not contains_single_quote(outputs):
        return False
    quote = re.search(r'"(.*?)"', outputs).group(1)[:-1]
    if lower and az_only:
        return quote.lower() in REF_LOWER
    if lower and az_only:
        print(re.sub(r"[^a-z]", "", quote.lower()))
        return re.sub(r"[^a-z]", "", quote.lower()) in REF_LOWER_AZ
    if lower and az_only:
        return quote in REF
    raise NotImplementedError


@scorer
def is_verbatim_quote(outputs: str, expectations: dict) -> bool:
    """100% corect quote, string matching."""
    if not contains_single_quote(outputs):
        return False
    return verbatim_quote_check(outputs)


@scorer
def is_verbatim_lower_quote(outputs: str, expectations: dict) -> bool:
    """ "Correct quote regardsless of capitalisation."""
    if not contains_single_quote(outputs):
        return False
    return verbatim_quote_check(outputs, lower=True)


@scorer
def is_verbatim_alpha_quote(outputs: str, expectations: dict) -> bool:
    """Correct quote regarding only lower case a to z chars."""
    if not contains_single_quote(outputs):
        return False
    return verbatim_quote_check(outputs, lower=True, az_only=True)


@scorer
def has_more_than_5_chars_nonquote(outputs: str, expectations: dict) -> bool:
    """Evaluate if response also contain context besides the quote."""
    if not contains_single_quote(outputs):
        return False
    return len(re.sub(r'"[^"]*"', "", outputs)) > 5


@scorer
def percentage_nonquote(outputs: str, expectations: dict) -> float:
    """Return percentage of non-quote text to response text length"""
    if not contains_single_quote(outputs):
        return 0
    return 100 * len(re.sub(r'"[^"]*"', "", outputs)) / len(outputs)


scorers = [
    contains_any_quotes,
    contains_single_quote,
    is_verbatim_quote,
    is_verbatim_lower_quote,
    is_verbatim_alpha_quote,
    has_more_than_5_chars_nonquote,
    percentage_nonquote,
]
