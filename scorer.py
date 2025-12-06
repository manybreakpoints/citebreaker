import re
from mlflow.genai.scorers import scorer


with open("data/History_of_Denmark.md", "r") as file:
    REF = file.read()
REF_LOWER = REF.lower()
REF_LOWER_AZ = re.sub(r"[^a-z]", "", REF_LOWER)


@scorer
def contains_any_quotes(outputs: str, expectations: dict) -> float:
    return '"' in outputs


@scorer
def contains_single_quote(outputs: str, expectations: dict = {}) -> float:
    return outputs.count('"') == 2


def verbatim_quote_check(outputs: str, lower=False, az_only=False):
    if not contains_single_quote(outputs):
        return False

    # quote_lower = knowledge_base.lower()
    # knowledge_base_az = re.sub(r'[^a-z]', '', knowledge_base_lower)

    quote = re.search(r'"(.*?)"', outputs).group(1)[:-1]
    return quote in REF


@scorer
def is_verbatim_quote(outputs: str, expectations: dict) -> float:
    return verbatim_quote_check(outputs)


@scorer
def has_explanation(outputs: str, expectations: dict) -> float:
    pass
    # TODO


scorers = [
    contains_any_quotes,
    contains_single_quote,
    is_verbatim_quote,
]
