#!/usr/bin/env -S uv run --script
import re
import mlflow
import openai

from mlflow.genai import evaluate
from mlflow.genai.datasets import search_datasets
from mlflow.genai.scorers import Correctness, Guidelines, scorer

from bot import CitationBot

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("citebreaker")
mlflow.openai.autolog()

with open('data/History_of_Denmark.md', 'r') as file:
    knowledge_base = file.read()

@scorer
def contains_quotes(outputs: str, expectations: dict) -> float:
    return '"' in outputs

@scorer
def contains_single_citation(outputs: str, expectations: dict) -> float:
    return outputs.count('"') == 2

@scorer
def is_verbatim_citation(outputs: str, expectations: dict) -> float:
    if not contains_single_citation(outputs, expectations):
        return False
    match = re.search(r'"(.*?)"', outputs).group(1)[:-1]
    return match in knowledge_base

datasets = search_datasets(
    experiment_ids=["citebreaker"],
    filter_string="tags.status = 'active' AND name LIKE '%history%'",
    order_by=["last_update_time DESC"],
    max_results=10,
)
for ds in datasets:
    print(f"{ds.name} ({ds.dataset_id}): {len(ds.records)} records")
bot = CitationBot(temperature=0.5)
scorers = [
    contains_quotes,
    contains_single_citation,
    is_verbatim_citation,
]
results = evaluate(
    data=datasets[0],
    predict_fn=bot.answer,
    scorers=scorers,
    #model_id="citation-bot-v1",
)
