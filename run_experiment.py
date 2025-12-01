#!/usr/bin/env -S uv run --script
import mlflow
import openai

from mlflow.genai import evaluate
from mlflow.genai.datasets import search_datasets
from mlflow.genai.scorers import Correctness, Guidelines, scorer

from bot import CitationBot

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("citebreaker")
mlflow.openai.autolog()

@scorer
def contains_quotes(outputs: str, expectations: dict) -> float:
    """Check if response contains qoutes."""
    return '"' in outputs

datasets = search_datasets(
    experiment_ids=["citebreaker"],
    filter_string="tags.status = 'active' AND name LIKE '%history%'",
    order_by=["last_update_time DESC"],
    max_results=10,
)
bot = CitationBot()
scorers = [
    contains_quotes,
]
for ds in datasets:
    print(f"{ds.name} ({ds.dataset_id}): {len(ds.records)} records")
results = evaluate(
    data=datasets[0],
    predict_fn=bot.answer,
    scorers=scorers,
    model_id="citation-bot-v1",
)
metrics = results.metrics
detailed_results = results.tables["eval_results_table"]
