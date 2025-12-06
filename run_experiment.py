#!/usr/bin/env -S uv run --script
import mlflow

from mlflow.genai import evaluate
from mlflow.genai.datasets import search_datasets
from bot import CitationBot
from scorer import scorers


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("citebreaker")
mlflow.openai.autolog()

bot = CitationBot(temperature=0.5)

datasets = search_datasets(
    experiment_ids=["citebreaker"],
    filter_string="tags.status = 'active' AND name LIKE '%history%'",
    order_by=["last_update_time DESC"],
    max_results=10,
)

results = evaluate(
    data=datasets[0],
    predict_fn=bot.answer,
    scorers=scorers,
)
