#!/usr/bin/env -S uv run --script
import mlflow
from mlflow.genai.datasets import create_dataset
from data import questions

mlflow.set_tracking_uri("http://127.0.0.1:5000")

dataset = create_dataset(
    name="history_of_Denmark_questions",
    experiment_id="citebreaker",
    tags={"version": "1.0", "team": "ME", "status": "active"},
)
records = []
for question in questions.questions[:5]:
    records.append({"inputs": {"question": question},
        "expectations": {}})
dataset.merge_records(records)
print(f"Dataset now has {len(dataset.records)} records")
