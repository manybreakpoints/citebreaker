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
for question in questions.questions:
    records.append({"inputs": {"question": question}})
dataset.merge_records(records)
print(f"Dataset now has {len(dataset.records)} records")




from mlflow.genai.datasets import search_datasets
datasets = search_datasets(
    experiment_ids=["citebreaker"],
    filter_string="tags.status = 'active' AND name LIKE '%history%'",
    order_by=["last_update_time DESC"],
    max_results=10,
)
for ds in datasets:
    print(f"{ds.name} ({ds.dataset_id}): {len(ds.records)} records")
