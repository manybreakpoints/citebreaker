import os
import openai
import mlflow
from mlflow.genai.scorers import Correctness, Guidelines

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("genai-evaluation")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dataset = [
    {
        "inputs": {"question": "Can MLflow manage prompts?"},
        "expectations": {"expected_response": "Yes!"},
    },
    {
        "inputs": {"question": "Can MLflow create a taco for my lunch?"},
        "expectations": {
            "expected_response": "No, unfortunately, MLflow is not a taco maker."
        },
    },
]

def predict_fn(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        Correctness(),
        Guidelines(name="is_english", guidelines="The answer must be in English"),
    ],
)
