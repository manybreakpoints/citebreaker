import mlflow
import openai

mlflow.openai.autolog()


class CitationBot:
    def __init__(self, temperature=0.3):
        self.client = openai.OpenAI()
        self.temperature = temperature
        with open("data/History_of_Denmark.md", "r") as file:
            self.knowledge_base = file.read()

    def _get_context(self, question: str) -> str:  # TODO should be rag?
        return self.knowledge_base

    @mlflow.trace
    def answer(self, question: str) -> str:
        context = self._get_context(question)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            # model="gpt-5.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a an citation bot that creates a brief answers with only "
                    'one verbatum citation in double quote symbols (like: "the relevant quote"), '
                    "from the provided context. Quote in full, don't stich "
                    "sentences together. Give context/explanation and a the qoute.",
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}",
                },
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
