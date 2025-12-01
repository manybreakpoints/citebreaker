import mlflow
import openai

mlflow.openai.autolog()


class CitationBot:
    def __init__(self):
        self.client = openai.OpenAI()
        with open('data/History_of_Denmark.md', 'r') as file:
            self.knowledge_base = file.read()

    def _get_context(self, question: str) -> str:  # TODO should be rag?
        return self.knowledge_base

    @mlflow.trace
    def answer(self, question: str) -> str:
        context = self._get_context(question)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a an answer and citation bot that creates good answers with precise citations from your large context of danish history. Always cite in quoutes and be very correct when citing."},
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
