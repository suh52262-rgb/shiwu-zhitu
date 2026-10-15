"""One request/one parse adapter for bounded engineering probes only.

The production structured API can retry a malformed response independently of
network retries. This adapter deliberately fails on the first malformed result.
It does not change production behavior or claim a hard wall-clock deadline.
"""
import json

from app.services.llm.base import JSON_INSTRUCTION, extract_json


class SingleAttemptStructuredClient:
    def __init__(self, client):
        if client.max_retries != 0:
            raise ValueError('Bounded probe requires zero network retries')
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat_structured(self, system, user, schema):
        instruction = JSON_INSTRUCTION.format(
            schema_json=json.dumps(schema.model_json_schema(), ensure_ascii=False))
        raw = self.client.chat(system + instruction, user)
        return schema.model_validate(extract_json(raw))
