from pathlib import Path
import sys

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rag_bounded_model_call import SingleAttemptStructuredClient


class Output(BaseModel):
    value: int


class Client:
    max_retries = 0
    calls = 0
    def __init__(self, raw):
        self.raw = raw
    def chat(self, system, user):
        self.calls += 1
        assert 'value' in system
        return self.raw


def test_valid_response_uses_one_request():
    client = Client('{"value": 7}')
    assert SingleAttemptStructuredClient(client).chat_structured('rule', 'data', Output).value == 7
    assert client.calls == 1


@pytest.mark.parametrize('raw', ['not json', '{"value": "wrong"}', '{}'])
def test_bad_format_never_generates_a_second_request(raw):
    client = Client(raw)
    with pytest.raises(ValueError):
        SingleAttemptStructuredClient(client).chat_structured('rule', 'data', Output)
    assert client.calls == 1


def test_network_retries_must_also_be_disabled():
    client = Client('{}')
    client.max_retries = 1
    with pytest.raises(ValueError):
        SingleAttemptStructuredClient(client)
    assert client.calls == 0
