"""Display aliases only. Never rename stored documents, paths, or model evidence."""
import json
from pathlib import Path
from functools import lru_cache

REGISTRY = Path(__file__).resolve().parents[1] / 'data' / 'document_names.json'


@lru_cache(maxsize=1)
def document_names():
    try:
        data = json.loads(REGISTRY.read_text(encoding='utf-8'))
        documents = data.get('documents')
        if data.get('schema_version') != 1 or not isinstance(documents, dict):
            return {}
        return {key: value for key, value in documents.items() if isinstance(value, dict)
                and isinstance(value.get('title'), str) and value['title'].strip()}
    except (OSError, ValueError, AttributeError):
        return {}


def document_title(doc_id):
    value = doc_id if isinstance(doc_id, str) and doc_id else '未知教材'
    return document_names().get(value, {}).get('title', value)


def document_label(doc_id):
    title = document_title(doc_id)
    return title if title == doc_id else f'{title} · {doc_id}'


def document_matches(doc_id, query):
    from problem_browser import normalize
    metadata = document_names().get(doc_id, {})
    text = normalize(' '.join([doc_id, document_title(doc_id), metadata.get('source_filename', '')]))
    return all(word in text for word in normalize(query).split())
