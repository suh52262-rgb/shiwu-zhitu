from copy import deepcopy
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from select_rag_region_probe import check_sources


@pytest.fixture
def database():
    connection = sqlite3.connect(':memory:')
    connection.execute('CREATE TABLE document_blocks (id TEXT, doc_id TEXT, page INTEGER, block_index INTEGER, text TEXT, indexable INTEGER, is_generated INTEGER)')
    connection.execute('INSERT INTO document_blocks VALUES (?,?,?,?,?,?,?)',
                       ('real', 'book', 2, 3, 'Original condition; complete relation.', 1, 0))
    yield connection
    connection.close()


def material():
    block = {'id': 'real', 'source': 'book', 'page': 2, 'block_index': 3,
             'text': 'Original condition; complete relation.'}
    return [block], {'sources': [deepcopy(block)]}


def test_partial_verbatim_material_is_recorded_as_partial(database):
    pool, baseline = material()
    pool[0]['text'] = baseline['sources'][0]['text'] = 'Original condition;'
    verified = check_sources(pool, baseline, database)
    assert verified[0]['id'] == 'real'
    assert verified[0]['whole_source'] is False


@pytest.mark.parametrize('defect', ['unknown', 'duplicate', 'wrong_book', 'wrong_block', 'rewrite', 'non_indexable', 'generated', 'fallback_rewrite'])
def test_invalid_source_cannot_reach_selector(database, defect):
    pool, baseline = material()
    if defect == 'unknown':
        pool[0]['id'] = 'fabricated'
    elif defect == 'duplicate':
        pool.append(deepcopy(pool[0]))
    elif defect == 'wrong_book':
        pool[0]['source'] = 'different-book'
    elif defect == 'wrong_block':
        pool[0]['block_index'] = 4
    elif defect == 'rewrite':
        pool[0]['text'] = 'Newly invented assumption'
    elif defect == 'non_indexable':
        database.execute('UPDATE document_blocks SET indexable=0')
    elif defect == 'generated':
        database.execute('UPDATE document_blocks SET is_generated=1')
    else:
        baseline['sources'][0]['text'] = 'Altered baseline material'
    with pytest.raises(ValueError):
        check_sources(pool, baseline, database)
