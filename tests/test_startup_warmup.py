from types import SimpleNamespace
from app.services import startup_warmup
from app.services.llm.tracking import track_usage, timed_node


def test_local_warmup_uses_real_cached_path_once():
    seen=[]
    client=SimpleNamespace(embed=lambda texts:seen.append(texts))
    result=startup_warmup.warm_local_embedding(SimpleNamespace(embedding_warmup_enabled=True,embedding_provider='bge'),loader=lambda:client)
    assert result['ready'] and result['state']=='ready' and len(seen)==1


def test_remote_warmup_never_calls_paid_embedding():
    result=startup_warmup.warm_local_embedding(SimpleNamespace(embedding_warmup_enabled=True,embedding_provider='qwen'),loader=lambda:(_ for _ in ()).throw(AssertionError('paid')))
    assert result['ready'] and result['state']=='not_requested'


def test_failed_warmup_does_not_claim_readiness_or_expose_paths():
    def fail():raise ValueError('private/path/key')
    result=startup_warmup.warm_local_embedding(SimpleNamespace(embedding_warmup_enabled=True,embedding_provider='bge'),loader=fail)
    assert not result['ready'] and result['state']=='failed'
    assert 'private' not in str(result)


def test_node_measurements_do_not_modify_output_and_are_request_scoped():
    fn=timed_node('stage',lambda x:{'result':x})
    with track_usage() as a:
        assert fn(4)=={'result':4}
    with track_usage() as b:
        assert b.node_timings_ms=={}
    assert a.node_timings_ms['stage']>=0
