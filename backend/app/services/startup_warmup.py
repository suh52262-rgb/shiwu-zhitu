"""Optional local-only startup work, separate from dependency-free liveness."""
import time

_status = {'state': 'not_requested', 'ready': True, 'elapsed_s': 0.0}


def status():
    return dict(_status)


def warm_local_embedding(settings, *, loader=None):
    global _status
    if not settings.embedding_warmup_enabled or settings.embedding_provider != 'bge':
        _status = {'state': 'not_requested', 'ready': True, 'elapsed_s': 0.0}
        return status()
    start = time.perf_counter()
    _status = {'state': 'warming', 'ready': False, 'elapsed_s': 0.0}
    try:
        if loader is None:
            from app.rag.embedding import get_embedding_client
            loader = get_embedding_client
        loader().embed(['本地启动预热'])
        _status = {'state': 'ready', 'ready': True,
                   'elapsed_s': round(time.perf_counter() - start, 3)}
    except Exception as exc:
        _status = {'state': 'failed', 'ready': False,
                   'elapsed_s': round(time.perf_counter() - start, 3),
                   'error_type': type(exc).__name__,
                   'message': '本地教材模型预热失败，请检查模型文件与后端日志后重启'}
    return status()
