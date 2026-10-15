"""Single-process, durable account stop for 401/402. Never probes or auto-resumes."""
from __future__ import annotations
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from app.services.graph_versions import atomic_json

_LOCK=threading.RLock()


class AccountCircuit:
    def __init__(self, provider, endpoint, api_key, root=None):
        url=urlsplit(endpoint)
        path=url.path.rstrip('/')
        if path.endswith('/v1'):path=path[:-3]
        # Role/provider aliases and the OpenAI-compatible /v1 suffix must not
        # give one billing account a fresh circuit. Never persist the URL/key.
        account=urlunsplit((url.scheme.lower(),url.netloc.rsplit('@',1)[-1].lower(),path,'','')) if endpoint else 'default:'+provider
        self.scope=hashlib.sha256((account+'\0'+api_key).encode()).hexdigest()
        self.path=Path(root or os.getenv('LLM_CIRCUIT_DIR') or Path(__file__).resolve().parents[4]/'data/llm_circuits')/(self.scope+'.json')

    def _read(self):
        from app.services.llm.base import LLMConfigError
        if not self.path.exists():return {'scope':self.scope,'revision':0,'blocked':False,'status_code':None}
        try:
            state=json.loads(self.path.read_text(encoding='utf-8'))
            if (state['scope']!=self.scope or type(state['revision']) is not int or state['revision']<0
                    or type(state['blocked']) is not bool or state.get('status_code') not in (None,401,402)):
                raise ValueError('Invalid state')
            return state
        except (ValueError,KeyError,OSError) as exc:
            raise LLMConfigError('模型费用故障状态不可读；请由本地操作者检查，未自动清空') from exc

    def status(self):
        with _LOCK:return self._read().copy()

    def before_call(self):
        from app.services.llm.base import LLMResponseError
        with _LOCK:
            state=self._read()
            if state['blocked']:
                raise LLMResponseError('模型账户已停止新增调用；完成本地处理后显式恢复。未发出供应商请求',state['status_code'])
            return state['revision']

    def failed(self, error, ticket):
        code=getattr(error,'status_code',None)
        if code not in (401,402):return
        with _LOCK:
            state=self._read()
            # A late response from before an explicit operator resume cannot
            # revoke that newer revision. Other in-flight calls are not refunded.
            if state['revision']!=ticket:return
            state.update(blocked=True,status_code=code,revision=ticket+1,
                         updated_at=datetime.now(timezone.utc).isoformat())
            self.path.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(self.path,state)

    def resume(self, expected_revision):
        with _LOCK:
            state=self._read()
            if state['revision']!=expected_revision:raise ValueError('故障状态已更新，请重新读取后恢复')
            if not state['blocked']:return state.copy()
            state.update(blocked=False,status_code=None,revision=expected_revision+1,
                         updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(self.path,state)
            return state.copy()


def attach_guard(client, provider, endpoint, api_key):
    client.request_guard=AccountCircuit(provider,endpoint,api_key)
    return client
