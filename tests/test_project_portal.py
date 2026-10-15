"""Overview navigation is read-only and shares validated local URL contracts."""
from html.parser import HTMLParser
import importlib
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.append(dict(attrs))


def test_overview_does_not_call_api_and_links_to_all_four_workspaces(monkeypatch):
    for name in ('get','post','request'):
        monkeypatch.setattr(httpx,name,lambda *a,**k: (_ for _ in ()).throw(AssertionError('overview is static')))
    app=AppTest.from_file(str(ROOT/'frontend/portal_app.py')).run()
    assert not app.exception
    markup='\n'.join(m.value for m in app.markdown)
    links=Links();links.feed(markup)
    assert [link['href'] for link in links.links]==[f'http://127.0.0.1:{port}' for port in range(8501,8505)]
    assert all(link['target']=='_blank' and 'noopener' in link['rel'] for link in links.links)
    assert '大学物理' in markup and '分级提示' in markup and '可核对' in markup
    assert not app.get('file_uploader') and not app.button


@pytest.mark.parametrize('port',['0','65536','garbage','" onclick="alert(1)'])
def test_portal_and_existing_navigation_share_safe_port_fallback(monkeypatch,port):
    monkeypatch.syspath_prepend(str(ROOT/'frontend'))
    ui=importlib.import_module('workspace_style')
    monkeypatch.setenv('WORKSPACE_STUDENT_PORT',port)
    assert ui.workspace_url('student')=='http://127.0.0.1:8501'
    assert 'http://127.0.0.1:8501' in ui.navigation_markup('student')
    monkeypatch.setenv('WORKSPACE_OVERVIEW_PORT',port)
    assert ui.workspace_url('overview')=='http://127.0.0.1:8500'


def test_portal_respects_the_same_custom_port_as_other_pages(monkeypatch):
    monkeypatch.setenv('WORKSPACE_TEACHER_PORT','9512')
    app=AppTest.from_file(str(ROOT/'frontend/portal_app.py')).run()
    assert not app.exception
    links=Links();links.feed('\n'.join(m.value for m in app.markdown))
    assert links.links[1]['href']=='http://127.0.0.1:9512'
