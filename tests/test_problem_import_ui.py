"""Exercise editing and publication controls in the actual authoring app."""
from pathlib import Path
import httpx
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'frontend/problem_import_app.py'

def test_import_ui_saves_explicit_review_and_keeps_publish_disabled_until_saved(monkeypatch):
    item={'item_id':'item1','problem_id':'','title':'','statement':'Original question','answer':'original solution',
          'source_pages':[1],'crops':[],'revision':0,'review':None,'publication':None,'history':[],
          'original_statement':'Original question','original_answer':'original solution'}
    record={'import_id':'abc','filename':'scan.png','status':'draft','source_sha256':'a'*64,'warnings':[],
            'pages':[{'page':1,'method':'vision','status':'ready','original_text':'original scan','preview_width':300,'preview_height':200}],
            'items':[item]}
    calls=[]
    def request(method,url,**kwargs):
        if method=='GET':
            result={'items':[record]} if url.endswith('/problem-imports') else record
        else:
            calls.append((method,kwargs.get('json')))
            body=kwargs.get('json',{})
            item.update(body.get('values',{}))
            item['revision']+=1
            item['review']={'reviewer':body.get('reviewer')} if body.get('confirmed') else None
            result=item
        return httpx.Response(200,json=result,request=httpx.Request(method,url))
    monkeypatch.setattr(httpx,'request',request)
    app=AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert next(b for b in app.button if b.label=='发布已核对的保存版本').disabled
    next(x for x in app.text_input if x.label.startswith('题目编号')).input('custom_001')
    next(x for x in app.text_input if x.label.startswith('复核者')).input('actual operator')
    next(x for x in app.checkbox if x.label.startswith('我已对照')).check()
    next(b for b in app.button if b.label=='保存核对结果').click().run()
    assert not app.exception
    assert calls[0][0]=='PUT' and calls[0][1]['base_revision']==0
    assert calls[0][1]['confirmed'] is True
    assert not next(b for b in app.button if b.label=='发布已核对的保存版本').disabled
