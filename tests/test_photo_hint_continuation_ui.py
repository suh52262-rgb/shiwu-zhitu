"""A help click after a photo must not submit fabricated blank-answer prose."""
from pathlib import Path
import httpx
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'frontend' / 'streamlit_app.py'


def photo_app(monkeypatch, *, finished=False):
    def get(url, **kwargs):
        if url.endswith('/problems'):
            data = {'items': [{'problem_id': 'p', 'title': '题'}]}
        elif url.endswith('/problems/p'):
            data = {'problem_id': 'p', 'statement': '完整题干'}
        else:
            data = {'status': 'ok'}
        return httpx.Response(200, json=data, request=httpx.Request('GET', url))
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs['json'])
        return httpx.Response(200, json={'attempt_id': 'photo-attempt', 'hint_level': 3,
            'hint': '沿已有方程检查方向', 'finished': False}, request=httpx.Request('POST', url))
    monkeypatch.setattr(httpx, 'get', get)
    monkeypatch.setattr(httpx, 'post', post)
    at = AppTest.from_file(str(APP)).run()
    at.session_state.attempt_id = 'photo-attempt'
    at.session_state.attempt_status = 'completed' if finished else 'in_progress'
    at.session_state.tutor = {'hint_level': 2, 'hint': '照片的L2提示', 'finished': finished}
    at.session_state.video_result = {'attempt_id': 'photo-attempt', 'media_type': 'image',
        'recognized_text': 'F=m a', 'submitted': 1,
        'rounds': [{'answer': 'F=m a', 'hint_level': 2, 'hint': '照片的L2提示'}]}
    at.run()
    return at, calls


def test_photo_l2_help_sends_empty_context_request_and_keeps_original(monkeypatch):
    at, calls = photo_app(monkeypatch)
    next(button for button in at.button if button.label == '请求提示').click().run()
    assert not at.exception
    assert len(calls) == 1
    assert calls[0]['answer'] == '' and calls[0]['request_hint'] is True
    assert calls[0]['attempt_id'] == 'photo-attempt'
    assert at.session_state.tutor['hint_level'] == 3
    assert at.session_state.video_result['recognized_text'] == 'F=m a'


def test_completed_photo_help_preserves_hint_and_does_not_start_new_attempt(monkeypatch):
    at, calls = photo_app(monkeypatch, finished=True)
    next(button for button in at.button if button.label == '请求提示').click().run()
    assert not at.exception and not calls
    assert at.session_state.tutor['hint_level'] == 2
    assert any('这次作答已结束' in message.value for message in at.warning)


def test_new_written_answer_takes_precedence_over_photo_context(monkeypatch):
    at, calls = photo_app(monkeypatch)
    at.text_area(key='answer_input').input('F=49 N').run()
    next(button for button in at.button if button.label == '请求提示').click().run()
    assert calls[0]['answer'] == 'F=49 N'
    assert at.session_state.get('video_result') is None

def test_pending_photo_shows_ocr_and_help_does_not_post(monkeypatch):
    at,calls=photo_app(monkeypatch)
    old_get=httpx.get
    def get(url,**kwargs):
        if '/photo/jobs/' in url:
            return httpx.Response(200,json={'job_id':'a'*32,'attempt_id':'photo-attempt','status':'running','recognized_text':'F=ma'},request=httpx.Request('GET',url))
        return old_get(url,**kwargs)
    monkeypatch.setattr(httpx,'get',get)
    at.session_state['photo_pending']={'student_id':at.session_state.active_session[0], 'problem_id':'p','base_attempt_id':'photo-attempt','request_id':'request','job_id':'a'*32,'status':'running','epoch':0}
    at.run()
    assert not at.exception
    assert any('F=ma' in item.value for item in at.code)
    assert next(b for b in at.button if b.label=='重新开始本题').disabled
    next(b for b in at.button if b.label=='请求提示').click().run()
    assert not calls
