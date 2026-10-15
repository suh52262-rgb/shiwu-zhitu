"""Candidate Tutor v5: literal snippets come from the current request's pool."""
import json
import re
from app.services.review_builder import CITATION_TOKEN, parse_citations, render_citations, _locate

FREE_QUOTES=re.compile(r'「[^」]+」|『[^』]+』|“[^”]+”|"[^"\n]+"|(?<![A-Za-z0-9_])\x27[^\x27\n]+\x27(?![A-Za-z0-9_])')


def pool_for_hint(state, prior):
    return [
        {'id':'H0','source':'student_answer','text':state.current_answer,'revision':0},
        {'id':'Q0','source':'statement','text':state.problem_content,'revision':0},
        *[{'id':f'H{i}','source':'student_answer','text':text[:200],'revision':0}
          for i,text in enumerate(prior,1)],
    ]


def catalogue(pool):
    # H0/Q0 are already present in the request. Do not duplicate the whole writing.
    return '\n【提示引文目录】\n'+json.dumps([
        {'id':e['id'],'source':e['source'],'length':len(e['text']),
         'text':e['text'] if e['id'] not in {'H0','Q0'} else '见上方学生当前作答/题目正文'}
        for e in pool],ensure_ascii=False)


def render_hint(text,pool):
    rendered, defects=render_citations(text,pool)
    residue=CITATION_TOKEN.sub('',text or '')
    defects += [{'kind':'free_quote','raw':m.group(0)} for m in FREE_QUOTES.finditer(residue)]
    refs=[]
    index={e['id']:e for e in pool}
    if not defects:
        for token in parse_citations(text):
            evidence=index[token['evidence_id']]; body=evidence['text']
            if token['quote'] is not None:start,end=_locate(body,token['quote'])
            else:start,end=token['start'],token['end'] if token['end'] is not None else len(body)
            refs.append({'evidence_id':evidence['id'],'source':evidence['source'],
                         'start':start,'end':end,'text':body[start:end],
                         'semantic_validated':False})
    return rendered,refs,defects
