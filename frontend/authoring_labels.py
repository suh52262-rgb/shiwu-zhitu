"""Distinct option labels: Streamlit selects formatted strings, not hidden IDs."""


def item_label(item, ordinal):
    title=item.get('title') or item.get('statement','')[:35] or '未命名题目'
    pid=item.get('problem_id') or '未设题号'
    version=item.get('question_revision',1)
    status='已发布' if item.get('publication') else '草稿'
    return f'{ordinal+1}. {title} · {pid} · v{version} · {status}'
