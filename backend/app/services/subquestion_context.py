"""Resolve configured subquestion anchors, without grading the student's physics.

Anchors are authored problem metadata. They identify a condition, never correctness.
A conflicting handwritten number is uncertainty, not an error or a corrected OCR fact.
"""
import json
import re
import ast


def compact(text):
    return re.sub(r'\s+', '', text).replace('＝', '=').replace('₁', '1').replace('₂', '2')


def resolve_subquestion(problem, answer):
    questions = (getattr(problem, 'metadata', {}) or {}).get('subquestions', {})
    if not questions:
        return {}
    normalized = compact(answer)
    label = re.match(r'^\s*[（(]\s*(\d+)\s*[)）]', answer)
    label = label.group(1) if label else ''
    matches = []
    for key, question in questions.items():
        for anchor in question.get('condition_anchors', []):
            if compact(anchor) in normalized:
                matches.append(key)
                break
    matches = list(dict.fromkeys(matches))
    target = matches[0] if len(matches) == 1 else (label if label in questions and not matches else '')
    conflict = bool(target and label and label != target) or len(matches) > 1
    result = {'target': target, 'label': label, 'status': 'conflict' if conflict else
              ('resolved' if target else 'unresolved'), 'basis': 'configured_condition' if matches else 'written_label'}
    if target:
        result.update(statement=questions[target]['statement'], nodes=questions[target].get('nodes', []))
    # Exact, local evidence for result presence. This does NOT decide correctness or
    # direction from a positive magnitude. No absent-result claim comes from an LLM.
    facts = []
    for field in questions.get(target, {}).get('result_fields', []):
        pattern = field.get('presence_pattern')
        found = re.search(pattern, answer) if pattern else None
        facts.append({'key': field['key'], 'label': field['label'], 'present': bool(found),
                      'quote': found.group(0) if found else '',
                      'start': found.start() if found else None, 'end': found.end() if found else None})
    result['result_fields'] = facts
    direction = next((item for item in facts if item['key'] == 'F_direction'), {})
    expected_direction = questions.get(target, {}).get('force_direction')
    window = answer[max(0,(direction.get('start') or 0)-16):(direction.get('end') or 0)+16]
    asserted = direction.get('present') and not any(word in window for word in
        ('为正','正方向','不是','并非','不向','是否','如果','不要','吗','？','?'))
    if expected_direction and asserted and '向左' in direction['quote'] and expected_direction == '向右':
        result['verified_error'] = ('已写出的水平力方向「' + direction['quote'] +
                                    '」与本题T=0条件下应向右作用的水平力不符。')
    forms = questions.get(target, {}).get('valid_partial_equations', [])
    body=re.sub(r'^[（(]\d+[）)]','',normalized)
    equations = re.findall(r'[A-Za-z0-9_()+*/.−-]+=[A-Za-z0-9_()+*/.−-]+', body)
    residue = body
    for equation in equations:residue=residue.replace(equation,'')
    residue=re.sub(r'^[（(]\d+[）)]','',residue)
    only_equations=bool(re.fullmatch(r'[(){}，。；,;∴∵即令有所以得列式]*',residue))
    if forms and only_equations and len(equations) >= 3 and not any(item.get('present') for item in facts) and not result.get('verified_error'):
        # Algebraic equivalence, not keyword correctness. Unknown syntax refuses
        # the shortcut; it is still analysed by the normal workflow.
        try:
            expected = [_difference(form) for form in forms]
            result['verified_partial'] = all(any(_equivalent(_difference(eq), ref) for ref in expected)
                                             for eq in equations)
        except (ValueError, SyntaxError):
            result['verified_partial'] = False
    return result


def _difference(equation):
    """Parse a tiny arithmetic grammar safely; never eval/sympify user strings."""
    import sympy as sp
    names = {name: sp.Symbol(name) for name in ('F','T','m1','m2','a','g')}
    def expression(text):
        text=text.replace('−','-')
        tokens=re.findall(r'm[12]|[FTga]|\d+(?:\.\d+)?|[()+*/-]',text)
        if ''.join(tokens)!=text:raise ValueError('Unknown token')
        joined=''
        for i,token in enumerate(tokens):
            if i and (tokens[i-1] in names or tokens[i-1][-1].isdigit() or tokens[i-1]==')') and (token in names or token[0].isdigit() or token=='('):joined+='*'
            joined+=token
        def visit(node):
            if isinstance(node,ast.Expression):return visit(node.body)
            if isinstance(node,ast.Name) and node.id in names:return names[node.id]
            if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):return sp.Rational(str(node.value))
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
                return -visit(node.operand) if isinstance(node.op,ast.USub) else visit(node.operand)
            if isinstance(node,ast.BinOp):
                left,right=visit(node.left),visit(node.right)
                if isinstance(node.op,ast.Add):return left+right
                if isinstance(node.op,ast.Sub):return left-right
                if isinstance(node.op,ast.Mult):return left*right
                if isinstance(node.op,ast.Div):return left/right
            raise ValueError('Unsupported arithmetic')
        return visit(ast.parse(joined,mode='eval'))
    left,right=equation.split('=')
    return expression(left)-expression(right)


def _equivalent(left,right):
    import sympy as sp
    return sp.simplify(left-right)==0 or sp.simplify(left+right)==0


def context_block(context):
    if not context:
        return ''
    return '\n【服务端统一小问与书写事实】\n' + json.dumps(context, ensure_ascii=False) + (
        '\n归属resolved时只针对target小问分析；unresolved时不能假定小问。result_fields中present=true的内容已经写出，不能声称遗漏。'
        '大小、代数符号、方向分别核对；未写方向不等于方向错误。')


def clarification(context):
    return (f"识别标号为第({context.get('label') or '?'})问，但当前条件与小问归属存在冲突。"
            f"请核对是否在回答第({context.get('target') or '?'})问，确认后更正识别标号。"
            "这轮暂不判定作答对错，原文已保留。")


def render_missing_fields(keys, context):
    """Generate absence wording only for authored fields actually absent in input."""
    catalogue = {item['key']: item for item in context.get('result_fields', [])}
    valid = [catalogue[key]['label'] for key in dict.fromkeys(keys)
             if key in catalogue and not catalogue[key]['present']]
    rejected = [key for key in keys if key not in catalogue or catalogue[key]['present']]
    return ('请补充或核对：' + '、'.join(valid) if valid else ''), rejected
