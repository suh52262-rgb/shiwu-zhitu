"""Deterministic review focus. Valid IDs/coverage are not semantic certification."""
from app.services.knowledge_vocabulary import validate


def inspect_quality(graph,evidence,quantities,*,require_grounding=True):
    ids=graph.node_ids();available={e['id'] for e in evidence}
    issues=[]
    refs=graph.metadata.get('node_evidence',{})
    coverage=graph.metadata.get('quantity_coverage',{})
    if not isinstance(refs,dict):
        issues.append({'kind':'source_reference','description':'node_evidence应为对象'});refs={}
    if not isinstance(coverage,dict):
        issues.append({'kind':'quantity_coverage','description':'quantity_coverage应为对象'});coverage={}
    if require_grounding:
        for node in graph.nodes:
            cited=refs.get(node.id,[])
            if not isinstance(cited,list) or not cited or any(not isinstance(r,str) or r not in available for r in cited):
                issues.append({'kind':'source_reference','node':node.id,'description':'缺少当前材料中的有效来源编号'})
        for node in set(refs)-ids:
            issues.append({'kind':'source_reference','node':node,'description':'来源挂到了不存在的节点'})
    if require_grounding:
        for quantity in quantities:
            targets=coverage.get(quantity,[])
            if not isinstance(targets,list) or not targets or any(not isinstance(n,str) or n not in ids for n in targets):
                issues.append({'kind':'quantity_coverage','quantity':quantity,'description':'目标量没有映射到有效节点'})
                continue
            node_map=graph.node_map()
            if not any(node_map[n].type in {'calculation','result','verification'} for n in targets):
                issues.append({'kind':'quantity_coverage','quantity':quantity,'description':'目标只挂在建模或泛化推理，未指向计算/结果节点'})
            for index,path in enumerate(graph.valid_paths):
                if not set(targets).intersection(path):
                    issues.append({'kind':'quantity_coverage','quantity':quantity,'path_index':index,'description':'此完整路径漏掉一个必须回答的目标量'})
    points=graph.metadata.get('knowledge_points',[])+[k for n in graph.nodes for k in n.knowledge_points]
    vocab=validate(points)
    if not vocab['ok']:
        issues.append({'kind':'vocabulary','description':'表外或禁用知识点','labels':vocab['unknown']+vocab['dropped']})
    questions=graph.metadata.get('open_questions',[])
    if questions:issues.append({'kind':'unresolved_input','description':'模型报告输入或参考解待核对','questions':questions})
    return {'validator_version':'graph-quality-v1','issues':issues,'ok':not issues,
            'coverage_status':'checked' if quantities and require_grounding else 'not_checked',
            'semantic_validated':False,'vocabulary':vocab}


def repairable(quality,report):
    if any(i['kind']=='unresolved_input' for i in quality['issues']):return False
    return bool(quality['issues'] or not report.valid or report.checks.get('physics')=='issues')


def improvement(before_quality,before_report,after_quality,after_report):
    """Strict improvement without new defect families; don't erase a physics alarm."""
    if before_report.checks.get('physics') in {'pass','issues'} and after_report.checks.get('physics')=='skipped':
        return False
    def defects(q,r):
        groups={i['kind'] for i in q['issues']}
        groups.update('check:'+k for k,v in r.checks.items() if v=='issues')
        score=len(q['issues'])+sum(2 if i.severity=='error' else 1 for i in r.issues)
        return groups,score
    before,score=defects(before_quality,before_report)
    after,new_score=defects(after_quality,after_report)
    return after <= before and new_score < score
