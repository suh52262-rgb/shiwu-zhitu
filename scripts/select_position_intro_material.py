"""One bounded paid selector diagnostic, known string query, no retries."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from probe_rag_prose_spans import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    case_id='string_modes_boundaries'
    input_path=a.input/(case_id+'.json');baseline_path=a.baseline/(case_id+'.json')
    current=json.loads(input_path.read_text(encoding='utf-8'));old=json.loads(baseline_path.read_text(encoding='utf-8'))
    if current['id']!=case_id or old['id']!=case_id or not current['added']:raise ValueError('Missing validated additions')
    protocol={'scope':'Known development single-query selection; not student teaching, blind test or semantic acceptance',
        'input_sha256':sha(input_path),'baseline_sha256':sha(baseline_path),
        'sources':{n:sha(ROOT/n) for n in ['scripts/select_position_intro_material.py','backend/app/services/rag_joint_selection.py','backend/app/services/rag_selection.py']},
        'expected_model':'deepseek-flash','max_model_calls':1,'max_retries':0,'timeout_seconds':60,
        'material_budget':1200,'defaults_changed':False,'required_points_to_selector':False}
    write(a.out/'protocol.json',protocol)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    os.environ.update(LLM_MAX_RETRIES='0',LLM_TIMEOUT='60')
    from app.services.llm import get_llm
    from app.services.rag_joint_selection import unified_select
    from prepare_joint_delivery_review import validate_material
    result={'id':case_id,'query':old['query'],'status':'started'}
    try:
        llm=get_llm('tutor')
        if llm.model!='deepseek-flash':raise ValueError('Configured tutor model differs from authorized model')
        llm.reset_last_call()
        selected=unified_select(current['candidate_pool'],old['query'],old['baseline_plan'],budget=1200,llm=llm)
        validate_material(current['candidate_pool'],old['baseline_plan'],selected)
        result.update(status='completed' if selected['selection_status']=='ready' else 'failed',
            baseline_plan=old['baseline_plan'],selected_plan=selected,
            model_call=llm.last_call.as_dict() if llm.last_call else None,
            new_ids_selected=[b['id'] for b in current['added'] if b['id'] in selected.get('selected_ids',[])],
            semantic_sufficiency_evaluated=False)
    except Exception as error:result.update(status='failed',error=type(error).__name__+': '+str(error))
    if sha(input_path)!=protocol['input_sha256'] or sha(baseline_path)!=protocol['baseline_sha256'] or any(sha(ROOT/n)!=h for n,h in protocol['sources'].items()):raise RuntimeError('Frozen input/source drift')
    write(a.out/'result.json',result);print(result['status'],result.get('selected_plan',{}).get('content_characters'),flush=True)


if __name__=='__main__':main()
