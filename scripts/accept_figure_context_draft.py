"""Actual registered figure copied into an isolated bank; no production publication."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')


def main(out):
    from app.models.problem import Problem
    from app.services.problem_repository import ProblemRepository
    from app.services.graph_versions import atomic_json
    from app.services.figure_context_drafts import FigureContextDrafts
    from app.services.problem_import import ImportConflict
    out=Path(out)
    if out.exists():raise ValueError('Use a fresh evidence directory')
    out.mkdir(parents=True)
    source=ProblemRepository(ROOT/'data/problems')
    original=source.get_problem('phy_002')
    repo=ProblemRepository(out/'isolated_bank')
    original.metadata={**original.metadata,'source':'reviewed_import','origin_note':'AI engineering acceptance copy, not a new imported problem'}
    atomic_json(repo.problem_path('phy_002'),original.model_dump())
    for name in original.figures:
        destination=repo.problem_dir('phy_002')/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source.figure_path('phy_002',name),destination)
    service=FigureContextDrafts(out/'draft_records',repo)
    rid=str(uuid4())
    first=service.submit('phy_002',rid)
    repeated=service.submit('phy_002',rid)
    assert first==repeated
    record={'scope':'Actual vision model + isolated registered figure; no teacher gold, graph generation or publication',
            'first':first,'duplicate_equal':first==repeated,'source_image_sha256':first['source']['figure_sha256'],
            'semantic_accuracy':'not_independently_verified'}
    if first['status']=='ready':
        assert service.validate('phy_002',rid)['request_id']==rid
        changed=repo.get_problem('phy_002');changed.statement+='\nAI测试：题面修订。'
        atomic_json(repo.problem_path('phy_002'),changed.model_dump())
        assert service.get('phy_002',rid)['stale']
        try:service.validate('phy_002',rid)
        except ImportConflict:record['stale_rejected']=True
        else:raise AssertionError('Stale figure draft accepted')
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VISION_DRAFT '+first['status']+' duplicate cached; no graph published')


if __name__=='__main__':main(sys.argv[1])
