"""Local operator control. No model request, no keys or endpoints printed."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from app.services.llm.factory import get_llm
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('action',choices=['status','resume'])
p.add_argument('--expected-revision',type=int)
p.add_argument('--confirm',action='store_true',help='Confirm local account/key issue was handled; does not probe balance')
args=p.parse_args();guard=get_llm('tutor').request_guard
if args.action=='resume':
    if not args.confirm or args.expected_revision is None:p.error('Explicit --confirm and --expected-revision required')
    state=guard.resume(args.expected_revision)
else:state=guard.status()
print(json.dumps(state,ensure_ascii=False,indent=2))
