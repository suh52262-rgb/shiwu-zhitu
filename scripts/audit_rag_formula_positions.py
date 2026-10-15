"""Frozen nearby-page label audit against original searchable PDFs; read-only."""
import json,sys,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 import fitz
 from app.rag.formula_positions import label_position_links,standalone_label
 src=ROOT/'artifacts/rag_candidate_stage_prototype_20261014';facts={};inputs={}
 for case in ['wave_positive','spherical_cap','monatomic_heat','rod_emf','dielectric_supply','em_wave_vectors','oscillator_energy']:
  p=src/(case+'.json');inputs[str(p.relative_to(ROOT))]=sha(p)
  for b in json.loads(p.read_text(encoding='utf-8'))['nearby_blocks']:facts[b['id']]=b
 links,omissions=label_position_links(list(facts.values()));docs={};checks=[];source_hashes={}
 for link in links:
  doc=link['source'];prefix=doc.split('-')[1];suffix=''
  if prefix=='1':suffix='*学习辅导*' if doc.endswith('jieda') else '*上册*'
  matches=list((ROOT/'data/textbooks').glob(f'教材-{prefix}-*.pdf'))
  if prefix=='1':matches=[p for p in matches if ('学习辅导' in p.name if doc.endswith('jieda') else '上册' in p.name)]
  if len(matches)!=1:checks.append({**link,'source_check':'pdf_mapping_missing'});continue
  p=matches[0]
  if doc not in docs:docs[doc]=fitz.open(p);source_hashes[p.name]=sha(p)
  page=docs[doc][link['page']-1];rect=fitz.Rect(link['label_bbox'])
  if not page.rect.contains(rect):status='outside_page_bounds';text=''
  else:
   text=page.get_text('text',clip=rect+(-1,-1,1,1)).strip();status='verified_label_text' if standalone_label(text)==link['reference'] else 'label_clip_mismatch'
  checks.append({**link,'source_check':status,'pdf_label_text':text,'pdf_page_size':[page.rect.width,page.rect.height]})
 for d in docs.values():d.close()
 write(a.out/'results.json',{'unique_blocks':len(facts),'links':checks,'omissions':omissions})
 from collections import Counter
 write(a.out/'summary.json',{'inputs':inputs,'pdf_hashes':source_hashes,'module_sha256':sha(ROOT/'backend/app/rag/formula_positions.py'),'script_sha256':sha(Path(__file__)),'source_checks':dict(Counter(c['source_check'] for c in checks)),'numbered_label_links':len(links),'scope':'Same-page exact unique printed labels in known nearby-page snapshots; not equation-body geometry or semantic dependency; default unchanged','paid_calls':0});print(json.dumps(dict(Counter(c['source_check'] for c in checks)),ensure_ascii=False))
if __name__=='__main__':main()
