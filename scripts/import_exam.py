"""PDF/图片 -> 私有题目草稿；核对后从题目导入页发布。

不再直接写学生题库，也不在未核对草稿上自动建图。
--dry-run 只检查PDF文字层切题，零模型调用且不落盘。
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from app.rag import exam_split
from app.services.problem_import import ProblemImporter

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pdf')
    ap.add_argument('--prefix',default='imported')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--use-vision',action='store_true',help='显式允许模型转写与API费用')
    ap.add_argument('--min-chars',type=int,default=10)
    ap.add_argument('--build-graph',action='store_true',help='旧参数仅用于给出迁移提示')
    args=ap.parse_args()
    if args.build_graph:
        print('未核对草稿不能直接建图。先从导入页发布，再运行generate_graph.py，候选Graph仍需独立审核。')
        return 1
    source=Path(args.pdf)
    if not source.is_file():
        print('找不到文件')
        return 1
    if args.dry_run:
        if args.use_vision or source.suffix.lower()!='.pdf':
            print('--dry-run仅检查PDF文字层，不调用模型；图片请创建草稿再核对')
            return 1
        drafts=exam_split.split_pdf(source)
        for p in drafts:
            print(f'第{p.number}题，来源页{p.pages}\n题干：{p.statement}\n答案：{p.solution}\n')
        print(f'切出{len(drafts)}道；未写入学生题库。扫描件无文字层时需视觉转写或手工录入。')
        return 0 if drafts else 1
    importer=ProblemImporter()
    record=importer.create(source.read_bytes(),source.name,use_vision=args.use_vision)
    if record['status']=='failed':
        print(record['error'])
        return 1
    for n,item in enumerate(record['items'],1):
        importer.edit(record['import_id'],item['item_id'],base_revision=0,
                      values={'problem_id':f'{args.prefix}_{n:03d}','title':item['statement'][:60]})
    print(f"已创建草稿 {record['import_id']}，共{len(record['items'])}道；未发布。")
    print('启动题目导入页核对：python -m streamlit run frontend/problem_import_app.py --server.port 8502')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
