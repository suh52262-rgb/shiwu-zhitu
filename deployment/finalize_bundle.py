"""Finalize a staging bundle after validation, excluding local runtime configuration.
Run from source workspace; no deployments or database mutation.
"""
from pathlib import Path
import json,shutil,hashlib,zipfile
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from deployment_bundle import sha,write
def finalize(bundle,zip_path,archive=True,evidence=None):
    bundle=Path(bundle);zip_path=Path(zip_path)
    # Update deployment-only tooling/guide after tested corrections.
    for p in (ROOT/'deployment').glob('*'):
        if p.is_file():shutil.copy2(p,bundle/'deployment'/p.name)
    shutil.copy2(ROOT/'scripts/deployment_bundle.py',bundle/'scripts/deployment_bundle.py')
    shutil.copy2(ROOT/'deployment/compose.yml',bundle/'docker-compose.yml')
    shutil.copy2(ROOT/'deployment/快速部署.md',bundle/'开始部署.md')
    (bundle/'README.md').write_text('# 视物知途部署包\n\n版本见VERSION.json。首先阅读[开始部署](开始部署.md)，然后按顺序执行deployment/Setup.ps1、Restore.ps1、Accept.ps1、Start.ps1。\n\n数据清单见DATA_INVENTORY.json；本轮验证及限制见deployment/验收记录.md。包内wheel支持Windows/Python3.11依赖离线安装；Python本身与Docker仍需对方准备。密钥需对方本地配置，首次拉取Docker镜像和在线教学需要联网。\n',encoding='utf-8')
    inv=json.loads((bundle/'DATA_INVENTORY.json').read_text(encoding='utf-8'))
    titles=['大学物理学习辅导与习题解答','大学物理学上册','大学物理学下册','新概念物理教程·力学','新概念物理教程·热学','新概念物理教程·电磁学','新概念物理教程·光学','新概念物理教程·量子物理']
    package_version=json.loads((bundle/'VERSION.json').read_text(encoding='utf-8'))['version']
    inventory=['# 部署数据清单',f'版本：{package_version}。正文、向量和图片均有逐文件SHA256。','','|教材|文档ID|扫描页|解析块|向量|','|---|---|---|---|---|']
    for title,(doc,values) in zip(titles,sorted(inv['documents'].items())):inventory.append(f"|{title}|{doc}|{values['pages']}|{values['blocks']}|{values['vectors']}|")
    inventory+=['',f"合计：{inv['document_count']}本文档、{inv['pages']}页、{inv['blocks']}块、{inv['vectors']}向量、{inv['figures']}个教材图片文件。",'page为扫描页序号，不能按统一偏移换算印刷页。块数与向量数不同，保留了不参与索引的原文块。',f"题库目录：{', '.join(inv['problems'])}。",f"另含：受控知识点词表、BGE模型、{len(list((bundle/'wheels').glob('*.whl')))}个Windows/Python3.11依赖wheel、部署脚本与验收证据。",'未包含：API密钥、学生数据库记录、原视频、作答运行产物、原教材PDF、虚拟环境和Docker镜像。','模型修订号见VERSION.json；wheel来源与公开SHA256见deployment/wheel-provenance.json。','向量在恢复时重新建近似索引；近并列排序可能变化，不宣称每次检索逐位一致。']
    (bundle/'数据清单.md').write_text('\n'.join(inventory)+'\n',encoding='utf-8')
    val=bundle/'deployment/validation';val.mkdir(exist_ok=True)
    # A new release must explicitly provide its own evidence, rather than
    # automatically relabelling the old 2026-10-12 run as current acceptance.
    paths={} if evidence is not None else {'deployment_offline_20261012.xml':'offline.xml','deployment_offline_20261012.log':'offline.log','deployment_accept_20261012.log':'restore_acceptance.log','deployment_browser_20261012.png':'browser.png','deployment_browser_20261012.txt':'browser.txt','deployment_clean_install_offline_final_20261012.log':'clean_install.log','deployment_setup_exact_20261012.log':'setup.log','deployment_clean_offline_20261012.xml':'clean_offline.xml','deployment_clean_offline_20261012.log':'clean_offline.log','deployment_clean_accept_20261012.log':'clean_acceptance.log'}
    for src,dst in paths.items():
        p=ROOT/'output'/src
        if p.exists():shutil.copy2(p,val/dst)
    if evidence is not None:
        evidence=Path(evidence)
        for p in evidence.iterdir():
            if p.is_file():shutil.copy2(p,val/p.name)
    acceptance=bundle/'deployment/acceptance-result.json'
    if acceptance.exists():shutil.copy2(acceptance,val/'acceptance-result.json')
    version=json.loads((bundle/'VERSION.json').read_text(encoding='utf-8'))
    identity={p.relative_to(bundle).as_posix():sha(p) for folder in ['backend','frontend','prompts','scripts','deployment'] for p in (bundle/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['acceptance-result.json'] and 'validation' not in p.relative_to(bundle).parts}
    version['source_tree_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(bundle/'VERSION.json',version)
    # No .env or local logs/PIDs/bytecode will enter archive or manifest.
    files={}
    allowed_roots={'backend','frontend','prompts','scripts','tests','evaluation','docs','simulation','.github','deployment','models','wheels','.streamlit'}
    allowed_root_files={'README.md','开始部署.md','数据清单.md','VERSION.json','DATA_INVENTORY.json','requirements.txt','requirements-deploy.lock.txt','pytest.ini','首次使用.cmd','THIRD_PARTY_NOTICES.md','LICENSE','LICENSE.md','LICENSE.txt','NOTICE','COPYING','启动学生端.cmd','启动题目导入.cmd','启动错误复核.cmd','启动教师端.cmd','启动教材管理.cmd','启动导览页.cmd','启动全部工作区.cmd','停止全部工作区.cmd','docker-compose.yml','.env.example'}
    for p in sorted(bundle.rglob('*')):
        rel=p.relative_to(bundle)
        if len(rel.parts)==1 and rel.name not in allowed_root_files:continue
        if len(rel.parts)>1 and rel.parts[0] not in allowed_roots and not (rel.parts[0]=='data' and rel.parts[1] in {'problems','documents','knowledge_vocabulary.json','document_names.json'}):continue
        if not p.is_file() or p.name in ['manifest.json','.env','acceptance-result.json'] or any(x in rel.parts for x in ['__pycache__','.venv','logs','video_jobs','photo_jobs','llm_circuits','video_runs','.pytest_cache']):continue
        files[rel.as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    # Never silently include current API keys in plaintext materials.
    from dotenv import dotenv_values
    secrets=[v for k,v in dotenv_values(ROOT/'.env').items() if k.endswith('API_KEY') and v and len(v)>12]
    for rel in files:
        p=bundle/rel
        if p.suffix.lower() in ['.py','.md','.json','.txt','.ps1','.cmd','.example','.log','.yml'] and p.stat().st_size<5_000_000:
            text=p.read_text(encoding='utf-8',errors='replace')
            if any(secret in text for secret in secrets):raise ValueError('Credential leak in '+rel)
    write(bundle/'manifest.json',{'schema':'shiwu-deploy.v1','version':version['version'],'files':files,'excluded':['.env','.venv','.git','student data/history','raw textbook PDFs','original videos','runtime logs/PIDs','presentations','temporary files']})
    from deployment_bundle import verify
    verify(bundle)
    if not archive:
        print('PACKAGE_STAGED',len(files));return
    zip_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for rel in sorted(files):z.write(bundle/rel,bundle.name+'/'+rel)
        z.write(bundle/'manifest.json',bundle.name+'/manifest.json')
    with zipfile.ZipFile(zip_path) as z:
        assert not z.testzip()
        assert all(not n.endswith('/.env') and '/logs/' not in n for n in z.namelist())
    digest=sha(zip_path)
    zip_path.with_suffix('.zip.sha256').write_text(digest+'  '+zip_path.name+'\n',encoding='utf-8')
    print(json.dumps({'zip':str(zip_path),'bytes':zip_path.stat().st_size,'sha256':digest,'files':len(files)},ensure_ascii=False,indent=2))
if __name__=='__main__':
    evidence=Path(sys.argv[sys.argv.index('--evidence')+1]) if '--evidence' in sys.argv else None
    finalize(sys.argv[1],sys.argv[2],'--stage' not in sys.argv[3:],evidence)
