#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, pathlib, subprocess, sys, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE=ROOT/'ai_hypothesis/population_compute/gate9d_router_architecture_sweep_v4.py'
BRANCH='agent/gate9d-router-architecture-sweep-v4'

def load():
    spec=importlib.util.spec_from_file_location('gate9d_router_sweep_cli',MODULE)
    if spec is None or spec.loader is None: raise RuntimeError('could not load router sweep')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',type=pathlib.Path,required=True); a=p.parse_args()
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=BRANCH: raise RuntimeError('wrong branch')
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if status: raise RuntimeError('working tree must be clean')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    out=a.output_root.resolve(); archive=out.with_suffix('.zip')
    if out.exists() or archive.exists(): raise FileExistsError('output exists')
    m=load(); summary=m.run(out,head)
    (out/'git-head.txt').write_text(head+'\n',encoding='ascii',newline='\n')
    (out/'git-status.txt').write_text('',encoding='ascii',newline='\n')
    (out/'run-config.json').write_text(json.dumps({'version':m.VERSION,'branch':BRANCH,'execution_head':head,'development_only':True},indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    rows=[]
    for path in sorted(x for x in out.rglob('*') if x.is_file() and x.name!='manifest.sha256'): rows.append(f'{sha(path)}  {path.relative_to(out).as_posix()}')
    (out/'manifest.sha256').write_text('\n'.join(rows)+'\n',encoding='ascii',newline='\n')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(x for x in out.rglob('*') if x.is_file()): z.write(path,arcname=f'{out.name}/{path.relative_to(out).as_posix()}')
    print(json.dumps({'status':summary['status'],'diagnosis':summary['diagnosis'],'winning_variant':summary['winning_variant'],'archive':str(archive),'archive_sha256':sha(archive)},indent=2,sort_keys=True))
if __name__=='__main__': main()
