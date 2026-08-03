#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, pathlib, platform, subprocess, sys, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE=ROOT/'ai_hypothesis/population_compute/gate9d_router_convergence_robustness_v6.py'
BRANCH='agent/gate9d-router-convergence-robustness-v6'

def load():
    spec=importlib.util.spec_from_file_location('gate9d_router_convergence_cli',MODULE)
    if spec is None or spec.loader is None: raise RuntimeError('cannot load v6')
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',type=pathlib.Path,required=True); a=ap.parse_args()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if branch!=BRANCH or status: raise RuntimeError('v6 requires exact branch and clean tree')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    out=a.output_root.resolve(); archive=out.with_suffix('.zip')
    if out.exists() or archive.exists(): raise FileExistsError('output exists')
    m=load(); summary=m.run(out,head)
    (out/'git-head.txt').write_text(head+'\n',encoding='ascii',newline='\n')
    (out/'git-status.txt').write_text(status,encoding='utf-8',newline='\n')
    (out/'run-config.json').write_text(json.dumps({'version':m.VERSION,'branch':BRANCH,'execution_head':head,'python':platform.python_version(),'torch':m.torch.__version__,'output_root':str(out),'development_only':True},indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    manifest=out/'manifest.sha256'; rows=[]
    for p in sorted(x for x in out.rglob('*') if x.is_file() and x!=manifest): rows.append(f'{sha(p)}  {p.relative_to(out).as_posix()}')
    manifest.write_text('\n'.join(rows)+'\n',encoding='ascii',newline='\n')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(x for x in out.rglob('*') if x.is_file()): z.write(p,arcname=f'{out.name}/{p.relative_to(out).as_posix()}')
    print(json.dumps({'status':summary['status'],'diagnosis':summary['diagnosis'],'reliable_variants':summary['reliable_variants'],'archive':str(archive),'archive_sha256':sha(archive),'manifest_sha256':sha(manifest)},indent=2,sort_keys=True))
    return 0
if __name__=='__main__': raise SystemExit(main())
