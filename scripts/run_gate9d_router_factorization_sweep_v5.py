#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, pathlib, subprocess, sys, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE=ROOT/'ai_hypothesis/population_compute/gate9d_router_factorization_sweep_v5.py'
BRANCH='agent/gate9d-router-factorization-sweep-v5'

def load():
    spec=importlib.util.spec_from_file_location('gate9d_router_factorization_cli',MODULE)
    if spec is None or spec.loader is None: raise RuntimeError('could not load factorization sweep')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-root',type=pathlib.Path,required=True); args=parser.parse_args()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if branch!=BRANCH: raise RuntimeError(f'must run from {BRANCH}')
    if status: raise RuntimeError('working tree must be clean')
    output=args.output_root.resolve(); archive=output.with_suffix('.zip')
    if output.exists() or archive.exists(): raise FileExistsError('output or archive exists')
    module=load(); summary=module.run(output,head)
    (output/'git-head.txt').write_text(head+'\n',encoding='ascii',newline='\n')
    (output/'git-status.txt').write_text(status,encoding='utf-8',newline='\n')
    (output/'run-config.json').write_text(json.dumps({'version':module.VERSION,'branch':BRANCH,'execution_head':head,'development_only':True},indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    manifest=output/'manifest.sha256'; files=sorted(p for p in output.iterdir() if p.is_file() and p!=manifest)
    manifest.write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files),encoding='ascii',newline='\n')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            if p.is_file(): z.write(p,arcname=f'{output.name}/{p.name}')
    print(json.dumps({'status':summary['status'],'diagnosis':summary['diagnosis'],'winning_variant':summary['winning_variant'],'archive':str(archive),'archive_sha256':sha(archive),'manifest_sha256':sha(manifest)},indent=2,sort_keys=True))
    return 0
if __name__=='__main__': raise SystemExit(main())
