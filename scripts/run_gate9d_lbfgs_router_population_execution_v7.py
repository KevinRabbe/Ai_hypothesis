#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, pathlib, subprocess, sys, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATH=ROOT/'ai_hypothesis/population_compute/gate9d_lbfgs_router_population_execution_v7.py'
BRANCH='agent/gate9d-lbfgs-router-population-execution-v7'

def load():
 spec=importlib.util.spec_from_file_location('gate9d_v7_cli',PATH); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output-root',type=pathlib.Path,required=True); a=ap.parse_args()
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(); status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 if branch!=BRANCH or status: raise RuntimeError('qualified branch and clean tree required')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); out=a.output_root.resolve(); archive=out.with_suffix('.zip')
 if out.exists() or archive.exists(): raise FileExistsError('output path exists')
 m=load(); summary=m.run(out,head)
 (out/'git-head.txt').write_text(head+'\n',encoding='ascii'); (out/'git-status.txt').write_text(status,encoding='utf-8')
 manifest=out/'manifest.sha256'; manifest.write_text('\n'.join(f'{sha(p)}  {p.relative_to(out).as_posix()}' for p in sorted(out.rglob('*')) if p.is_file() and p!=manifest)+'\n',encoding='ascii')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file(): z.write(p,arcname=f'{out.name}/{p.relative_to(out).as_posix()}')
 print(json.dumps({'status':summary['status'],'diagnosis':summary['diagnosis'],'summary_sha256':sha(out/'aggregate-summary.json'),'rows_sha256':sha(out/'rows.jsonl'),'manifest_sha256':sha(manifest),'archive_sha256':sha(archive),'archive':str(archive)},indent=2,sort_keys=True))
if __name__=='__main__': raise SystemExit(main())
