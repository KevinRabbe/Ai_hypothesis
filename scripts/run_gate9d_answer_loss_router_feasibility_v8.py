#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,importlib.util,json,pathlib,subprocess,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATH=ROOT/'ai_hypothesis/population_compute/gate9d_answer_loss_router_feasibility_v8.py'
BRANCH='agent/gate9d-answer-loss-router-feasibility-v8'

def load():
 spec=importlib.util.spec_from_file_location('gate9d_v8_cli',PATH)
 if spec is None or spec.loader is None: raise RuntimeError('cannot load v8 runtime')
 m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

def git(*a): return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-root',type=pathlib.Path,required=True); a=p.parse_args()
 if git('branch','--show-current')!=BRANCH: raise RuntimeError(f'must run from {BRANCH}')
 if git('status','--porcelain'): raise RuntimeError('working tree must be clean')
 head=git('rev-parse','HEAD'); out=a.output_root.resolve(); arc=out.with_suffix('.zip')
 if out.exists() or arc.exists(): raise FileExistsError('output or archive exists')
 m=load(); summary=m.run(out,head)
 (out/'git-head.txt').write_text(head+'\n',encoding='ascii',newline='\n'); (out/'git-status.txt').write_text('',encoding='ascii',newline='\n')
 lines=[]
 for f in sorted(x for x in out.rglob('*') if x.is_file() and x.name!='manifest.sha256'): lines.append(f'{sha(f)}  {f.relative_to(out).as_posix()}')
 manifest=out/'manifest.sha256'; manifest.write_text('\n'.join(lines)+'\n',encoding='ascii',newline='\n')
 with zipfile.ZipFile(arc,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(x for x in out.rglob('*') if x.is_file()): z.write(f,arcname=f'{out.name}/{f.relative_to(out).as_posix()}')
 print(json.dumps({'status':summary['status'],'diagnosis':summary['diagnosis'],'reliable_variants':summary['reliable_variants'],'summary_sha256':sha(out/'aggregate-summary.json'),'rows_sha256':sha(out/'rows.jsonl'),'manifest_sha256':sha(manifest),'archive_sha256':sha(arc),'archive':str(arc)},indent=2,sort_keys=True))
 return 0
if __name__=='__main__': raise SystemExit(main())
