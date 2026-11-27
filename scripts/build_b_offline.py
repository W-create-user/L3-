"""Build a same-platform offline research bundle, without private trajectories.
Run with the validated B runtime interpreter; MPCTools wheel/source are explicit.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
import importlib.metadata as metadata
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mpctools-wheel',type=Path,required=True);p.add_argument('--mpctools-source',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=False)
 source=a.output/'source';source.mkdir();wheels=a.output/'wheels';wheels.mkdir()
 names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
 names+=[str(f.relative_to(root)) for f in (root/'scripts').glob('*.py')]
 for prefix in ['configs/controllers','configs/environments','configs/scenarios','configs/training','patches']:
  names += [str(f.relative_to(root)) for f in (root/prefix).rglob('*') if f.is_file()]
 names+=['docs/data_dictionary.csv','requirements-runtime-20261120.lock']
 for name in sorted(set(filter(None,names))):
  src=root/name
  if src.is_file():
   dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
 lock='\n'.join(('MPCTools=='+metadata.version('MPCTools')) if line.startswith('MPCTools @ file:') else line for line in lock.splitlines())+'\n'
 if any(' @ ' in line or line.startswith('-e') for line in lock.splitlines()):raise ValueError('Lock contains nonportable local/direct requirement')
 (a.output/'runtime.lock').write_text(lock)
 shutil.copy2(a.mpctools_wheel,wheels/a.mpctools_wheel.name)
 subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--find-links',str(wheels),'--wheel-dir',str(wheels),'-r',str(a.output/'runtime.lock')],check=True)
 # GPL source and modifications accompany the MPCTools binary wheel.
 mpc=a.output/'third_party/mpctools';mpc.mkdir(parents=True)
 for f in a.mpctools_source.rglob('*'):
  rel=f.relative_to(a.mpctools_source)
  if f.is_file() and not any(s in rel.parts for s in ['.git','__pycache__','build','dist']) and f.suffix!='.pyc':
   dst=mpc/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
 shutil.copy2(root/'scripts/install_b_offline.py',a.output/'install.py')
 (a.output/'READ_ME.txt').write_text('B offline research bundle. Same Python minor/platform as build host required.\nInstall: python3.12 install.py --venv <new-directory>\nThen enter source; run <venv>/bin/python -m experiments.verify_b_delivery --output <new-output>\nNo raw data/weights or device control. Formal holdout requires separate A authorization; not launched by this bundle.\nRun acceptance using the command above; it exits after all checks. For interactive entry points follow the repository README. Stop with Ctrl+C. Logs reside in explicit output directories. Preserve prior bundles for upgrades.\nThis is an offline equivalent, not a Docker image or independent C/D acceptance.\n')
 hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()}
 (a.output/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print('Offline bundle:',a.output)
if __name__=='__main__':main()
