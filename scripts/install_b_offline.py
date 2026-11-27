"""Verify and install the adjacent B bundle with pip networking disabled."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--venv',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
for name,want in json.loads((root/'SHA256SUMS.json').read_text()).items():
 f=root/name
 if not f.resolve().is_relative_to(root) or hashlib.sha256(f.read_bytes()).hexdigest()!=want:raise ValueError('Bundle checksum mismatch: '+name)
if a.venv.exists():raise FileExistsError('Use a fresh venv, preserve existing environments')
subprocess.run([sys.executable,'-m','venv',str(a.venv)],check=True)
python=a.venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
subprocess.run([str(python),'-m','pip','install','--no-index','--find-links',str(root/'wheels'),'-r',str(root/'runtime.lock')],check=True)
subprocess.run([str(python),'-m','pip','check'],check=True)
print('Offline installation ready:',python.absolute())
