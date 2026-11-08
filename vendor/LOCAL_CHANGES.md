# Vendored runtime changes

SMPL upstream source and original SHA-256 values are retained in
`smpl/SOURCE_VERSION.json`. All process equations, Atropine parameters, initial
states and model artifacts are unchanged.

Local changes:

1. `smpl/smpl/envs/utils.py`: replace the eager mzutils import with a local subset;
   move optional torch Dataset construction inside the dataset export factory.
   Normal environment execution requires neither torch nor d3rlpy.
2. `smpl/smpl/envs/_runtime_utils.py`: copy `normalize_spaces` and
   `denormalize_spaces` unchanged from mzutils 0.2022's
   `mzutils/gym_space_management.py`; retain its MIT license in `MZUTILS_LICENSE`.
   `mkdir_p` uses Python's `os.makedirs(..., exist_ok=True)`.
3. Other upstream environments remain available as research source. Their
   optional dependencies are not installed by the Atropine runtime requirements.

Do not install the upstream all-environment `requirements.txt` for this prototype;
use the repository root requirements. `SOURCE_VERSION.json` is the baseline hash
manifest, not a claim that locally modified files still match the original hash.
