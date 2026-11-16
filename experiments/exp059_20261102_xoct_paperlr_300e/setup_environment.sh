#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
export TMPDIR="$PWD/runs/install_tmp"
mkdir -p "$TMPDIR"
if [ ! -x environment/bin/python ]; then
  /opt/conda/bin/conda create -y -p "$PWD/environment" python=3.11 pip --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
fi
# visdom 0.2.4 setup imports pkg_resources, removed by recent setuptools.
environment/bin/python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'setuptools==80.9.0' wheel
environment/bin/python -B download_torch.py
environment/bin/python -m pip install --no-deps --no-index --find-links runs/wheelhouse 'torch==2.7.0+cu126' 'torchvision==0.22.0+cu126' 'torchaudio==2.7.0+cu126'
environment/bin/python -B download_pins.py
environment/bin/python -m pip install --no-build-isolation --no-index --find-links runs/wheelhouse -r source/author/requirements.txt
environment/bin/python -m pip freeze > verification/pip_freeze.txt
environment/bin/python -m pip check > verification/pip_check.txt
environment/bin/python - <<'PY'
from pathlib import Path
import hashlib,importlib.metadata,json
requirements=Path('source/author/requirements.txt')
pins={line.split('==')[0]:line.split('==')[1] for line in requirements.read_text().splitlines() if '==' in line}
actual={name:importlib.metadata.version(name) for name in pins}
assert actual==pins,{name:(pins[name],actual[name]) for name in pins if pins[name]!=actual[name]}
Path('verification/environment_pins.json').write_text(json.dumps({'all_author_pins_match':True,'pins':actual,'requirements_sha256':hashlib.sha256(requirements.read_bytes()).hexdigest(),'setuptools_supplement':importlib.metadata.version('setuptools')},indent=2)+'\n')
PY
environment/bin/python -c 'import torch,torchvision,sys; print(sys.version,torch.__version__,torchvision.__version__,torch.version.cuda); assert torch.__version__=="2.7.0+cu126"'
touch verification/environment_install_complete
