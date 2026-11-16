"""Launch only the authorized exp163 using its unchanged delivered launcher function."""
import json
import os
from pathlib import Path
import shutil
import sys
import common as c
import suite

record = Path(__file__).resolve().parent
assert os.environ.get('USER') == 'lvyuehang'
assert c.ROOT == Path('/workspace/experience/exp163_20261114_startproj_layermean_a100_200e')
assert (record/'acceptance.exit').read_text().strip() == '0'
cfg = c.read(c.ROOT/'experiment.json')
assert cfg['experiment_id'] == 'exp163_20261114_startproj_layermean_a100_200e'
assert cfg['epochs'] == cfg['steps_per_epoch'] == 200
assert cfg['run_test_automatically'] is False
contract = c.source_contract()
for name in ['preparation.json','synthetic.json','preflight.json','full_preflight.json']:
    receipt = c.read(c.ROOT/'verification'/name)
    assert receipt['passed'] and receipt['source_contract'] == contract
    if 'versions' in receipt:
        assert receipt['versions'] == c.versions()
    shutil.copy2(c.ROOT/'verification'/name, record/name)
c.write(record/'source_contract_before_training.json', contract)
suite.start([c.ROOT], 'lyh_tmux')
launch = c.read(c.ROOT/'verification/latest_launch.json')
assert launch['session'] == 'exp163_layerproj'
assert launch['action'] == 'train'
shutil.copy2(c.ROOT/'verification/latest_launch.json', record/'latest_launch.json')
print('Single exp163 launch completed; awaiting actual training-step evidence.', flush=True)
