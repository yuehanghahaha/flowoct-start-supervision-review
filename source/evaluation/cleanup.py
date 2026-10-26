"""Remove only regenerable bytecode and the superseded partial metric check."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    assert ROOT.name=='exp010_20261025_3dfm_start70_test50' and ROOT.parent.name=='experience'
    check=json.loads((ROOT/'verification/report_checks.json').read_text())
    assert check['state']=='passed' and check['test_patients']==50
    removed=[]
    paths=list(ROOT.rglob('*.pyc'))+[ROOT/'verification/partial_numeric_check.json']
    for p in paths:
        path=p.resolve();assert path.is_relative_to(ROOT)
        assert '__pycache__' in path.parts or path.name=='partial_numeric_check.json'
        if path.is_file():
            removed.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size));path.unlink()
    for p in sorted(ROOT.rglob('__pycache__'),key=lambda p:len(p.parts),reverse=True):
        path=p.resolve();assert path.is_relative_to(ROOT)
        if path.is_dir() and not any(path.iterdir()):path.rmdir()
    record=dict(removed=removed,removed_bytes=sum(r['bytes'] for r in removed),
        reason='Regenerable bytecode and incomplete check superseded by complete 50-case verification',
        protected=['checkpoints','results','evidence','report','logs'],delivery_archives_created=0,
        full_volume_predictions_saved=0)
    (ROOT/'verification/cleanup.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Cleaned',len(removed),'small temporary files; all scientific results retained')


if __name__=='__main__':main()
