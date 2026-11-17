"""Decode a binary release mask using explicit foreground/background codes."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--foreground',required=True,help='JSON list of integer codes or RGB triples')
    p.add_argument('--background',required=True,help='JSON list of integer codes or RGB triples')
    args=p.parse_args();fg=json.loads(args.foreground);bg=json.loads(args.background)
    image=Image.open(args.input);rgb=bool(fg and isinstance(fg[0],list))
    array=np.asarray(image.convert('RGB') if rgb else image)
    def matches(codes):
        result=np.zeros(array.shape[:2],bool)
        for code in codes:
            if rgb:
                if array.ndim!=3 or len(code)!=3:raise ValueError('Expected RGB triples.')
                result|=(array==np.asarray(code)).all(-1)
            else:
                if array.ndim!=2:raise ValueError('Expected an index image.')
                result|=array==code
        return result
    foreground=matches(fg);background=matches(bg)
    if (foreground&background).any() or not (foreground|background).all():raise ValueError('Ambiguous or unmapped mask values.')
    if args.output.suffix!='.npy' or args.output.exists():raise ValueError('Choose a new .npy output.')
    args.output.parent.mkdir(parents=True,exist_ok=True);np.save(args.output,foreground.astype(np.uint8),allow_pickle=False)
    args.output.with_suffix('.provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
         'foreground':fg,'background':bg,'orientation':'unchanged'},indent=2)+'\n')


if __name__=='__main__':main()
