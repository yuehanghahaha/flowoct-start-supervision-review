"""Download official VGG19 only; never load pretrained XOCT patient weights."""
import hashlib
from pathlib import Path
import urllib.request
from common import ROOT, write, now

url='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
dest=ROOT/'weights/hub/checkpoints/vgg19-dcbb9e9d.pth'
dest.parent.mkdir(parents=True,exist_ok=True)
if not dest.exists():
    tmp=dest.with_suffix('.partial')
    with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as f:
        while True:
            block=response.read(8*1024*1024)
            if not block:break
            f.write(block)
            print(f'VGG19 downloaded {f.tell()} bytes',flush=True)
    h=hashlib.sha256(tmp.read_bytes()).hexdigest()
    assert h.startswith('dcbb9e9d'), 'Official VGG19 checksum mismatch'
    tmp.replace(dest)
else:
    h=hashlib.sha256(dest.read_bytes()).hexdigest()
    assert h.startswith('dcbb9e9d')
write(ROOT/'verification/vgg19.json',dict(url=url,sha256=h,bytes=dest.stat().st_size,time=now(),xoct_pretrained_weights_used=False))
