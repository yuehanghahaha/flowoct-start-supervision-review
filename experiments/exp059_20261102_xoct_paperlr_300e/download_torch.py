"""Verified ranged retrieval of the exact public PyTorch wheels."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'runs/wheelhouse'
DEST.mkdir(parents=True,exist_ok=True)


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href')


def retrieve(url,workers=16):
    clean,fragment=urllib.parse.urldefrag(url)
    expected=urllib.parse.parse_qs(fragment)['sha256'][0]
    name=urllib.parse.unquote(clean.rsplit('/',1)[1]);target=DEST/name
    if target.exists() and sha(target)==expected:return target
    # The primary official CDN supports ranged downloads without the R2 redirect.
    clean=clean.replace('download-r2.pytorch.org','download.pytorch.org')
    probe=urllib.request.Request(clean,headers={'Range':'bytes=0-0'})
    with urllib.request.urlopen(probe,timeout=30) as r:
        assert r.status==206
        length=int(r.headers['Content-Range'].split('/')[-1]);r.read()
    partial=DEST/(name+f'.{os.getpid()}.partial')
    with partial.open('wb') as f:f.truncate(length)
    chunk=4*1024*1024
    def part(start):
        end=min(start+chunk,length)-1
        for attempt in range(6):
            try:
                request=urllib.request.Request(clean,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=45) as r:
                    assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                    data=r.read();assert len(data)==end-start+1
                with partial.open('r+b',buffering=0) as f:f.seek(start);f.write(data)
                return len(data)
            except Exception:
                if attempt==5:raise
                time.sleep(1+attempt)
    done=0;begin=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(part,s) for s in range(0,length,chunk)]
        for f in as_completed(futures):
            done+=f.result()
            print(json.dumps({'wheel':name,'downloaded':done,'total':length,'seconds':round(time.monotonic()-begin,1)}),flush=True)
    actual=sha(partial);assert actual==expected,(name,actual,expected)
    partial.replace(target)
    return target


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


if __name__=='__main__':
    records=[]
    for package,version in [('torch','2.7.0'),('torchvision','0.22.0'),('torchaudio','2.7.0')]:
        url=f'https://download.pytorch.org/whl/cu126/{package}/'
        with urllib.request.urlopen(url,timeout=30) as r:page=r.read().decode()
        links=Links();links.feed(page)
        match=[urllib.parse.urljoin(url,v) for v in links.links if
               f'{package}-{version}+cu126-cp311-cp311-manylinux_2_28_x86_64.whl' in urllib.parse.unquote(v)]
        assert len(match)==1,(package,len(match))
        result=retrieve(match[0]);records.append(dict(package=package,url=match[0],path=str(result),sha256=sha(result)))
    (ROOT/'verification/torch_wheel_downloads.json').write_text(json.dumps(records,indent=2)+'\n')
