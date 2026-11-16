"""Download unchanged author pins from a mirror with published SHA256 verification."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.metadata
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename,parse_sdist_filename,canonicalize_name
from download_torch import ROOT,DEST,Links,retrieve,sha

TAGS={tag:i for i,tag in enumerate(sys_tags())}


def download(line):
    req=Requirement(line);version=next(iter(req.specifier)).version
    try:
        if importlib.metadata.version(req.name)==version:
            return dict(package=req.name,version=version,already_installed=True)
    except importlib.metadata.PackageNotFoundError:pass
    if canonicalize_name(req.name) in ['torch','torchvision','torchaudio']:return dict(package=req.name,official_torch_wheel=True)
    index='https://pypi.tuna.tsinghua.edu.cn/simple/'+canonicalize_name(req.name)+'/'
    with urllib.request.urlopen(index,timeout=30) as r:body=r.read().decode()
    parser=Links();parser.feed(body);choices=[];sources=[]
    for link in parser.links:
        url=urllib.parse.urljoin(index,link);name=urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit('/',1)[-1])
        try:
            if name.endswith('.whl'):
                package,ver,build,tags=parse_wheel_filename(name)
                if str(ver)==version and canonicalize_name(package)==canonicalize_name(req.name):
                    ranks=[TAGS[t] for t in tags if t in TAGS]
                    if ranks:choices.append((min(ranks),url))
            elif name.endswith(('.tar.gz','.zip')):
                package,ver=parse_sdist_filename(name)
                if str(ver)==version and canonicalize_name(package)==canonicalize_name(req.name):sources.append(url)
        except ValueError:continue
    if choices:url=min(choices)[1]
    else:
        assert sources, 'No compatible distribution: '+line
        url=sources[0]
    path=retrieve(url,workers=6)
    return dict(package=req.name,version=version,url=url,path=str(path),sha256=sha(path))


if __name__=='__main__':
    import fcntl
    lock=(DEST/'pins_download.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    requirements=[l.strip() for l in (ROOT/'source/author/requirements.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]
    records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(download,line) for line in requirements]):
            record=future.result();records.append(record)
            print(json.dumps({'completed':len(records),'required':len(requirements),'record':record}),flush=True)
    (ROOT/'verification/pinned_downloads.json').write_text(json.dumps(records,indent=2)+'\n')
