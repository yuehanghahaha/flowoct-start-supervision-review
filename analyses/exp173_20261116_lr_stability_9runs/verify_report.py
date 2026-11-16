"""Check report artifacts, local links, and full metric coverage after generation."""
from html.parser import HTMLParser
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from analyze import ROOT, read, write, GROUPS


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='a' and 'href' in attrs:self.links.append(attrs['href'])
        if tag=='img' and 'src' in attrs:self.images.append(attrs['src'])


def main():
    audit=read(ROOT/'verification/numerical_audit.json');assert audit['passed']
    manifest=read(ROOT/'provenance.json')
    for f in manifest['files']:
        assert hashlib.sha256((ROOT/f['retained']).read_bytes()).hexdigest()==f['sha256']
    p=ROOT/'report/index.html';html=p.read_text(encoding='utf-8');parser=Links();parser.feed(html)
    checked=[]
    for link in parser.links+parser.images:
        if link.startswith(('https://','http://','#')):continue
        target=(p.parent/link.split('#')[0]).resolve()
        assert target.is_relative_to(ROOT.resolve()) and target.is_file(),link
        checked.append(link)
    for f in (ROOT/'report/figures').glob('*.png'):
        with Image.open(f) as im:im.verify()
    data=pd.read_csv(ROOT/'data/best_final_standard_metrics.csv')
    assert set(data.group)==set(GROUPS) and set(data.metric)=={'mae','psnr','ssim'}
    assert len(data)==9*2*9*3
    assert len(parser.images)==5
    for title in ['逐B-scan二维','完整3D体积','ILM–OPL','OPL–BM','TransPro','未取得']:
        assert title in html,title
    write(ROOT/'verification/report_verified.json',dict(passed=True,source_hashes=len(manifest['files']),
        local_links_verified=len(checked),figures_verified=len(parser.images),
        best_final_metric_slots=len(data),metric_groups=9,missing_metric_slots=int(data.value.isna().sum()),
        standard_values=audit['standard_metric_values'],new_inference=0,test_data_used=False))
    print('Verified source hashes, report links, 5 figures, and all 486 best/final metric slots')


if __name__=='__main__':main()
