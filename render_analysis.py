"""Render the supplemental scientific report using the parent's existing style."""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
parent = (ROOT / 'report/test50/index.html').read_text(encoding='utf-8')
style = re.search(r'<style[^>]*>(.*?)</style>', parent, re.S).group(1)
body = markdown.markdown((ROOT / 'report/vessel_scale_analysis.md').read_text(encoding='utf-8'),
                         extensions=['tables', 'fenced_code'])
header = ('<p><a href="https://github.com/yuehanghahaha/flowoct-start-supervision-review">'
          '公开GitHub项目</a> · <a href="test50/index.html">原完整50例报告</a></p>')
document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>主干与细网差异：50例补充分析</title><style>' + style +
            '</style><body><main>' + header + body + '</main></body></html>')
(ROOT / 'report/index.html').write_text(document, encoding='utf-8')
print('Supplemental HTML report rendered; original report preserved.')
