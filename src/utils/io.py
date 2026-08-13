from pathlib import Path
import pandas as pd
from jinja2 import Template

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

def save_html(df, path, title="Report"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tpl = Template("""
    <html><head><meta charset="utf-8"><title>{{title}}</title></head>
    <body><h1>{{title}}</h1>
    {{table}}
    </body></html>
    """)
    html = tpl.render(title=title, table=df.to_html())
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
