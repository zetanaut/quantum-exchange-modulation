#!/usr/bin/env python3
"""Verify generated HTML links, figure assets and the saved evidence inventory."""
from __future__ import annotations
import argparse, hashlib, json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

ROOT=Path(__file__).resolve().parents[1]


class Document(HTMLParser):
    def __init__(self,text):
        super().__init__();self.links=[];self.ids=set();self.canonical=None;self.feed(text)
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=="link" and values.get("rel")=="canonical":self.canonical=values.get("href")
        if "id" in values:self.ids.add(values["id"])
        for field in ["href","src"]:
            if field in values:self.links.append((tag,values[field]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site",type=Path,default=ROOT/"site")
    args=p.parse_args();site=args.site.resolve()
    pages={path:Document(path.read_text()) for path in site.rglob("*.html")}
    if site/"index.html" not in pages:raise RuntimeError("Build the site before checking it.")
    prefix=urlsplit(pages[site/"index.html"].canonical or "/").path
    if not prefix.endswith("/"):prefix=prefix.rsplit("/",1)[0]+"/"
    problems=[];checked=0
    for page,document in pages.items():
        for tag,link in document.links:
            u=urlsplit(link)
            if u.scheme or u.netloc or link.startswith("//"):continue
            path=unquote(u.path)
            if path.startswith("/"):
                # MkDocs deliberately makes the error page's links absolute:
                # it can be served for any missing path, at any depth.
                if page.name!="404.html" or not path.startswith(prefix):
                    problems.append(f"{page.name}: root-absolute local URL outside the configured error page {link}");continue
                target=(site/path[len(prefix):]).resolve()
            else:
                target=(page.parent/path).resolve() if path else page
            if target.is_dir():target=target/"index.html"
            checked+=1
            if not target.is_file():problems.append(f"{page.name}: missing {link}")
            elif u.fragment and target in pages and unquote(u.fragment) not in pages[target].ids:
                problems.append(f"{page.name}: missing anchor {link}")
    assets=site/"assets"
    manifest=json.loads((assets/"manifest.json").read_text())
    for record in manifest["files"]:
        path=assets/record["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
            problems.append(f"Asset hash mismatch: {record['path']}")
    for name in ["network_architecture","exchange_block","learning_results","generalization",
                 "measurement_cost","learning_curve","correlation_sensitivity","correlation_copy_cost",
                 "dephasing_transfer","qfi_sensitivity","spin_gibbs","streaming_training","resource_scaling",
                 "directional_response_map","spectator_purity"]:
        if not (assets/"figures"/(name+".svg")).is_file():problems.append(f"Missing required figure: {name}")
    for name in ("directional_response_map.csv", "spectator_purity.csv", "manifest.json"):
        if not (assets/"results"/"physics_applications"/name).is_file():
            problems.append(f"Missing application diagnostic: {name}")
    for name in ("qip_Fig2", "qip_Fig3", "qip_Fig4"):
        for extension in ("png", "pdf"):
            if not (assets/"figures"/(name+"."+extension)).is_file():
                problems.append(f"Missing QIP figure: {name}.{extension}")
    for name in ("protocol.json", "summary.json", "reviewer_diagnostics.json",
                 "amortization_break_even.json", "release_gate.json",
                 "transfer_checks/finite_time.json", "transfer_checks/directional_response_map.csv"):
        if not (assets/"results/submission_revision"/name).is_file():
            problems.append(f"Missing QIP evidence: {name}")
    report=dict(pages=len(pages),local_links_checked=checked,assets_hashed=len(manifest["files"]),
                passed=not problems,problems=problems)
    print(json.dumps(report,indent=2))
    if problems:raise SystemExit(1)


if __name__=="__main__":
    main()
