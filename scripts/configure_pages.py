#!/usr/bin/env python3
"""Set GitHub project URLs locally. This command performs no remote action."""
import argparse, re
from pathlib import Path
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner",required=True)
    parser.add_argument("--repo",default="quantum-exchange-modulation")
    args=parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*",args.owner):
        raise ValueError("Invalid GitHub owner")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",args.repo):
        raise ValueError("Invalid repository name")
    root=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((root/"mkdocs.yml").read_text())
    user_site=args.repo.lower()==f"{args.owner.lower()}.github.io"
    config["site_url"]=f"https://{args.owner}.github.io/"+("" if user_site else f"{args.repo}/")
    config["repo_url"]=f"https://github.com/{args.owner}/{args.repo}"
    config["repo_name"]=f"{args.owner}/{args.repo}"
    config["edit_uri"]="edit/main/docs/"
    (root/"mkdocs.yml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    print(config["site_url"])


if __name__=="__main__":
    main()
