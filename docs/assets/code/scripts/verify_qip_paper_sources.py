#!/usr/bin/env python3
"""Audit QIP source references and selected claims against saved revision data.

This static check complements a pdfLaTeX/BibTeX build; it cannot replace one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/qip"
RESULT = ROOT / "results/submission_revision"
source = (PAPER / "quantum_exchange_modulation_paper.tex").read_text()
bib = (PAPER / "references.bib").read_text()

cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", source)
         for key in group.split(",")}
entries = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
assert cited == entries, {"missing": sorted(cited - entries), "uncited": sorted(entries - cited)}
arxiv_entries = {
    "cha_reupload": "2509.18530v6",
    "connerty_memory": "2605.12713v3",
    "haboury_routing": "2307.15682",
    "lee_observables": "2609.18655v1",
}
for key, identifier in arxiv_entries.items():
    entry = re.search(rf"(?ms)^@article\{{{key},(.*?)(?=^@\w+\{{|\Z)", bib)
    assert entry, f"Missing styled arXiv bibliography entry: {key}"
    assert "journal={arXiv preprint}" in entry.group(1), key
    assert f"eprint={{{identifier}}}" in entry.group(1), key
labels = set(re.findall(r"\\label\{([^}]+)\}", source))
references = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source))
assert references <= labels, sorted(references - labels)
figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
assert len(figures) == 4 and all((PAPER / name).is_file() for name in figures)
assert (PAPER / "sn-jnl.cls").is_file() and (PAPER / "sn-mathphys-num.bst").is_file()
assert r"\bibliography{references}" in source

summary = json.loads((RESULT / "summary.json").read_text())
for budget in (8, 16, 32, 64):
    for method in ("joint_exchange", "exchange_matched_qnn", "within_even__film", "gram_product_36"):
        value = summary["methods"][str(budget)][method]["mean_of_dataset_means"]
        mantissa, exponent = f"{value:.2e}".split("e")
        formatted = rf"${mantissa}\times10^{{{int(exponent)}}}$"
        assert formatted in source, (budget, method, formatted)
transfer = json.loads((RESULT / "transfer_checks/finite_time.json").read_text())
grid = json.loads((RESULT / "transfer_checks/directional_response_map.json").read_text())
assert transfer["queries"] == 8064 and grid["grid_rows"] == 231
assert r"6.27\times10^{-13}" in source and r"6.05\times10^{-12}" in source
assert r"qip-revision-v1.2" in source
print(json.dumps({"status": "PASS", "citation_entries": len(entries),
                  "cross_references": len(references), "figures": figures,
                  "table_2_cells_checked": 16,
                  "transfer_queries": transfer["queries"]}, indent=2))
