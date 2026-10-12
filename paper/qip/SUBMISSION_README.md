# QIP revision: author review and submission handoff

The compiled manuscript is `quantum_exchange_modulation_paper.pdf`; the editable Springer source is `quantum_exchange_modulation_paper.tex`. The source archive includes the class, bibliography style, references, four vector PDF figures, and the three figure-generation scripts. Figures 2 and 4 use the fresh QIP revision records. Figure 3 was regenerated from the original saved one-state direction-map CSV. The separate Git bundle contains the code, datasets, checkpoints, analyses, and frozen protocol.

From the repository checkout, run `python paper/qip/redraw_learning.py`, `python paper/qip/redraw_copies.py`, and `python paper/qip/redraw_direction.py` to regenerate Figs. 2–4. If using the flat manuscript source archive outside the checkout, set `QFILM_REPO` to the checkout path first. The scripts need NumPy and Matplotlib.

The numerical record was introduced at commit `6db56fd39ef10f117a53b05b801cdc72442176ee`; the branch also includes this manuscript source. For a standalone inspection, run `git clone qfilm-revision.bundle qfilm-review`, then `git -C qfilm-review switch codex/submission-revision` (the bundle has no default `HEAD`). To import it into the existing repository, run `git fetch /path/to/qfilm-revision.bundle codex/submission-revision` and review the fetched commit before merging.

## Scientific findings to verify

- Five independent confirmation datasets reserve physical axes with `|n_x|>0.55` for the held-out tests, avoiding the antipodal overlap in the historical oriented cap.
- At 64 training states, the compact exchange joint-test MSE averages `1.56e-9` across five datasets and three starts; the matched QNN averages `1.75e-8`. The 36-feature Gram product regression recovers the analytic target to numerical precision even at eight training states. This does not support a general quantum data-efficiency advantage.
- At 3,840 source copies per single query, the compact model's expected MSE is `8.63e-4`; direct collective `K_P` is `8.19e-4`. With 8 or 32 repeated queries sharing that total budget per state, reusable Pauli features outperform context-specific direct `K_P` under the stated estimators. This is inference-only ideal measurement accounting.
- Ablations are sensitive to initialization and prescribed structure under the frozen training schedules. Some neutral starts succeed; the paper does not claim a general expressivity separation.

## Required before journal submission

1. The corresponding author must supply and verify email, full department and postal affiliation, and optional ORCID on the title page.
2. The author must replace the three declaration placeholders for funding, competing interests, and contributions with accurate statements.
3. The author must review every scientific claim, the use of an AI coding assistant disclosed in the Reproducibility section, and the exact dataset and measurement assumptions. The assistant cannot certify authorship or the author's interpretation.
4. Publish or otherwise deposit the versioned revision code and data, then replace the release placeholder in the Reproducibility and Data and code availability statements with an actual persistent location and commit/version identifier.
5. Complete QIP's submission-system fields and verify the current [journal instructions](https://link.springer.com/journal/11128/submission-guidelines). The journal-specific abstract is 93 words and the paper has six keywords; the site also contains a conflicting generic template abstract range, so the author should resolve that at submission.

Set the main document to `quantum_exchange_modulation_paper.tex` and the compiler to **pdfLaTeX** in Overleaf. The local reading PDF was compiled with Tectonic and visually inspected page by page; an exact pdfLaTeX build remains to be checked in Overleaf. It has 14 A4 pages, with no undefined references or overfull-box warnings. The numerical release gate passes, including five datasets, 240 quantum fits, 480 classical fits, 30 generic/control fits, checkpoint reload checks, and the independent circuit and dataset checks. No journal submission or public repository update has been made.
