# QIP revision: author review and submission handoff

The editable Springer source is `quantum_exchange_modulation_paper.tex`. The repository includes the class, bibliography style, references, four vector PDF figures, and the three figure-generation scripts. Figures 2–4 use the fresh QIP revision records. Figure 3 and the finite-time diagnostic use the validation-selected seed-17 checkpoint from confirmation dataset 1. The complete validation package is in `results/submission_revision/`; tag `qip-revision-v1` identifies the public revision. The previously generated manuscript PDF in the checkout predates these edits; use the current Overleaf preview for review.

From the repository checkout, run `python paper/qip/redraw_learning.py`, `python paper/qip/redraw_copies.py`, and `python paper/qip/redraw_direction.py` to regenerate Figs. 2–4. If using the flat manuscript source archive outside the checkout, set `QFILM_REPO` to the checkout path first. The scripts need NumPy and Matplotlib.

The numerical record was introduced at commit `6db56fd39ef10f117a53b05b801cdc72442176ee`; the versioned tag contains the final manuscript source and subsequent audits. Clone the repository and check out `qip-revision-v1` for a frozen inspection, or use the current `main` branch for the latest public project.

## Scientific findings to verify

- Five independent confirmation datasets reserve physical axes with `|n_x|>0.55` for the held-out tests, avoiding the antipodal overlap in the historical oriented cap.
- At 64 training states, the compact exchange joint-test MSE averages `1.56e-9` across five datasets and three starts; the matched QNN averages `1.75e-8`. The 36-feature Gram product regression recovers the analytic target to numerical precision even at eight training states. This does not support a general quantum data-efficiency advantage.
- At 3,840 source copies per single query, the compact model's expected MSE is `8.63e-4`; direct collective `K_P` is `8.19e-4`. With 8 or 32 repeated queries sharing that total budget per state, reusable Pauli features outperform context-specific direct `K_P` under the stated estimators. This is inference-only ideal measurement accounting.
- A fresh-split confirmation checkpoint gives finite-time purity-loss MSE `6.27e-13` over 8,064 queries and one-state direction-grid MSE `6.05e-12`. The new grid includes training-region axes and is exploratory.
- Ablations are sensitive to initialization and prescribed structure under the frozen training schedules. Ten of 15 neutral starts pass a descriptive validation-MSE threshold of `1e-4`, versus all 15 aligned starts; the neutral mean joint-test MSE hides substantial failures.
- An exploratory integer break-even check using the frozen nested query pool finds direct `K_P` lower at two queries and reusable Pauli data lower at three. See `results/submission_revision/amortization_break_even.json`.

## Required before journal submission

1. The corresponding author must supply and verify email, full department and postal affiliation, and optional ORCID on the title page.
2. The author must replace the three declaration placeholders for funding, competing interests, and contributions with accurate statements.
3. The author must review every scientific claim, the use of an AI coding assistant disclosed in the Reproducibility section, and the exact dataset and measurement assumptions. The assistant cannot certify authorship or the author's interpretation.
4. Confirm the public `qip-revision-v1` tag and its complete `results/submission_revision/` archive are accessible. The older `de6cfa787a13` snapshot supports historical results only.
5. Complete QIP's submission-system fields and verify the current [journal instructions](https://link.springer.com/journal/11128/submission-guidelines). The revised abstract is 97 words and the paper has six keywords; the site also contains a conflicting generic template abstract range, so the author should resolve that at submission.
6. Resolve a reference-policy conflict with the journal: the manuscript now numbers four directly relevant arXiv preprints, as requested, while QIP's current guidelines say its reference list should contain published or accepted works only. Verify whether any have been accepted or ask the editor whether these preprints may remain in the numbered list.

Set the main document to `quantum_exchange_modulation_paper.tex` and the compiler to **pdfLaTeX** in Overleaf; use Recompile from scratch after uploading the revised `.bib`. An exact pdfLaTeX build remains to be checked in Overleaf. The numerical release gate passes, including five datasets, 240 quantum fits, 480 classical fits, 30 generic/control fits, checkpoint reload checks, and the independent circuit and dataset checks. No journal submission has been made.
