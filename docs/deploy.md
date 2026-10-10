# Deploy and maintain the GitHub Pages site

The repository builds a static **MkDocs Material** site. GitHub Pages serves the generated HTML, figures, search index, and download files. A separate workflow checks the scientific code. Visitors can read and explore the analytic controls immediately; they run Torch/PennyLane experiments in a local checkout or another Python environment.

The included `.github/workflows/pages.yml` follows GitHub's [custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages): build the site, upload a Pages artifact, and deploy it with the Pages action. The deployment job has `pages: write` and `id-token: write`, depends on the successful build, and uses the `github-pages` environment. Pull requests build and check the site without deploying it.

## 1. Validate the repository locally

```bash
python -m pip install -r requirements-docs.txt
python scripts/prepare_site_assets.py
python -m mkdocs build --strict
python scripts/check_site.py
python -m mkdocs serve
```

Inspect the overview, architecture, physics, evidence, and control pages. Check that equations render, figures remain readable on a narrow screen, downloads open, and the exchange/resource values update. The local checker validates relative links, anchors, required figures, and the hashes of files in the actual built `site/assets/` tree.

`site/` is generated output and is ignored by Git. The source, recorded evidence, and both workflow files belong in the repository. The website build only needs `requirements-docs.txt`; it uses the saved results and does not retrain models.

## 2. Set the owner and repository URL

Choose a repository under your GitHub account or an explicitly selected organization. Use the existing authenticated GitHub CLI session:

```bash
gh auth status
QEM_OWNER=$(gh api user --jq .login)
QEM_REPO=quantum-exchange-modulation
python scripts/configure_pages.py --owner "$QEM_OWNER" --repo "$QEM_REPO"
```

The configuration script only edits local MkDocs metadata. It sets the site URL, repository link, and edit link. A project repository uses `https://OWNER.github.io/REPOSITORY/`; a repository named `OWNER.github.io` uses the account-site root. Replace the variables deliberately if publishing to an organization or using another repository name.

The site uses relative file URLs (`use_directory_urls: false`), which makes its navigation and download paths work under the GitHub project prefix. Avoid introducing leading-slash asset links. Rebuild and rerun `scripts/check_site.py` after configuring the destination.

## 3. Create or use the intended repository

For a new, empty destination, the following sequence initializes this source and creates a public repository without pushing yet:

```bash
git init -b main
git add .
git commit -m "Add reproducible quantum exchange modulation site"
gh repo create "$QEM_OWNER/$QEM_REPO" --public --source . --remote origin
```

Git must already have an appropriate commit identity, or one should be configured locally for this repository from the authenticated account. If the destination already exists, inspect its contents and branch rules before using it. An existing project should be updated through its normal branch/PR process; these new-repository commands are not an instruction to replace unrelated work.

## 4. Enable Pages and push

In the repository's GitHub settings, select **Pages → Build and deployment → Source → GitHub Actions**. This one-time configuration is required; the default `configure-pages` action reads the site's settings rather than automatically enabling Pages.

For a newly created destination, the equivalent authenticated REST call is:

```bash
gh api --method POST "repos/$QEM_OWNER/$QEM_REPO/pages" -f build_type=workflow
git push -u origin main
```

If a Pages site already exists, inspect it with `gh api "repos/$QEM_OWNER/$QEM_REPO/pages"` and use the documented update endpoint when its build source needs to change. Do not treat every API failure as an already-existing site; permissions, organizational policy, and an unrelated destination require different handling. GitHub documents [Pages creation and updates](https://docs.github.com/en/rest/pages/pages).

The push to `main` starts the documentation workflow and, for matching code paths, scientific verification. Subsequent documentation-only pushes build the site without rerunning expensive scientific fits. `workflow_dispatch` is available for manually rerunning either workflow.

## 5. Verify the deployment

```bash
gh run list --workflow pages.yml --limit 3
gh api "repos/$QEM_OWNER/$QEM_REPO/pages" --jq .html_url
```

Watch the run corresponding to the pushed commit with `gh run watch RUN_ID --exit-status`. Open the returned Pages URL and inspect several internal pages, a figure, a numerical download, and the control explorer. The expected public URL alone does not establish that deployment succeeded: check the workflow conclusion, actual HTTP response, and rendered content.

If a workflow fails, inspect its failed-job logs, fix the concrete source/configuration issue, validate locally, and push a normal follow-up commit. Typical first-deployment issues are a Pages source that is not set to Actions, insufficient repository permissions, or an organization environment approval rule. Follow the actual repository policy rather than trying to bypass it.

## Maintain the site and evidence

Keep credentials in the authenticated environment. GitHub Actions uses its scoped built-in token for deployment; this site does not need a personal access token in source code, JavaScript, or a repository secret. Public assets include the bundled simulation dataset and checkpoints, so new private experimental data should be reviewed before intentionally adding them to the published evidence set.

For a documentation change, edit `docs/`, prepare assets, build strictly, and run the site checker. For a model change, also run `scripts/run_all.py` and a relevant example. If a scientific comparison changes, retain the previous record, rerun the declared protocol, publish the new seeds/data/selection procedure, and update the figures and claim wording together. The [reproduction guide](reproduction.md) explains the canonical result directories.

Remote or hardware experiments can be added as explicitly labeled extensions. Record device, shots, compilation/routing, state-preparation costs, and training resources. Keep them distinct from the supplied exact-simulator fits until the new comparison is actually executed.
