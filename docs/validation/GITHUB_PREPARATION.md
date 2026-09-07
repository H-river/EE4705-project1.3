# Private GitHub upload preparation

Prepared on 7 September 2026 for `/home/jiamo/EE4705/project1.3`.
The intended repository is `EE4705-project1.3`, private, under the GitHub CLI
account once authenticated. At preparation time there was no remote and
GitHub CLI reported no authenticated hosts. **No upload is claimed here.**

## Upload scope

Source, tests, trial definitions, configuration, generated robot XML,
README/student guides, validation reports, and selected small report images
and CSV files are tracked. Existing Git history is retained.

The course PDF remains local. `runs/` (including recordings, API audits,
depth images and temporary upload tools), `.venv/`, caches, logs, credential
files and the fetched `assets/menagerie/` checkout are excluded. No Git LFS
configuration was introduced. No existing commit was rewritten.

The README now explains environment-variable API setup and how to regenerate
offline artifacts. Historical reports explicitly label `runs/`, localhost
links and author-machine paths as local artifacts, not files hosted by GitHub.
The original paid-model response bundles are not distributed in this upload.

## Checks completed

- Baseline history: 14 commits, 262 unique blobs, 3,151,979 total uncompressed
  blob bytes. Largest historical blob: 97,275 bytes. No blob exceeds 50 MiB.
  No tracked/history paths under `runs/`, `.venv/` or the fetched Menagerie
  checkout were found in the inventory.
- Gitleaks 8.30.1 scanned reachable `HEAD` history and an export of the current
  tracked working files, with redacted reports: zero findings in each scan.
  An additional scan of all reachable blobs for provider tokens and private
  key headers also returned zero findings. These are scan results, not a
  mathematical guarantee that arbitrary secrets cannot exist.
- Thirteen representative local-output/credential/course-PDF paths were
  checked against the ignore rules; all were excluded. The PDF still exists.
- Full offline tests: **221 passed, 1 skipped in 42.30 s**. The skip is the
  existing experimental contact-only grasp check. No live-model or hardware
  verification is claimed by this preparation.
- Menagerie checkout revision matches `assets/menagerie_revision.txt`:
  `8161bba264d7fa7c99ca301e91e7fb44737676ad`.
  Both tracked license notices match the fetched upstream notices byte for
  byte: Unitree G1 BSD-3-Clause and Robotiq BSD-2-Clause.
- GitHub CLI 2.100.0 and Gitleaks 8.30.1 were downloaded from their official
  release repositories and archive checksums verified. Binaries and audit
  logs stay under ignored `runs/github_upload_audit/`.

GitHub warns above 50 MiB and blocks ordinary Git files above 100 MiB;
this repository is below both thresholds. See the
[GitHub large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
No repository-wide license has been assigned to original coursework code;
upstream notices remain included, and the intended upload is private.

## Pending authenticated step

Authenticate GitHub CLI, inspect the active account and any same-name remote
repository, then create a private repository only if appropriate. Push `main`
without force and compare the remote branch SHA to local `HEAD`. If a same-name
repository already contains unrelated work, stop instead of overwriting it.
