# ANONYMIZATION_CHECK — artifact and manuscript

Checked 2026-09-22 against `artifact/pcrl_satml_anon/` and `papers/pcrl_satml_final_v1/`.

## Automated scans (all clean)

| pattern class | result |
|---|---|
| author name, surname, username, Git identity, email | **0 hits** |
| institution / lab / advisor names | **0 hits** |
| absolute local paths (`/Users/...`), worktree names, hostnames | **0 hits** |
| AWS account/instance identifiers, S3 URIs, bucket names, credentials | **0 hits** |
| repository URL or remote, GitHub org/repo | **0 hits** |

Manuscript: the author block is `Anonymous Submission`; no acknowledgements; the Open Science section
describes the artifact without naming a host or account.

## Allowlist, not denylist

The archive was assembled by copying an explicit list of files (generators, verification scripts, tables,
figures, ledger, corrections, scope, fixtures, README) rather than by excluding from a working tree. Nothing
was copied wholesale.

## Deliberately excluded

* Fitted model weights, channel solutions and per-person predictions (individual-level or operational; not
  needed to check any reported number).
* Private handoff directories, author/policy notes, and the registration package's author checklist.
* Raw or derived person records. The archive names the **public** ACS PUMS source and how to obtain it.
* Git history, remotes and commit metadata.

The README states plainly that this is not a claim that every private archive has been released.

## Residual risks the author should weigh

1. **Self-citation.** The paper cites the prior-work literature but does not cite the authors' own earlier
   submission. Venue rules require citing one's own related work in the third person rather than omitting
   it. If the earlier paper is public by submission time, add it as a third-person citation; if it is not
   public, its absence is defensible. Either way the decision is the author's.
2. **A public repository.** This project's repository is public by the author's earlier explicit decision.
   Anonymity of the *submission* concerns the PDF and the artifact, but a reviewer who searches distinctive
   phrases could find the repository. Consider whether to keep it public during review.
3. **The artifact archive is built but not published.** Uploading it is an external action reserved to the
   author.
