# Devkit

Shared developer tooling for Realm Ruler mod workflows.

## Branch Model

- `dev`: default integration branch for day-to-day changes.
- `main`: release branch for stable, review-gated releases.

## Ownership Contract

- This repository is the single source of truth for devkit files.
- Other repositories (including `Shieldudaram/Colonists`) must not track `devkit/` paths.
- Boundary checks in CI enforce this contract.

## Release Workflow

1. Develop on feature branches from `dev` and merge PRs into `dev`.
2. Cut a release PR from `dev` into `main`.
3. Tag releases from `main`.

## Hotfix Workflow

1. Branch from `main` for urgent fixes.
2. Merge hotfix PR into `main`.
3. Immediately back-merge `main` into `dev` to avoid divergence.
