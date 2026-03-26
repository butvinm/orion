# Release Orion

Create a new release of orion packages (orion, orion-compiler, orion-evaluator).

**Arguments:** `$ARGUMENTS` — the version to release (e.g. `2.1.0`). Required.

## Prerequisites

Before starting, verify:

1. Working tree is clean (`git status` shows no uncommitted changes)
2. On `main` branch
3. CI is passing on the latest commit: `gh pr checks` or check the latest CI run
4. The version tag does not already exist: `git tag -l "v$ARGUMENTS"`

If any check fails, stop and tell the user.

## Version files to bump

Update the `version` field in these files (and only these — do NOT touch `python/lattigo/` or `js/lattigo/`):

- `pyproject.toml` (root)
- `python/orion-compiler/pyproject.toml`
- `python/orion-evaluator/pyproject.toml`

## Steps

1. Run all prerequisite checks
2. Bump version in the 3 files listed above
3. Commit: `chore: bump version to $ARGUMENTS`
   - Stage only the 3 specific files, never `git add .`
4. Push to main: `git push origin main`
5. Create and push tag: `git tag v$ARGUMENTS && git push origin v$ARGUMENTS`
6. Monitor the release workflow: `gh run list --workflow release.yml --limit 1`
   - The release workflow builds Python wheels, publishes to PyPI, publishes `js/lattigo` to npm, and creates a GitHub Release
7. Print the GitHub Release URL when done: `https://github.com/butvinm/orion/releases/tag/v$ARGUMENTS`

## What NOT to bump

- `python/lattigo/pyproject.toml` — version tracks upstream lattigo, not orion
- `js/lattigo/package.json` — version tracks upstream lattigo, not orion
