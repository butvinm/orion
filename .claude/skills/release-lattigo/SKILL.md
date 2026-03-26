# Release Lattigo Bindings

Create a new release of lattigo binding packages (orion-v2-lattigo on PyPI and npm).

**Arguments:** `$ARGUMENTS` — the version to release (e.g. `6.3.0`). Must match the upstream Lattigo version these bindings target. Required.

## Prerequisites

Before starting, verify:

1. Working tree is clean (`git status` shows no uncommitted changes)
2. On `main` branch
3. CI is passing on the latest commit: `gh pr checks` or check the latest CI run
4. The version tag does not already exist: `git tag -l "lattigo-v$ARGUMENTS"`
5. Both lattigo packages have the same version set to `$ARGUMENTS`:
   - `python/lattigo/pyproject.toml`
   - `js/lattigo/package.json`

If the versions don't match `$ARGUMENTS`, bump them first.

## Version files to bump (if needed)

- `python/lattigo/pyproject.toml`
- `js/lattigo/package.json`

## Steps

1. Run all prerequisite checks
2. If versions need bumping, update both files and commit: `chore: bump lattigo bindings to $ARGUMENTS`
   - Stage only the 2 specific files, never `git add .`
3. Push to main: `git push origin main`
4. Create and push tag: `git tag lattigo-v$ARGUMENTS && git push origin lattigo-v$ARGUMENTS`
5. Monitor the release workflow: `gh run list --workflow release-lattigo.yml --limit 1`
   - The workflow builds Python wheel (CGO) and JS/WASM package, publishes to PyPI and npm, and creates a GitHub Release
6. Print the GitHub Release URL when done: `https://github.com/butvinm/orion/releases/tag/lattigo-v$ARGUMENTS`

## What this does NOT publish

- `orion-compiler` and `orion-evaluator` have their own release workflow (`release.yml`), triggered by `v*` tags. Use `/release` for those.
