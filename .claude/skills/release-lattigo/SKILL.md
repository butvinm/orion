# Release Lattigo Bindings

Release a lattigo binding package independently.

**Arguments:** `$ARGUMENTS` — format: `<target> <version>` where target is `py` or `js`. Example: `py 6.2.3` or `js 6.2.1`. Required.

## Versioning

Lattigo binding versions use 3-part semver matching upstream Lattigo's major.minor, with patch for our changes:

- `6.2.0` — initial bindings for upstream Lattigo 6.2.0
- `6.2.1` — our patch to the bindings (Python or JS independently)

Python and JS versions are **independent** — they don't need to match.

## Prerequisites

Before starting, verify:

1. Working tree is clean (`git status` shows no uncommitted changes)
2. On `main` branch
3. CI is passing on the latest commit
4. The version tag does not already exist:
   - For `py`: `git tag -l "lattigo-py-v<version>"`
   - For `js`: `git tag -l "lattigo-js-v<version>"`

If any check fails, stop and tell the user.

## Version file to bump

- For `py`: `python/lattigo/pyproject.toml`
- For `js`: `js/lattigo/package.json`

## Steps

1. Run all prerequisite checks
2. If version needs bumping, update the file and commit: `chore: bump lattigo <target> bindings to <version>`
   - Stage only the 1 specific file, never `git add .`
3. Push to main: `git push origin main`
4. Create and push tag:
   - For `py`: `git tag lattigo-py-v<version> && git push origin lattigo-py-v<version>`
   - For `js`: `git tag lattigo-js-v<version> && git push origin lattigo-js-v<version>`
5. Monitor the release workflow:
   - For `py`: `gh run list --workflow release-lattigo-python.yml --limit 1`
   - For `js`: `gh run list --workflow release-lattigo-js.yml --limit 1`
6. Print the GitHub Release URL when done

## What this does NOT publish

- `orion-compiler` and `orion-evaluator` use `/release` (triggered by `v*` tags).
