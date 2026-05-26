# Releasing the ENNOVA teqp fork

This fork extends upstream teqp with two ENNOVA-specific additions:

- **Jog-Chapman polar term** (alternative to the Gross-Vrabec polar kernel that ships upstream)
- **C++ density solver** (`solve_density_from_guess` and `get_bmix`)

We do **not** publish to PyPI — upstream owns the `teqp` name there. Instead, prebuilt wheels are attached to GitHub Releases on this fork and consumed by downstream packages (notably `fpFlash`) via direct release-asset URLs.

## Versioning

The wheel version uses a PEP 440 local-version identifier:

```
<upstream-version>+ennova.<iteration>
```

Examples:

- `0.23.1+ennova.1` — first ENNOVA release based on upstream teqp 0.23.1
- `0.23.1+ennova.2` — subsequent iteration on the same upstream base
- `0.24.0+ennova.1` — first iteration after rebasing onto upstream 0.24.0

The `+ennova.N` segment guarantees the wheel is distinct from anything upstream might publish at the same version.

Tags use `-` instead of `+` for URL safety:

```
v0.23.1-ennova.1   (git tag)
0.23.1+ennova.1    (version in pyproject.toml and the resulting wheel filename)
```

## Release process

1. **Make your changes** on the relevant branch (typically `jog-chapman-polar` for the JC kernel work, or `main` once changes have been integrated).

2. **Bump the version** in `pyproject.toml`:

   ```toml
   [project]
   version = "0.23.1+ennova.2"
   ```

3. **Commit the version bump**:

   ```bash
   git add pyproject.toml
   git commit -m "release: bump version to 0.23.1+ennova.2"
   ```

4. **Tag and push**:

   ```bash
   git tag v0.23.1-ennova.2
   git push
   git push --tags
   ```

5. **Wait for CI** (~30 minutes). The `cibuildwheel` workflow builds wheels for Linux, Windows, and macOS across Python 3.9-3.13, then attaches them to a GitHub Release created automatically from the tag.

6. **Verify the release** at `https://github.com/ENNOVA-LLC/teqp/releases/tag/v0.23.1-ennova.2`. You should see ~20 wheel files (one per Python × OS combination), plus a source tarball.

7. **Update consumer packages** to point at the new release. For fpFlash, update the `teqp` dependency URLs in `pyproject.toml` to reference the new tag and wheel filenames, then regenerate the lockfile:

   ```bash
   poetry lock --no-update
   poetry install
   ```

## What the CI workflow does

`.github/workflows/build_cibuildwheel.yml` runs on every push and PR (verifies that wheels still build) and additionally attaches wheels to a GitHub Release when a `v*` tag is pushed.

It does **not** publish to PyPI or TestPyPI — the publish jobs from upstream were removed because we don't own the `teqp` name on PyPI.

## Rebasing onto a new upstream version

When upstream teqp releases a new version that you want to incorporate:

1. Fetch upstream:

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main      # or rebase, depending on your preference
   ```

2. Resolve any merge conflicts in the JC polar kernel or density-solver code.

3. Bump the version in `pyproject.toml` to reflect the new upstream base:

   ```toml
   version = "0.24.0+ennova.1"
   ```

4. Follow the release process above.

## Build prerequisites (only needed if building locally)

If you build wheels locally (for debugging, not for distribution):

- CMake ≥ 3.18
- C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)
- Python ≥ 3.9
- Submodules initialized: `git submodule update --init --recursive`

Then:

```bash
pip install build
python -m build --wheel
```

The wheel lands in `dist/`.

## Troubleshooting

**CI fails to attach wheels to the release.** Check that the tag matches `v*` and the `attach_to_release` job has `contents: write` permission (it should — verify in the workflow file).

**Wheel filename doesn't match what consumers expect.** The cibuildwheel filename format is deterministic: `teqp-<version>-cp<py>-cp<py>-<platform_tag>.whl`. For example, `teqp-0.23.1+ennova.1-cp312-cp312-manylinux_2_28_x86_64.whl`. If a consumer's URL is wrong, fix the consumer's `pyproject.toml`.

**`poetry install` in a consumer says "Could not find a matching version."** Ensure the consumer's `pyproject.toml` references the exact wheel filename produced by CI. The `+` character in the version is preserved literally in the filename.
