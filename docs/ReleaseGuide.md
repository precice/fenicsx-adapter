# Guide to release new version of the FEniCSx-preCICE adapter

Before starting this process make sure to check that all relevant changes are included in the `CHANGELOG.md`. The developer who is releasing a new version of FEniCSx-preCICE adapter is expected to follow this workflow:

1. If it does not already exist, create a release branch with the version number of the planned release. Use develop as base for the branch. `git checkout develop`; `git checkout -b fenicsx-adapter-vX.X.X`. Perform the following steps only on the release branch, if not indicated differently.  

2. [Open a Pull Request from the branch `fenicsx-adapter-vX.X.X` to `main`](https://github.com/precice/fenicsx-adapter/compare) named after the version (i.e. `Release v1.0.0`) and briefly describe the new features of the release in the PR description.

3. Bump the version in the following places:

    a) Before merging the PR, make sure to bump the version in `CHANGELOG.md` on `fenicsx-adapter-vX.X.X`  
    b) There is no need to bump the version anywhere else, since we use the [python-versioneer](https://github.com/python-versioneer/python-versioneer/) for maintaining the version everywhere else.

4. [Draft a New Release](https://github.com/precice/fenicsx-adapter/releases/new) in the `Releases` section of the repository page in a web browser. The release tag needs to be the exact version number (i.e.`v1.0.0` or `v1.0.0rc1`, compare to [existing tags](https://github.com/precice/fenicsx-adapter/tags)). Use `@target:main`. Release title is also the version number (i.e. `v1.0.0` or `v1.0.0rc1`, compare to [existing releases](https://github.com/precice/fenicsx-adapter/tags)).
*Note:* If it is a pre-release then the option *This is a pre-release* needs to be selected at the bottom of the page. Use `@target:fenicsx-adapter-vX.X.X` for a pre-release, since we will never merge a pre-release into main.

    a) If a pre-release is made: Directly hit the "Publish release" button in your Release Draft. Now you can check the artifacts (e.g. release on PyPI (**TODO:** Link to `https://pypi.org/project/fenicsxprecice/#history` as soon as release exists.) of the release. *Note:* As soon as a new tag is created github actions will take care of deploying the new version on PyPI using [this workflow](https://github.com/precice/fenicsx-adapter/actions?query=workflow%3A%22Upload+Python+Package%22).

    b) If this is a "real" release: As soon as one approving review is made, merge the release PR (`fenicsx-adapter-vX.X.X`) into `main`.

5. Merge `main` into `develop` for synchronization of `develop`.

6. If everything is in order up to this point then the new version can be released by hitting the "Publish release" button in your Release Draft.
