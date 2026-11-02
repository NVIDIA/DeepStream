<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Contributing to DeepStream

Thank you for your interest in improving DeepStream. This guide explains how to
propose a change, what a reviewable pull request looks like, and the licensing
terms that apply to contributions.

A contribution is ready for review when it is the smallest complete change that
solves the problem, includes the validation that proves it, and follows the pull
request template. Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). To report a security vulnerability, follow
[SECURITY.md](SECURITY.md); do not open a public issue or pull request for it.

## Choose a contribution path

| You want to... | Start with |
|---|---|
| Report a bug | Open an issue using the bug report template. |
| Suggest a feature or a larger change | Open an issue using the feature request template and wait for maintainer feedback **before** writing code. |
| Fix a typo or a small documentation issue | Open a pull request directly. |
| Fix a bug that already has an issue | Comment on the issue, then open a pull request that links it. |
| Ask a usage question | Use the [DeepStream SDK Developer Forum](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/15). |

New plugins, new reference applications, new third-party dependencies, and
changes to public headers in `includes/` or to the build system need maintainer
agreement in an issue before a pull request is opened. Changes to prebuilt
proprietary runtime assets are not accepted; those are published by NVIDIA as
GitHub Release assets.

## Development setup

1. Fork the repository, then clone your fork with submodules and Git LFS:

   ```bash
   git lfs install
   git clone --recurse-submodules https://github.com/<your-user>/DeepStream.git
   cd DeepStream
   git remote add upstream https://github.com/NVIDIA/DeepStream.git
   ```

2. Build. [`build/BUILD.md`](build/BUILD.md) is the authoritative build guide and
   lists per-platform prerequisites (x86 dGPU, Jetson, SBSA / DGX Spark).

   ```bash
   bash build/build.sh
   ```

   After the first full build, rebuild only the stage you changed:

   ```bash
   bash build/build.sh --only=<stage> --skip-deps --skip-artifacts
   ```

   Clear the GStreamer plugin cache after installing: `rm -rf ~/.cache/gstreamer-1.0/`.
   `gst-nvdsudp` and `gst-dsexample-cuda` are not built by `build/build.sh`; build
   them as described in their READMEs.

3. Run sample and reference applications **from their own source directory**.
   Their configuration files use paths relative to the config file.

4. Create a topic branch from the latest `develop`:

   ```bash
   git fetch upstream
   git checkout -b <topic-branch> upstream/develop
   ```

Each component's own `README` is authoritative for component-specific build and
run steps. `AGENTS.md` summarizes repository conventions for people and AI coding
agents alike.

## Making the change

- Define what success looks like before you start, and keep the change within
  that scope. Do not mix refactors, formatting churn, or unrelated fixes into a
  functional change.
- Read the component's `README` and match the surrounding code's naming,
  comment density, and idiom.
- C and C++: C++17, GStreamer naming conventions, `GST_ELEMENT_ERROR` / `GST_ERROR`
  for error paths instead of `printf`, and the `nvds_*` APIs for metadata access.
- Python: Python 3.10 or newer, `pyservicemaker` for metadata access, and
  `gi.repository.Gst` rather than shelling out to `gst-launch-1.0`.
- Do not hardcode absolute paths; use relative paths, `NVDS_VERSION`, or `CUDA_VER`.
- Leave no debug `printf` / `print` statements in committed code.
- Update documentation, sample configurations, and `CHANGELOG.md` when behavior,
  configuration, or public APIs change.
- Do not commit build artifacts: `.o`, `bin/`, `libs/`, `*.so.*`, `*.deb`,
  archives, `build/build.log`, or `build/.stage-state*`.

## Validate the change

Every pull request must meet the minimum build and sanity-test requirements in
[PULL_REQUEST_GUIDELINES.md](PULL_REQUEST_GUIDELINES.md).
Beyond that, run the narrowest checks that demonstrate the change works, then any
broader checks your change could affect:

- Build the affected stage and run the affected sample or reference application
  with the configuration it ships with.
- For plugin and library changes, include a short pipeline or command that
  exercises the new behavior.
- Test on every platform you can (x86 dGPU, Jetson, SBSA). State in the pull
  request which platforms you tested and which you could not.
- Where the component has tests (for example `tools/inference_builder/`), run them
  and add or update tests with your change.

Generated code, including code produced by AI tools, is a starting point. Review,
test, and security-validate it as you would any other code.

## Licensing of contributions

DeepStream source code is licensed under **Apache-2.0** and documentation under
**CC-BY-4.0**; see [LICENSE](LICENSE). By contributing, you agree that your
contribution is provided under those licenses.

- Every new or edited source file keeps the SPDX header, with the comment syntax
  of the language. For example, in Python or shell:

  ```text
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```

  Use `//` or `/* */` for C, C++, and CUDA, and `<!-- -->` with `CC-BY-4.0` for
  Markdown documentation.
- Only submit work you wrote or have the right to submit under these licenses.
  Do not copy code from sources with incompatible licenses.
- Do not vendor third-party code without prior agreement in an issue. If it is
  agreed, preserve its original license and notices and record it in
  `THIRD_PARTY_LICENSES.txt`.

## Signing Off Your Work

- We require that all contributors "sign-off" on their commits. This certifies
  that the contribution is your original work, or you have rights to submit it
  under the same license, or a compatible license.
  - Any contribution which contains commits that are not Signed-Off will not be
    accepted.
- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
  committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

  To add a missing sign-off, use `git commit --amend -s` for the latest commit,
  or `git rebase --signoff <base>` for several. Cryptographic commit signing
  (`-S`) is welcome but does not replace the sign-off.
- Full text of the DCO (<https://developercertificate.org/>):

  ```text
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

## Use of AI and LLM tools

AI-assisted development is welcome. You remain fully accountable for everything
you submit: you must understand it, have validated it, and be able to defend it in
review. Do not submit generated code you have not run, and do not include secrets,
internal hostnames, or proprietary data in prompts or in the change. AI coding
agents working in this repository should read [`AGENTS.md`](AGENTS.md) first.

## Submitting a pull request

1. Rebase on, or merge, the latest `develop` and resolve conflicts.
2. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
   messages and the pull request title, for example `fix(nvinfer): handle empty
   batch` or `docs(readme): clarify Jetson install`.
3. Sign off every commit (see above).
4. Open the pull request against `develop` and complete the pull request
   template: what changed and why, the linked issue, the platforms you tested, and
   how you validated it. Follow the step-by-step process and acceptance checklist
   in [PULL_REQUEST_GUIDELINES.md](PULL_REQUEST_GUIDELINES.md).
5. Review your own diff for secrets, build artifacts, and unrelated changes.
6. Keep the pull request as a draft until it is ready for review. Keep it small;
   split unrelated changes into separate pull requests.

Maintainers review when time allows and may ask for changes, request that a large
change be split, or decline a change that does not fit the product roadmap. A pull
request that has the required sign-off, template, and validation is much faster to
review. Continuous integration may need a maintainer to approve it before it runs
on a first-time contributor's pull request. Respond to review comments and CI
failures until the pull request is merged or closed.

## Internal changes by NVIDIA developers

NVIDIA developers making authorized internal changes must also follow the review,
validation, and approval requirements of the internal repository.

For changes explicitly prepared for the NVIDIA skill-publishing workflow, each
submission commit must carry the `Signed-off-by` trailer described above.
Unsigned skill-publication changes must be corrected before merge.
