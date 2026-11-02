<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# DeepStream Pull Request Submission Guidelines

Follow these steps when preparing a pull request (PR) for the `develop` branch of [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream). This guide complements [CONTRIBUTING.md](CONTRIBUTING.md), which covers contribution paths, coding conventions, licensing, and DCO sign-off.

## 1. Check out develop and create a working branch

Fork the repository to your GitHub account, then clone your fork. Replace `<your-github-username>` with your GitHub username.

```bash
git lfs install
git clone --recurse-submodules https://github.com/<your-github-username>/DeepStream.git
cd DeepStream
git remote add upstream https://github.com/NVIDIA/DeepStream.git
git fetch upstream
git switch -c feature/my-change upstream/develop
git submodule update --init --recursive
```

Use a descriptive working branch name. Keep each PR focused on one change or issue.

## 2. Make your changes

Implement the change and update any affected documentation, configuration files, or tests. Follow the coding conventions used in the surrounding code. Review your changes before building and submitting.

## 3. Build and run the minimum sanity test

Validate your changes on at least one supported platform: **x86 with an NVIDIA GPU or NVIDIA Jetson**. Set up the platform dependencies according to the repository's [build instructions](build/BUILD.md).

From the repository root, run the minimum required build command:

```bash
bash ./build/build.sh --package
```

The full build and package generation must complete successfully.

As the minimum sanity test, run the newly built `deepstream-app` with the **source4** sample configuration. It is installed with the `deepstream-sample-data` package under `/opt/nvidia/deepstream/deepstream-<version>/samples/configs/deepstream-app/`. Run it from that directory because the configuration uses relative paths:

```bash
cd /opt/nvidia/deepstream/deepstream-<version>/samples/configs/deepstream-app/
deepstream-app -c source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display.txt
```

Ensure the application loads the binaries and libraries produced by your build, not a different installed copy, and that the required models and sample streams are available.

Confirm that all four sources initialize and process frames, inference and configured output operate as expected, and no crashes or unexpected errors occur. Run additional tests for the components affected by your change. Resolve failures before submitting.

Record the platform, DeepStream version, tested commit, build result, source4 configuration, test duration, and test result for inclusion in the PR.

## 4. Commit and sign off your changes

After validation passes, stage the intended files and create a commit with a clear message and contributor sign-off:

```bash
git add <changed-files>
git commit -s -m "fix(nvinfer): handle empty batch"
```

Replace `<changed-files>` with the files you intend to submit. Write the commit message as a [Conventional Commit](https://www.conventionalcommits.org/), for example `fix(nvinfer): handle empty batch` or `docs(readme): clarify Jetson install`. The `-s` option adds a `Signed-off-by` trailer using your configured Git name and email.

By submitting the signed-off commit and completing the PR checklist below, you confirm that you have performed the required build and testing. Sign-off is a contributor declaration; maintainers determine whether the PR meets the acceptance criteria.

## 5. Rebase and push your working branch

Rebase on the latest `develop` and resolve any conflicts before pushing. If the rebase changes your code, rebuild and retest (Step 3).

```bash
git fetch upstream
git rebase upstream/develop
git push -u origin feature/my-change
```

## 6. Open a pull request targeting develop

On GitHub, create a PR with the following settings:

| Setting | Value |
| --- | --- |
| Base repository | `NVIDIA/DeepStream` |
| Base branch | `develop` |
| Head repository | Your fork |
| Compare branch | `feature/my-change` |

Include a clear summary, the reason for the change, related issue links where applicable, and the validation details recorded in Step 3. Disclose known limitations and any affected platforms that were not tested.

Copy this acceptance checklist into the PR description:

- [ ] My working branch is based on `NVIDIA/DeepStream:develop`.
- [ ] The PR targets `NVIDIA/DeepStream:develop`.
- [ ] `bash ./build/build.sh --package` completed successfully.
- [ ] The source4 sanity test passed with the newly built `deepstream-app` on x86 or Jetson.
- [ ] The PR records the tested platform, software version, commit, source4 configuration, test duration, and results.
- [ ] Additional tests relevant to the change passed, where applicable.
- [ ] My submitted commits include contributor sign-off.
- [ ] Affected documentation is updated, where applicable.

Address reviewer feedback and resolve any required check failures before merge. If code changes after validation, rebuild and rerun the relevant tests, including the minimum build and sanity test, then update the PR results to identify the latest tested commit. Final acceptance and merge are subject to maintainer review and the repository's configured requirements.
