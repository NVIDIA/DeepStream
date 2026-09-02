#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate the DeepStream Code Agent Brev launchable notebook.

SOURCE OF TRUTH for deepstream_code_agent_launchable.ipynb. The heavy engine (docker
orchestration, the prompt catalog, the agent install script, result rendering) lives in
deploy/brev/scripts/ds_agent_lab.py. The notebook shows the **trunk steps** -- readable
main code -- without the low-level detail:

  * Step 1 (Configuration) is plain Python and Step 2 (Check environment & create workspace) is a `%%bash` cell.
  * Step 3 (Install agent + skills + deps) = agent dropdown + Install.
  * Step 4 (Authenticate) = method dropdown (account sign-in / API key / endpoint) + its fields.
  * Step 5 (Generate) = prompt dropdown + a linked, editable textarea + Generate.
  * Step 6 (Run & view results) = a couple of named lab.* trunk calls.

    python3 deploy/brev/scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


MD_INTRO = """\
# DeepStream Code Agent

This notebook lets you try DeepStream Code Agent in a Brev launchable environment: start from a
natural-language scenario prompt, generate a DeepStream project or workflow, run it in the
preconfigured DeepStream container, and inspect verifiable results.

Generated results are not limited to `pyservicemaker` applications. Depending on the selected
prompt, the agent may create video analytics applications, model import and conversion artifacts,
automated performance analysis reports, message broker adapter native libraries, service wrappers,
OpenAPI specs, or other scenario-specific files.

The notebook uses two different operations:

- **Generate** asks the agent to write the project and supporting files for the selected scenario.
- **Run and view results** prepares the required services, runs the generated project, and shows
  the result. Runtime repairs are made in a separate copy, so the original generated project stays
  available for comparison.

### Flow

```text
Configuration -> Environment -> Install agent -> Authenticate
              -> Generate -> Run and view results -> optional Cleanup
```

| Stage | What you do | What the agent/skill does |
| --- | --- | --- |
| 1. Configuration | Confirm paths, image, and runtime defaults | Shares one configuration with every later step |
| 2. Environment | Check the GPU host and create the workspace | Starts an isolated DeepStream container |
| 3. Install agent | Choose Claude Code or Codex | Installs the CLI, DeepStream skills, and scenario dependencies |
| 4. Authenticate | Complete one sign-in method | Verifies access to the selected model provider |
| 5. Generate | Choose or edit a scenario prompt | Writes the project and supporting files |
| 6. Run and view results | Run the selected project | Starts services, executes the project, and renders evidence |
| 7. Cleanup _(optional)_ | Remove lab containers | Releases runtime resources while preserving outputs |

### Before you start

Use a Jupyter host with an NVIDIA GPU, Docker GPU support, and access to the configured DeepStream
image. Keep `deploy/`, `example_prompts/`, and `skills/` together under the Jupyter root directory.
You will need credentials for the agent you select. The first run can take several minutes while
images, packages, models, and TensorRT engines are downloaded or built.

Run the numbered cells from top to bottom. In each step, wait for the green success message before
moving on. If a step fails, fix the first red error and rerun that step; do not skip ahead.

> **Security model:** credentials are accepted at runtime, held in memory, and injected only for
> the relevant agent invocation. They are never baked into the DeepStream image or printed by
> the notebook. Generated projects and artifacts are written to the configured output workspace.

> **If you refresh the page:** the browser may clear live cell output, but the kernel and generated
> files remain. Enable **Settings → Save Widget State Automatically** if you want widget state to
> persist. You can redisplay saved output with `lab.show_generated_code()` or `lab.show_results()`.
"""

MD_CONFIG = """\
## 1. Configuration

This is the only cell you normally edit before starting. It sets the output location, DeepStream
image, container name, sample video, optional service endpoints, and time limits used by later
steps.

Before running the cell:

1. Confirm that `OUTPUT_ROOT` is under the Jupyter root so generated files appear in the file
   browser.
2. Confirm the DeepStream image and sample video paths.
3. Change optional Kafka, NATS, or VLM values only if your scenario needs them.
4. Adjust `AGENT_TIMEOUT` if a large model import or profiling run needs more time.

Optional authentication environment variables are listed as commented templates in the
configuration cell. Uncomment and replace them only for a local/private launch; otherwise use
the agent-specific controls in Step 4. Never commit real credentials to the notebook.

Run the cell after any edits. If you change a setting after loading the agent, run **Install**
again so the later steps use the new configuration.

**Checkpoint:** the cell ends with `Configuration set` and prints the active values. If a path or
image name is wrong, correct it before continuing to Step 2.
"""

# Plain command-line cell (no `lab` yet) -- colored with the SAME ANSI scheme as klog
# (cyan headers, green OK) so Step 1 reads consistently with the widget steps below.
CODE_CONFIG = '''\
import os, pathlib

_C = {"head": "\\033[1;36m", "ok": "\\033[1;32m", "key": "\\033[0;36m", "off": "\\033[0m"}

_repo = next((p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "deploy/brev/scripts/ds_agent_lab.py").exists()), pathlib.Path.cwd())

# All values below are assigned UNCONDITIONALLY (`os.environ[k] = v`, not `setdefault`): EDIT any of
# them and re-run this cell to change it. `setdefault` would be a no-op on re-run, because os.environ
# persists in the kernel and setdefault only writes when the key is absent. NOTE: for everything
# except AGENT_TIMEOUT you must ALSO re-run the Install cell afterwards -- the lab module snapshots
# these at import time; AGENT_TIMEOUT alone is re-read on the next Generate, so it takes effect at once.
os.environ["REPO_ROOT"] = str(_repo)
# Outputs live in a dir NEXT TO this notebook (deploy/brev/outputs) -- under the Jupyter root_dir, so
# they show in the left file browser. Edit here to relocate.
os.environ["OUTPUT_ROOT"] = str(pathlib.Path(os.environ["REPO_ROOT"]) / "deploy" / "brev" / "outputs")
os.environ["WORKSPACE"] = str(pathlib.Path(os.environ["OUTPUT_ROOT"]) / "workspace")
os.environ["DEEPSTREAM_IMAGE"] = "nvcr.io/nvidia/deepstream:9.1-triton-multiarch"
os.environ["AGENT_CONTAINER"] = "ds-agent-work"
os.environ["AGENT_HOME"] = "/home/agent"
# Optional provider credentials for advanced/headless use. Keep these commented in shared copies;
# Step 4 is the normal interactive path. Never commit real credentials here.
# Authentication variable mapping:
#   Claude Code | API key          | ANTHROPIC_API_KEY
#   Claude Code | Custom endpoint  | ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN + ANTHROPIC_MODEL
#   Codex       | API key          | OPENAI_API_KEY
#   Codex       | Custom endpoint  | OPENAI_BASE_URL + OPENAI_API_KEY (token) + OPENAI_MODEL
# os.environ["ANTHROPIC_API_KEY"] = "..."
# os.environ["ANTHROPIC_BASE_URL"] = "https://your-anthropic-proxy/v1"
# os.environ["ANTHROPIC_AUTH_TOKEN"] = "..."
# os.environ["OPENAI_API_KEY"] = "..."  # also the token for an OpenAI-compatible endpoint
# os.environ["OPENAI_BASE_URL"] = "https://your-openai-proxy/v1"
# os.environ["OPENAI_MODEL"] = "model-exposed-by-your-endpoint"
# Agent model for the 'claude' agent. Default = Claude Opus 4.8 (best code quality). Set to
# "claude-sonnet-4-6" for cheaper/faster, or "" to let Claude Code pick its own default.
# (Requires your account/key to have access to the chosen model. Codex uses its own default.)
os.environ["ANTHROPIC_MODEL"] = "claude-opus-4-8"
os.environ["KAFKA_TOPIC"] = "deepstream-agent-demo"
os.environ["KAFKA_BOOTSTRAP"] = "127.0.0.1:9092"
os.environ["NATS_URL"] = "nats://127.0.0.1:4222"
os.environ["NATS_SUBJECT"] = "deepstream.detections"
os.environ["VLM_ENDPOINT"] = "http://127.0.0.1:8000/v1"
os.environ["VLM_MODEL"] = "Qwen/Qwen2.5-VL-3B-Instruct"
os.environ["SAMPLE_VIDEO"] = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"
os.environ["MIN_DRIVER_VERSION"] = "550.0.0"
# Overall BACKSTOP (seconds) on one Generate run -- how long the agent may write + debug before it is
# killed. 3000 = 50 min (heavy prompts like profiling can legitimately need this). Does NOT affect
# demo runtime or MP4 length: live/RTSP runs are bounded separately at the Run step.
os.environ["AGENT_TIMEOUT"] = "3000"

print(f"{_C['head']}=== Global configuration ==={_C['off']}")
for key in [
    "REPO_ROOT", "OUTPUT_ROOT", "WORKSPACE", "DEEPSTREAM_IMAGE",
    "AGENT_CONTAINER", "AGENT_HOME", "ANTHROPIC_MODEL", "AGENT_TIMEOUT", "MIN_DRIVER_VERSION",
]:
    print(f"  {_C['key']}{key}{_C['off']}={os.environ[key]}")
print(f"{_C['ok']}\\u2705 Configuration set -- continue to Step 2 (Check Environment & Create a DeepStream container workspace).{_C['off']}")
'''

MD_PREPARE = """\
## 2. Check Environment & Create a DeepStream container workspace

This step prepares the place where the agent will work. It checks the GPU and Docker, makes sure
the DeepStream image can run with GPU support, creates the persistent output workspace, and starts
a clean container with the sample video and scenario settings available.

Run this cell after Step 1 and wait for the image pull to finish. Running it again is safe: it
recreates the container but keeps files already written under the host-side workspace. This is the
normal recovery action if the container becomes stale.

**Checkpoint:** continue only when the output says `Environment ready` and shows the container name
and mounted workspace. If it stops earlier, fix the first red error before continuing. Common
causes are an unavailable Docker daemon, missing GPU support, registry access, or an old driver.
"""

# Command-line (bash) step. Colored with the SAME ANSI scheme as klog -- section()/ok()/fail()
# emit cyan headers, green success, red errors, so this step reads consistently with the
# widget steps (which use step_status's HTML banner). `%%bash` must stay the first line.
CODE_PREPARE = '''\
%%bash
set -euo pipefail

C_HEAD='\\033[1;36m'; C_OK='\\033[1;32m'; C_ERR='\\033[1;31m'; C_OFF='\\033[0m'
section() { printf "${C_HEAD}=== %s ===${C_OFF}\\n" "$1"; }
ok()      { printf "${C_OK}%s${C_OFF}\\n" "$1"; }
fail()    { printf "${C_ERR}ERROR: %s${C_OFF}\\n" "$*" >&2; exit 1; }

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "$1 is not installed or not on PATH"
}

version_ge() {
    [ "$(printf '%s\\n%s\\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

section "NVIDIA Driver & GPU"
require_command nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
[ "$GPU_COUNT" -gt 0 ] || fail "No NVIDIA GPUs detected"
ok "Detected $GPU_COUNT GPU(s)"

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')
version_ge "$DRIVER_VERSION" "$MIN_DRIVER_VERSION" \\
    || fail "NVIDIA driver $DRIVER_VERSION is older than required $MIN_DRIVER_VERSION"
ok "NVIDIA driver: OK ($DRIVER_VERSION >= $MIN_DRIVER_VERSION)"
echo ""

section "Docker"
require_command docker
docker ps >/dev/null 2>&1 \\
    || fail "Docker daemon is not reachable by this user. Add the user to the docker group or start Docker."
ok "Docker daemon: OK"
echo ""

section "Disk Space"
df -h / | tail -1 | awk '{print "Root:", $4, "available of", $2}'
ok "Prerequisites check passed."
echo ""

section "Prepare DeepStream sandbox"

for required_var in \\
    REPO_ROOT OUTPUT_ROOT WORKSPACE DEEPSTREAM_IMAGE AGENT_CONTAINER AGENT_HOME \\
    KAFKA_TOPIC KAFKA_BOOTSTRAP NATS_URL NATS_SUBJECT VLM_ENDPOINT VLM_MODEL \\
    SAMPLE_VIDEO MIN_DRIVER_VERSION; do
    [ -n "${!required_var:-}" ] || fail "$required_var is not set. Run the Step 1 configuration cell first."
done

python3 "$REPO_ROOT/deploy/brev/scripts/configure_docker_nvcr_workaround.py"

if ! docker image inspect "$DEEPSTREAM_IMAGE" >/dev/null 2>&1; then
    if [[ "$DEEPSTREAM_IMAGE" == nvcr.io* ]] && [ -n "${NGC_API_KEY:-}" ]; then
        printf '%s' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
    fi
    echo "Pulling $DEEPSTREAM_IMAGE ..."
    docker pull "$DEEPSTREAM_IMAGE"
else
    ok "DeepStream image present: $DEEPSTREAM_IMAGE"
fi

# Validate the NVIDIA Container Toolkit only after the image is present, so a missing
# image or NVCR credential failure is not misreported as a toolkit failure.
section "NVIDIA Container Toolkit"
docker run --rm --gpus all "$DEEPSTREAM_IMAGE" nvidia-smi >/dev/null 2>&1 \\
    || fail "NVIDIA Container Toolkit is not functional. Install or repair it: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
ok "NVIDIA Container Toolkit: OK"

mkdir -p "$WORKSPACE/agent_outputs"
rm -rf "$WORKSPACE/example_prompts"
cp -a "$REPO_ROOT/example_prompts" "$WORKSPACE/example_prompts"

if docker ps -aq -f "name=^${AGENT_CONTAINER}$" | grep -q .; then
    docker rm -f "$AGENT_CONTAINER" >/dev/null
fi

docker run -d \\
    --name "$AGENT_CONTAINER" \\
    --gpus all \\
    --network host \\
    -w /workspace \\
    -e "HOME=$AGENT_HOME" \\
    -e "KAFKA_TOPIC=$KAFKA_TOPIC" \\
    -e "KAFKA_BOOTSTRAP=$KAFKA_BOOTSTRAP" \\
    -e "NATS_URL=$NATS_URL" \\
    -e "NATS_SUBJECT=$NATS_SUBJECT" \\
    -e "VLM_ENDPOINT=$VLM_ENDPOINT" \\
    -e "VLM_MODEL=$VLM_MODEL" \\
    -v "$WORKSPACE:/workspace" \\
    "$DEEPSTREAM_IMAGE" \\
    sleep infinity

docker exec "$AGENT_CONTAINER" bash -lc "true"

if [ ! -f "$WORKSPACE/sample_720p.mp4" ]; then
    docker exec "$AGENT_CONTAINER" cp "$SAMPLE_VIDEO" /workspace/sample_720p.mp4 \\
        || echo "NOTE: could not stage $SAMPLE_VIDEO into the workspace; continuing."
fi

echo ""
ok "✅ Environment ready. Container '$AGENT_CONTAINER' is running."
ok "Workspace mounted at: $WORKSPACE"
'''

MD_STEP2 = """\
## 3. Install the agent, DeepStream skills, and runtime dependencies

Choose **Claude Code** or **Codex**, then click **Install**.

This step installs the selected agent CLI, DeepStream skills, `pyservicemaker`, and scenario
runtime dependencies inside the container created in Step 2. The host machine is not changed.

The first installation may take several minutes. When it completes, continue to Step 4 to
authenticate. If `ds-agent-work` already exists from an earlier run, this Install step reuses
that container directly; you do not need to rerun Step 1 or Step 2 just to authenticate again.

**Checkpoint:** wait for the green completion message. Re-run **Install** if you recreated the
container, changed the agent, or changed any configuration value in Step 1.
"""

CODE_STEP2 = '''\
try:
    import ipywidgets as w
    from IPython.display import display
    from html import escape as _html_escape
except ImportError:
    w = None

if w is not None:
    agent_dd = w.Dropdown(options=["claude", "codex"], value="claude", description="Agent:")
    install_btn = w.Button(description="Install", button_style="primary", icon="download")
    install_out = w.Output(layout=w.Layout(border="1px solid #ccc"))

    def _on_install(_):
        previous_agent = getattr(lab, "AGENT", "").strip().lower()
        next_agent = agent_dd.value
        if previous_agent != next_agent:
            lab.cancel_login()
        lab.AGENT = next_agent                               # read the dropdown on the UI thread
        lab.set_selection(lab.AGENT, None)
        lab.run_step(install_out, install_btn, "Install agent", lab.install_agent,
                              success_flag="installed")

    install_btn.on_click(_on_install)
    print("Pick an agent and click Install (CLI + DeepStream skills + deps). This may take a few minutes.\\n"
          "If ds-agent-work already exists, the current container is reused directly.")
    display(w.VBox([w.HBox([agent_dd, install_btn]), install_out]))
else:
    print("ipywidgets unavailable -- plain fallback:")
    print("  lab.AGENT = 'claude'              # or 'codex'")
    print("  lab.set_selection(lab.AGENT, None); lab.install_agent()")
'''

MD_AUTH = """\
## 4. Authenticate

Authenticate the agent you installed in Step 3. The controls are agent-specific: if you change the
agent in Step 3, rerun this cell so the notebook rebuilds the matching form.

### Claude Code authentication methods

| Method | Use it when | What to do |
| --- | --- | --- |
| **Account sign-in** | You use a Claude subscription or Anthropic Console account | Choose the account type, click **Start sign-in**, finish the browser login, paste the returned code, and click **Submit** |
| **API key** | You have an Anthropic key for Claude | Paste the key and click **Apply & verify** |
| **Custom endpoint** | Your organization provides an Anthropic-compatible gateway | Enter the base URL, token, and model, then click **Apply & verify** |

For Claude account sign-in, choose **Claude subscription** for Pro, Max, or Team accounts. Choose
**Anthropic Console / API billing** for API-billed accounts or when subscription sign-in is blocked.

> The Claude account sign-in flow is unchanged: it uses `claude auth login` and a paste-code
> submission.

### Codex authentication methods

Codex uses its own account, API-key, and OpenAI-compatible endpoint flows; Claude account options
and the Claude paste-code submit button are not used.

| Method | Use it when | What to do |
| --- | --- | --- |
| **ChatGPT account sign-in** | You want Codex usage through a ChatGPT plan | Click **Start device sign-in**, complete the device flow in the browser, then click **Check sign-in** |
| **OpenAI API key** | You want usage-based OpenAI API billing | Paste the key and click **Apply & verify** |
| **OpenAI-compatible endpoint** | Your organization provides a Codex-compatible gateway | Enter the base URL, token, and model, then click **Apply & verify** |

The Codex custom endpoint must implement the OpenAI **Responses API**. The current Codex CLI no
longer accepts a Chat Completions-only provider; the notebook configures the endpoint as an
ephemeral Codex model provider and performs a real prompt smoke test.

**Checkpoint:** continue only after the green verification message appears. If Generate later
reports an authentication error, return here and apply the credential again. Restarting the kernel
clears credentials held only in memory.
"""

CODE_AUTH_BODY = '''\
try:
    import ipywidgets as w
    from IPython.display import display
    from html import escape as _html_escape
except ImportError:
    w = None

agent = getattr(lab, "AGENT", "").strip().lower()

def _ensure_step3_agent(expected):
    if globals().get("lab") is not lab:
        raise RuntimeError(
            "These authentication controls are stale after an agent change; rerun Step 4."
        )
    actual = getattr(lab, "AGENT", "").strip().lower()
    if actual != expected:
        raise RuntimeError(
            "Step 3 agent changed; rerun Step 4 to rebuild authentication controls."
        )


# Codex has a different login flow and provider configuration, so it gets its
# own controls. The Claude branch below intentionally remains the original
# Claude Code form and callback flow.
if w is not None and agent == "codex":
    import time

    method = w.Dropdown(
        options=[("ChatGPT account sign-in", "account"),
                 ("OpenAI API key", "key"),
                 ("OpenAI-compatible endpoint", "endpoint")],
        value="account", description="Method:",
    )

    # -- ChatGPT device sign-in --
    account_note = w.HTML(
        "<i>Codex uses ChatGPT device sign-in. Complete the browser/device flow, "
        "copy the device number below, open the link, then click Check sign-in; "
        "there is no Claude paste-code field.</i>"
    )
    device_link = w.HTML(
        "<i>The device sign-in link will appear after Start device sign-in.</i>"
    )
    device_code = w.HTML(
        "<i>The device number will appear here. Select and copy it before verification.</i>"
    )
    device_status = w.HTML()
    start_btn = w.Button(description="Start device sign-in", button_style="primary", icon="sign-in")
    check_btn = w.Button(description="Check sign-in", button_style="success", icon="check")
    account_box = w.VBox([
        account_note, device_link, device_code, device_status, w.HBox([start_btn, check_btn])
    ])

    # -- OpenAI API key --
    key_in = w.Password(
        description="API key:", layout=w.Layout(width="55%"),
        placeholder="OpenAI API key",
    )
    key_apply = w.Button(description="Apply & verify", button_style="info", icon="key")
    key_box = w.VBox([key_in, key_apply])

    # -- OpenAI-compatible third-party endpoint --
    endpoint_label = "OpenAI Responses API endpoint"
    base_in = w.Text(description="Base URL:", layout=w.Layout(width="55%"),
                     placeholder="https://your-openai-proxy/v1")
    tok_in = w.Password(description="Token:", layout=w.Layout(width="55%"),
                        placeholder="token for the custom endpoint")
    model_in = w.Text(
        description="Model:", layout=w.Layout(width="55%"),
        value=getattr(lab, "OPENAI_MODEL", ""),
        placeholder="model exposed by the endpoint",
    )
    ep_apply = w.Button(description="Apply & verify", button_style="info", icon="key")
    endpoint_box = w.VBox([w.HTML(
                               f"<i>{endpoint_label}; it must implement the Responses API. "
                               "Verification sends a real 'Say OK' prompt.</i>"),
                           base_in, tok_in, model_in, ep_apply])

    out2 = w.Output(layout=w.Layout(border="1px solid #ccc"))
    _boxes = {"account": account_box, "key": key_box, "endpoint": endpoint_box}
    _auth_state = {"method": method.value}

    def _show(which):
        for _k, _b in _boxes.items():
            _b.layout.display = "" if _k == which else "none"

    def _on_method(change):
        if change["name"] == "value":
            _auth_state["method"] = change["new"]
            lab.cancel_login()
            lab.clear_codex_credentials()
            lab.CODEX_DEVICE_URL = ""
            lab.CODEX_DEVICE_CODE = ""
            out2.clear_output()
            device_link.value = (
                "<i>The device sign-in link will appear after Start device sign-in.</i>"
            )
            device_code.value = (
                "<i>The device number will appear here. Select and copy it before verification.</i>"
            )
            device_status.value = ""
            _show(change["new"])

    method.observe(_on_method, names="value")
    _show(method.value)

    def _ensure_codex_method(expected):
        _ensure_step3_agent("codex")
        if _auth_state["method"] != expected:
            raise RuntimeError("Select the matching Codex authentication method first.")

    def _render_device_info():
        url = getattr(lab, "CODEX_DEVICE_URL", "")
        code = getattr(lab, "CODEX_DEVICE_CODE", "")
        if code:
            device_code.value = (
                "<span style='display:inline-block;margin:6px 0;font-size:14px'>"
                "<b>Device number:</b> "
                f"<code style='font-size:18px;padding:4px 8px;background:#f6f8fa;"
                f"border:1px solid #d0d7de;border-radius:5px;user-select:all'>"
                f"{_html_escape(code)}</code> "
                "<span style='color:#57606a'>(select and copy)</span></span>"
            )
        else:
            device_code.value = (
                "<i>The device number was not captured; see the sign-in output below.</i>"
            )
        if url:
            safe_url = _html_escape(url, quote=True)
            device_link.value = (
                "<a style='display:inline-block;padding:6px 10px;background:#0969da;"
                "color:white;border-radius:6px;font-weight:600;text-decoration:none' "
                f'href="{safe_url}" target="_blank" rel="noopener noreferrer">'
                "Open ChatGPT device sign-in ↗</a>"
                f"<div style='margin-top:4px;word-break:break-all;color:#57606a'>"
                f"{_html_escape(url)}</div>"
            )
        else:
            device_link.value = "<i>Could not capture the device link; see the sign-in output below.</i>"

    if getattr(lab, "CODEX_DEVICE_URL", "") or getattr(lab, "CODEX_DEVICE_CODE", ""):
        _render_device_info()
        device_status.value = (
            "<span style='color:#0969da'><b>Device sign-in is ready.</b> "
            "Enter the device number, complete browser verification, then click "
            "<b>Check sign-in</b>.</span>"
        )

    def _start_codex_login():
        device_link.value = "<i>Starting device sign-in...</i>"
        device_status.value = ""
        result = lab.login_start()
        _render_device_info()
        device_status.value = (
            "<span style='color:#0969da'><b>Device sign-in started.</b> "
            "Enter the device number, complete browser verification, then click "
            "<b>Check sign-in</b>.</span>"
        )
        return result

    def _on_start(_):
        _ensure_codex_method("account")
        lab.run_step(
            out2,
            start_btn,
            "Start Codex sign-in",
            _start_codex_login,
        )

    def _on_check(_):
        _ensure_codex_method("account")
        out2.clear_output()
        if hasattr(out2, "outputs"):
            out2.outputs = ()
        locked = (method, start_btn, check_btn)
        previous_disabled = [(control, control.disabled) for control in locked]
        for control in locked:
            control.disabled = True
        check_btn.description = "Checking sign-in..."
        device_status.value = (
            "<span style='color:#0969da'><b>Checking device sign-in.</b> "
            "Finish browser verification; the result will appear here automatically.</span>"
        )
        try:
            # Keep Codex device verification deliberately simple: after the browser
            # flow, give the CLI three seconds to return, then reuse run_step. Do not
            # replace this with a background poller or change the Claude flow here.
            time.sleep(3)
            verified = lab.run_step(
                out2,
                check_btn,
                "Verify Codex sign-in",
                lambda: lab.check_auth("codex"),
            )
            device_status.value = (
                "<span style='color:#1a7f37'><b>Device sign-in verified.</b> "
                "Codex is ready to use.</span>"
                if verified else
                "<span style='color:#cf222e'><b>Device sign-in failed.</b> "
                "See the verification log above and try again.</span>"
            )
        finally:
            for control, was_disabled in previous_disabled:
                control.disabled = was_disabled
            check_btn.description = "Check sign-in"

    def _apply_key():
        _ensure_codex_method("key")
        value = key_in.value.strip()
        if not value:
            raise ValueError("API key is required.")
        lab.clear_codex_credentials()
        lab.OPENAI_API_KEY = value
        return lab.check_auth("codex")

    def _apply_endpoint():
        _ensure_codex_method("endpoint")
        base = lab.validate_codex_endpoint(base_in.value, tok_in.value, model_in.value)
        lab.clear_codex_credentials()
        lab.OPENAI_BASE_URL = base
        lab.OPENAI_API_KEY = tok_in.value.strip()
        lab.OPENAI_MODEL = model_in.value.strip()
        return lab.check_auth("codex")

    start_btn.on_click(_on_start)
    check_btn.on_click(_on_check)
    key_apply.on_click(lambda _: lab.run_step(out2, key_apply, "Apply & verify", _apply_key))
    ep_apply.on_click(lambda _: lab.run_step(out2, ep_apply, "Apply & verify", _apply_endpoint))

    display(w.VBox([method, account_box, key_box, endpoint_box, out2]))
elif w is not None and agent == "claude":
    method = w.Dropdown(
        options=[("Account sign-in", "account"), ("API key", "key"), ("Custom endpoint", "endpoint")],
        value="account", description="Method:",
    )

    # -- Account sign-in (OAuth paste-code) --
    acct_kind = w.Dropdown(
        options=[("Claude subscription (--claudeai)", "claudeai"),
                 ("Anthropic Console / API billing (--console)", "console")],
        value="claudeai", description="Account:", layout=w.Layout(width="55%"),
    )
    start_btn = w.Button(description="Start sign-in", button_style="primary", icon="sign-in")
    code_in = w.Text(description="Code:", placeholder="paste the code from the browser",
                     layout=w.Layout(width="55%"))
    submit_btn = w.Button(description="Submit", button_style="success", icon="check")
    account_box = w.VBox([acct_kind, w.HBox([start_btn, code_in, submit_btn])])

    # -- API key --
    key_in = w.Password(description="API key:", layout=w.Layout(width="55%"),
                        placeholder="Anthropic API key")
    key_apply = w.Button(description="Apply & verify", button_style="info", icon="key")
    key_box = w.VBox([key_in, key_apply])

    # -- Custom endpoint (Anthropic-compatible, Claude) --
    base_in = w.Text(description="Base URL:", layout=w.Layout(width="55%"),
                     placeholder="https://your-proxy/v1")
    tok_in = w.Password(description="Token:", layout=w.Layout(width="55%"),
                        placeholder="token for the custom endpoint")
    claude_model_in = w.Text(
        description="Model:", layout=w.Layout(width="55%"),
        value=getattr(lab, "ANTHROPIC_MODEL", ""),
        placeholder="model exposed by the endpoint",
    )
    ep_apply = w.Button(description="Apply & verify", button_style="info", icon="key")
    endpoint_box = w.VBox([base_in, tok_in, claude_model_in, ep_apply])

    out2 = w.Output(layout=w.Layout(border="1px solid #ccc"))
    _boxes = {"account": account_box, "key": key_box, "endpoint": endpoint_box}

    def _show(which):
        for _k, _b in _boxes.items():
            _b.layout.display = "" if _k == which else "none"

    def _on_method(change):
        if change["name"] == "value":
            lab.cancel_login()
            lab.clear_claude_credentials()
            out2.clear_output()
            _show(change["new"])

    method.observe(_on_method, names="value")
    _show(method.value)

    def _on_start(_):
        _ensure_step3_agent("claude")
        use_console = acct_kind.value == "console"
        lab.run_step(out2, start_btn, "Start sign-in", lambda: lab.login_start(use_console))

    def _on_submit(_):
        _ensure_step3_agent("claude")
        code = code_in.value
        # login_submit prints `claude auth status`, which is the verification for OAuth
        # (check_auth's smoke only fires for env-based key/endpoint creds, not stored OAuth).
        lab.run_step(out2, submit_btn, "Sign in", lambda: lab.login_submit(code))

    def _apply_key():
        _ensure_step3_agent("claude")
        value = lab.validate_claude_api_key(key_in.value)
        lab.ANTHROPIC_API_KEY = value
        lab.ANTHROPIC_BASE_URL = ""
        lab.ANTHROPIC_AUTH_TOKEN = ""
        return lab.check_auth("claude", "api-key")

    def _apply_endpoint():
        _ensure_step3_agent("claude")
        base = lab.validate_claude_endpoint(
            base_in.value, tok_in.value, claude_model_in.value
        )
        lab.ANTHROPIC_API_KEY = ""
        lab.ANTHROPIC_BASE_URL = base
        lab.ANTHROPIC_AUTH_TOKEN = tok_in.value.strip()
        lab.ANTHROPIC_MODEL = claude_model_in.value.strip()
        return lab.check_auth("claude", "custom-endpoint")

    start_btn.on_click(_on_start)
    submit_btn.on_click(_on_submit)
    key_apply.on_click(lambda _: lab.run_step(out2, key_apply, "Apply & verify", _apply_key))
    ep_apply.on_click(lambda _: lab.run_step(out2, ep_apply, "Apply & verify", _apply_endpoint))

    display(w.VBox([method, account_box, key_box, endpoint_box, out2]))
elif w is None:
    if agent == "codex":
        print("ipywidgets unavailable -- use the Codex fallback:")
        print("  # account : lab.login_start(); lab.check_auth('codex')")
        print("  # api key : lab.OPENAI_API_KEY='sk-...'; lab.check_auth('codex')")
        print("  # endpoint: lab.OPENAI_BASE_URL='https://.../v1'; lab.OPENAI_API_KEY='...'; lab.OPENAI_MODEL='...'; lab.check_auth('codex')")
    else:
        print("ipywidgets unavailable -- plain fallback, run ONE of:")
        print("  #  account : lab.login_start(); lab.login_submit('<code>')        # --claudeai (default)")
        print("  #            lab.login_start(use_console=True); lab.login_submit('<code>')  # --console")
        print("  #  api key : lab.ANTHROPIC_API_KEY='sk-...'; lab.check_auth('claude', 'api-key')")
        print("  #  endpoint: lab.ANTHROPIC_BASE_URL='https://.../v1'; lab.ANTHROPIC_AUTH_TOKEN='...'; lab.ANTHROPIC_MODEL='...'; lab.check_auth('claude', 'custom-endpoint')")
else:
    print(f"Unknown agent '{agent}' -- rerun Step 3, then rerun this authentication cell.")
'''

CODE_AUTH = (
    "def _build_auth_controls(_lab, _agent):\n"
    "    lab = _lab\n"
    "    agent = _agent\n"
    + textwrap.indent(CODE_AUTH_BODY, "    ")
    + "\n\n_build_auth_controls(lab, getattr(lab, \"AGENT\", \"\").strip().lower())\n"
)

MD_STEP3 = """\
## 5. Pick a prompt and generate

This is the authoring stage. Select an example from the dropdown to load its full text into the
editor, then run it as-is, refine it, or replace it with your own scenario. The selected prompt is
what the agent will use to write the project, so read it before clicking **Generate**.

### Choose a prompt

Use the table below to choose a scenario that matches your goal. For a first run, start with
`video_infer_app`; model import and profiling prompts download more files and take longer. The
time columns are rough planning estimates, not guarantees.

| Prompt | Typical usage | Expected result | Generate | Run & validate |
| --- | --- | --- | ---: | ---: |
| `import_vision_model_detection_pipeline` | Onboard an object-detection model end to end: convert it, build a TensorRT engine, benchmark it, and document the result | Model files, DeepStream configs, benchmark evidence, and a PDF report | ~15–30 min | ~1–2 min |
| `ds_profiling_efficient_pipeline` | Measure a multi-stream pipeline, identify its bottleneck, and estimate the sustainable stream count | Nsight-based profiling report and an optimized pipeline project | ~7–12 min | ~12–20 min |
| `msgbroker_nats` | Build a NATS/JetStream `nvds_msgapi` protocol adapter for DeepStream object metadata | Native adapter shared library, sample config, and local subscriber validation evidence | ~10–15 min | ~7–12 min |
| `msgconv_kafka` | Convert DeepStream metadata and publish it to a Kafka topic | Runnable pipeline plus captured Kafka JSON messages | ~3–6 min | ~5–10 min |
| `multi_stream_tracker` | Combine four RTSP sources with inference, tracking, and a tiled display | Annotated tiled MP4 with persistent track IDs | ~3–6 min | ~15–25 min |
| `nvdsanalytics_config_sample` | Demonstrate ROI filtering, line crossing, overcrowding, and direction rules | Annotated MP4 showing `nvdsanalytics` results | ~5–10 min | ~4–8 min |
| `nvdsdynamicsrcbin_app` | Exercise dynamic source addition and removal in a running DeepStream pipeline | Runnable dynamic-source app and lifecycle logs | ~3–6 min | ~4–8 min |
| `rtvi_vlm_core_app & rtvi_vlm_openapi_spec` | Build the RTVI VLM pipeline, then add a FastAPI service around it in a second generation stage | VLM summaries, Kafka output, service code, OpenAPI spec, and API response | ~15–25 min | ~8–15 min |
| `single_view_3d_tracker` | Track objects in 3D from a single camera and render the tracked result | Annotated MP4 from the single-view 3D tracker | ~7–12 min | ~8–15 min |
| `video_infer_app` | Start with a compact file-based TrafficCamNet inference and OSD example | Runnable starter project and annotated MP4 | ~2–5 min | ~3–6 min |
| `video_object_count` | Count detected objects in a video and present the counts with the stream | Object-count application and annotated MP4 | ~2–5 min | ~3–6 min |
| `video_parallel_infer_app` | Run parallel inference branches and merge their metadata into one output | Multi-model pipeline and annotated merged MP4 | ~12–20 min | ~20–30 min |
| `yolov26s_detection` | Convert YOLO26s for DeepStream and build a complete detection application | Converted model, DeepStream project, and annotated MP4 | ~5–10 min | ~8–15 min |

The `rtvi_vlm_core_app & rtvi_vlm_openapi_spec` entry is one two-stage workflow. The notebook runs
both prompts in order and shows the combined generated result only after the second stage finishes.

The ranges are coarse and assume a warm environment. A first run may take longer because images,
models, packages, or TensorRT engines need to be downloaded or built. Actual time also depends on
the GPU, network, and how much repair the selected scenario needs.

A productive prompt states the desired **input**, **processing or model**, **output**,
**integration constraints**, and **evidence of success**. Be explicit about details that matter
to your deployment -- for example, RTSP versus a file source, message schema, service ports,
model format, expected artifact, or a bounded run duration. The DeepStream skills supply
framework conventions, but they cannot infer unstated product requirements.

Click **Generate** after selecting or editing the prompt. This stage writes and previews the
project; it does not prove that services, models, streams, or external endpoints work in the
current runtime. An agent may perform its own checks, while the notebook's runtime validation is
Step 6.

**Checkpoint:** inspect the generated file list and summary before running it. Generated files are
under the host-side `WORKSPACE`; run artifacts are grouped under
`WORKSPACE/agent_outputs/<prompt_id>/`. Use `lab.show_generated_code()` to redisplay the preview
without another agent call.
"""

CODE_STEP3 = '''\
try:
    import ipywidgets as w
    from IPython.display import display
except ImportError:
    w = None

if w is not None:
    default_prompt = ("video_infer_app" if "video_infer_app" in lab.MENU_PROMPT_IDS
                      else lab.MENU_PROMPT_IDS[0])
    prompt_dd = w.Dropdown(options=lab.MENU_PROMPT_IDS, value=default_prompt,
                           description="Prompt:", layout=w.Layout(width="60%"))
    prompt_status = w.HTML()   # visible confirmation a pick loaded (prompts share a first line)
    prompt_tx = w.Textarea(value=lab.prompt_text(prompt_dd.value),
                           layout=w.Layout(width="98%", height="240px"))
    lab.set_selection(lab.AGENT, prompt_dd.value)

    def _set_status(pid):
        prompt_status.value = (f"<span style='color:#1a7f37'>&#9989; loaded prompt: "
                               f"<b>{pid}</b> &mdash; edit below or pick another</span>")

    def _on_pick(change):   # dropdown -> textarea: load the picked prompt's text (live link)
        lab.set_selection(lab.AGENT, change["new"])
        prompt_tx.value = lab.prompt_text(change["new"])
        _set_status(change["new"])      # green line changes even when the text's first line looks the same

    def _on_prompt_edit(change):
        if change["name"] == "value":
            lab.state["generated"] = False
            run_control = globals().get("run_btn")
            if run_control is not None:
                run_control.disabled = True

    prompt_dd.observe(_on_pick, names="value")
    prompt_tx.observe(_on_prompt_edit, names="value")
    _set_status(prompt_dd.value)

    gen_btn = w.Button(description="Generate", button_style="primary", icon="cogs")
    out3 = w.Output()

    def _on_generate(_):
        pid, txt = prompt_dd.value, prompt_tx.value          # read widgets on the UI thread
        generated_ok = lab.run_step(out3, gen_btn, "Generate",
                              lambda: (lab.select_from_ui(pid, txt), lab.generate()),
                              requires="installed", success_flag="generated",
                              controls=tuple(c for c in (
                                  prompt_dd, prompt_tx, globals().get("run_btn")
                              ) if c is not None))
        run_control = globals().get("run_btn")
        if run_control is not None:
            run_control.disabled = not generated_ok

    gen_btn.on_click(_on_generate)
    display(prompt_dd, prompt_status, prompt_tx, gen_btn, out3)
else:
    print("ipywidgets unavailable -- run the loader cell, or:  "
          "lab.set_selection(lab.AGENT, 'video_infer_app'); lab.generate()")
'''

MD_STEP4 = """\
## 6. Run and view results

This is the validation stage. Click **Run & view results** once and wait for the final result. The
button stays disabled while the work is in progress. The generated project is copied to an isolated
run workspace first; service setup, repairs, and runtime changes are applied only to that copy.

The button runs three phases:

1. **Prepare services:** start only the infrastructure required by the selected scenario, such
   as an RTSP source, Kafka, NATS, or the optional VLM endpoint.
2. **Run and repair:** ask the agent to execute the run copy, inspect concrete failures, patch the
   copy when needed, and retry within the configured limits.
3. **Render evidence:** collect the scenario-specific artifact rather than relying only on a
   process exit code.

Depending on the prompt, evidence can be an annotated video preview, Kafka or NATS messages, a
PDF or profiling report, service logs, an OpenAPI document, or a live API response. Long-running
video and service scenarios are intentionally bounded so that control returns to the notebook.

**Checkpoint:** review the inline result and the corresponding directory under
`WORKSPACE/agent_outputs/<prompt_id>/`. Agent transcripts explain what was attempted; application
logs and artifacts provide the stronger proof that the scenario ran. If the result is missing, read
the first runtime error before retrying. Use `lab.show_results()` to render saved artifacts again
after a page refresh without repeating generation or execution.
"""

CODE_STEP4 = '''\
try:
    import ipywidgets as w
    from IPython.display import display
except ImportError:
    w = None

if w is not None:
    run_btn = w.Button(description="Run & view results", button_style="success", icon="play",
                       disabled=not lab.state["generated"])
    out4 = w.Output()

    def _on_run(_):
        # Start services, then RE-INVOKE the agent to run + fix the app it wrote, then show results.
        lab.run_step(out4, run_btn, "Run",
                              lambda: (lab.prepare_services(), lab.run_and_fix(), lab.show_results()),
                              requires="generated",
                              controls=tuple(c for c in (
                                  globals().get("prompt_dd"), globals().get("prompt_tx"),
                                  globals().get("gen_btn")
                              ) if c is not None))

    run_btn.on_click(_on_run)
    display(run_btn, out4)
else:
    print("ipywidgets unavailable -- run: lab.prepare_services(); lab.run_and_fix(); lab.show_results()")
'''

MD_CLEANUP = """\
## 7. Cleanup _(optional)_

Click **Cleanup containers** when you are finished or need to release GPU and service resources.
Cleanup removes the lab containers and scenario services managed by the notebook.

Generated projects and result artifacts are kept in `OUTPUT_ROOT`, so you can inspect, download,
or reuse them after cleanup. To run another scenario later, return to Step 2, recreate the
container, and repeat installation and authentication before generating again.

**Before leaving the lab:** save the generated project, its README, the main result artifact, and
the run transcript. Together they record what was requested, generated, and validated.
"""

CODE_CLEANUP = '''\
try:
    import ipywidgets as w
    from IPython.display import display
except ImportError:
    w = None

if w is not None:
    clean_btn = w.Button(description="Cleanup containers", button_style="danger", icon="trash")
    outc = w.Output()

    def _on_cleanup(_):
        lab.run_step(outc, clean_btn, "Cleanup", lab.cleanup_containers)

    clean_btn.on_click(_on_cleanup)
    display(clean_btn, outc)
else:
    print("ipywidgets unavailable -- run: lab.cleanup_containers()")
'''

# Prepended to every step cell: the FIRST one to run loads `lab` AND ensures the control deps,
# then renders -- so "load the kernel" and "show the controls" are ONE step (no separate loader
# cell). Idempotent: skips if `lab` is already defined.
_BOOT = '''\
import sys, pathlib  # first step cell to run loads the lab + control deps, then renders
if "lab" not in globals():
    _r = next((p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
               if (p / "deploy/brev/scripts/ds_agent_lab.py").exists()), pathlib.Path.cwd())
    sys.path.insert(0, str(_r / "deploy/brev/scripts")); import ds_agent_lab as lab
    lab.ensure_ipywidgets()

'''

# FIRST lab-loading cell ONLY (Step 3, Install): (re)load the lab modules FRESH -- drop any cached
# copies first -- so a REDEPLOY of the .py files takes effect just by re-running it (no kernel
# restart needed). Later steps use _BOOT (reuse the already-loaded `lab`) so the agent/creds/
# selection set in the Install/Sign-in/Generate steps survive across cells.
_BOOT_FRESH = '''\
import sys, pathlib  # Install step: (re)load the lab modules fresh, ensure control deps, then render
_old_lab = globals().get("lab")
if _old_lab is not None:
    try:
        _old_lab.cancel_login()
    except Exception:
        pass
_r = next((p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
           if (p / "deploy/brev/scripts/ds_agent_lab.py").exists()), pathlib.Path.cwd())
sys.path.insert(0, str(_r / "deploy/brev/scripts"))
for _m in ("ds_agent_lab", "ds_lab_config"):
    sys.modules.pop(_m, None)
import ds_agent_lab as lab
lab.ensure_ipywidgets()

'''


def build_cells():
    # Step 1 (config) is plain Python and Step 2 (prepare) is a `%%bash` cell -- neither loads
    # `lab`, so the FIRST lab-loading cell is Step 3 (Install): it gets _BOOT_FRESH so re-running
    # it picks up a redeployed .py without a kernel restart. Later steps reuse
    # the already-loaded `lab` so the agent/creds/selection state survives.
    return [
        ("md", MD_INTRO),
        ("md", MD_CONFIG),
        ("code", CODE_CONFIG),
        ("md", MD_PREPARE),
        ("code", CODE_PREPARE),
        ("md", MD_STEP2),
        ("code", _BOOT_FRESH + CODE_STEP2),
        ("md", MD_AUTH),
        ("code", _BOOT + CODE_AUTH),
        ("md", MD_STEP3),
        ("code", _BOOT + CODE_STEP3),
        ("md", MD_STEP4),
        ("code", _BOOT + CODE_STEP4),
        ("md", MD_CLEANUP),
        ("code", _BOOT + CODE_CLEANUP),
    ]


def main() -> int:
    nb = new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    cells = []
    for idx, (kind, src) in enumerate(build_cells()):
        src = src.rstrip("\n")
        if kind == "md":
            cell = new_markdown_cell(src)
        else:
            cell = new_code_cell(src)
            cell["execution_count"] = None
            cell["outputs"] = []
        cell["id"] = f"cell-{idx:02d}"
        cells.append(cell)
    nb["cells"] = cells

    out_path = Path(__file__).resolve().parents[1] / "deepstream_code_agent_launchable.ipynb"
    with out_path.open("w") as f:
        nbformat.write(nb, f)
    text = out_path.read_text()
    if not text.endswith("\n"):
        out_path.write_text(text + "\n")
    print(f"Wrote {out_path} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
