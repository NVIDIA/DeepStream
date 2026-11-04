## Description: <br>
Run or tune NVIDIA AutoMagicCalib from DeepStream using sample data, local MP4s, or RTSP streams. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers working in a DeepStream checkout use this skill to set up the AutoMagicCalib (AMC) service stack and produce or tune multi-camera calibration artifacts from the bundled SDG sample dataset, local MP4 files, or live RTSP streams captured through VIOS. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Other [RTSP stream / VIOS service credentials]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [references/README.md](references/README.md) <br>
- [Service setup (references/setup.md)](references/setup.md) <br>
- [SDG sample dataset (references/sample-dataset.md)](references/sample-dataset.md) <br>
- [Local MP4 files (references/video-files.md)](references/video-files.md) <br>
- [RTSP through VIOS (references/rtsp.md)](references/rtsp.md) <br>
- [Custom-dataset tuning (references/tuning.md)](references/tuning.md) <br>
- [Calibration parameters (references/calibration-parameters.md)](references/calibration-parameters.md) <br>
- [Troubleshooting (references/troubleshooting.md)](references/troubleshooting.md) <br>
- [NVIDIA-AI-IOT AutoMagicCalib repository](https://github.com/NVIDIA-AI-IOT/auto-magic-calib) <br>
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, API Calls, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks, plus calibration artifact files written under repo-rooted output directories] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reports command sequence, service health, project/job IDs, rectification and post-process state, generated calibration file paths, and quality warnings; credentials are never printed or persisted.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
14 evaluation tasks (14 positive), 1 attempt per task, each run in an isolated k8s sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of `skill_efficiency` and `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.7% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 82.1% → 96.4% (+14.3 points) | 89.3% → 92.9% (+3.6 points) |
| Correctness | 38.6% → 100.0% (+61.4 points) | 58.6% → 95.7% (+37.1 points) |
| Discoverability | 82.9% — baseline ran, but no comparable score was available; uplift unavailable | 77.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.9% → 84.6% (+42.7 points) | 51.9% → 82.0% (+30.1 points) |
| Efficiency | 89.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
3.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
