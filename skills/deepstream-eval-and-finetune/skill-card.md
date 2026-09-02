## Description: <br>
Evaluate and improve an object detector in NVIDIA DeepStream, providing deployed mAP, FPS and latency measurement, TAO Skill Bank fine-tuning or AutoML, redeployment, and before/after reporting on HuggingFace, NGC, ONNX, or local models and KPI datasets. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to evaluate object detection model accuracy and performance in live DeepStream pipelines, fine-tune models using TAO Skill Bank or AutoML, and produce before/after comparison reports. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Run flow](references/run-flow.md) <br>
- [Accuracy evaluation](references/accuracy-eval.md) <br>
- [Fine-tuning and reporting](references/finetune-and-report.md) <br>
- [Web UI](references/ui.md) <br>
- [Script catalogue](references/scripts.md) <br>
- [Windows operation](references/windows.md) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Shell commands, Analysis] <br>
**Output Format:** [JSON metrics, PDF reports, Markdown summaries, and overlay images] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
12 evaluation tasks (12 positive), each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage through quality routing and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 53% → 76% (+23 points) |
| Security | Not available | 71% → 83% (+12 points) |
| Correctness | Not available | 47% → 82% (+35 points) |
| Discoverability | Not available | 55% → 77% (+22 points) |
| Effectiveness | Not available | 37% → 47% (+10 points) |
| Efficiency | Not available | 54% → 91% (+37 points) |

## Skill Version(s): <br>
0.6.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
