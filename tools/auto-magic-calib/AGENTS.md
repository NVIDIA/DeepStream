# Agent Guidance

Use the DeepStream repository's skills for AutoMagicCalib setup and calibration workflows.

- Use `../../skills/amc-setup-calibration-stack/SKILL.md` before running calibration, unless the user confirms that the AutoMagicCalib backend is already running and provides the service URL.
- Use `../../skills/amc-run-sample-calibration/SKILL.md` to verify a deployment with the bundled sample dataset.
- Use `../../skills/amc-run-video-calibration/SKILL.md` for pre-recorded MP4 inputs.
- Use `../../skills/amc-run-rtsp-calibration/SKILL.md` for live RTSP inputs through VIOS capture.
- Use `README.md` for product setup, UI workflow, dataset requirements, and REST API details.

Full calibration runs require a real runtime host with an NVIDIA GPU with NVENC, NVIDIA driver, Docker without sudo, NVIDIA Container Toolkit, and NGC image access. Optional VGGT refinement also requires HuggingFace access and the VGGT model. In restricted or no-GPU environments, inspect docs and prepare commands, but do not report runtime validation as complete.

Typical runtime depends on image pulls, model downloads, video length, and detector choice. The bundled sample can take several minutes up to about 30 minutes; custom video calibration can take 10-60+ minutes; VGGT refinement usually adds a few minutes when enabled.
