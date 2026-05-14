import torch

class FramePickerProcessor:
    name = "frame-picker"

    def __init__(self, config):
        self.num_frames = config['num_frames']

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError(
                "FramePickerProcessor expects exactly one argument"
            )
        frame_list = args[0]
        if len(frame_list) != self.num_frames:
            raise ValueError(
                f"FramePickerProcessor expects a list of "
                f"{self.num_frames} frames, while got {len(frame_list)}"
            )
        torch_tensors = [
            torch.utils.dlpack.from_dlpack(frame.tensor)
            for frame in frame_list
        ]
        return torch.stack(torch_tensors)

