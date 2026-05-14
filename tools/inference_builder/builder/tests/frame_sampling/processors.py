import torch
import logging

logger = logging.getLogger(__name__)

class FramePickerProcessor:
    name = "frame-picker"

    def __init__(self, config):
        self.num_frames = config['num_frames']
        # Eagerly initialize CUDA to avoid GIL deadlock with GStreamer threads
        torch.cuda.init()

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

        # Log frame dimensions for verification
        if torch_tensors:
            # Tensor shape is typically (H, W, C) from MediaExtractor
            h, w = torch_tensors[0].shape[0], torch_tensors[0].shape[1]
            logger.info(f"FramePickerProcessor: received {len(torch_tensors)} frames with dimensions {w}x{h}")

        return torch.stack(torch_tensors)
