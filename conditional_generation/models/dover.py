"""
DOVER code is borrowed and modified from a few sources
- Parts of model definition (forward pass) modified from https://github.com/VQAssessment/DOVER/blob/master/dover/models/evaluator.py#L44
    - This is mostly re-written to keep only relevant parts of the required checkpoint
- Data pre-processing borrowed from https://github.com/VQAssessment/DOVER/blob/master/dover/datasets/dover_datasets.py
    - This mostly borrows the fragmenting part, where we create spatio-temporal tubes from video-batches
"""

import random
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.video_quality.common.backbones import SwinTransformer3D, VQAHead
from models.video_quality.common.data_utils import (get_spatial_fragments,
                                                    prepare_input)
from utils import model_utils

_DEFAULT_WEIGHTS_NAME = "DOVER"
_DEFAULT_WEIGHTS_FILE = "DOVER.pth"


def get_single_view(
    video: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    video = get_spatial_fragments(video, **kwargs)
    return video


def spatial_temporal_view_decomposition(
    video_file: bytes,
    sampler: Any,
) -> Tuple[torch.Tensor, np.ndarray]:
    video = {}
    vreader = prepare_input(video_file)
    frame_inds = sampler(len(vreader), False)
    frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}
    imgs = [frame_dict[idx] for idx in frame_inds]
    video = torch.stack(imgs, 0).permute(3, 0, 1, 2)
    sampled_video = get_single_view(video)
    return sampled_video, frame_inds


class UnifiedFrameSampler:
    def __init__(
        self,
        fsize_t: int,
        fragments_t: int,
        frame_interval: int = 1,
        num_clips: int = 1,
        drop_rate: float = 0.0,
    ) -> None:
        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames: int) -> np.ndarray:
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t

        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)

        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval
            + rnd_t[:, None]
            + tgrids[:, None]
        )

        drop = random.sample(
            list(range(self.fragments_t)), int(self.fragments_t * self.drop_rate)
        )
        dropped_ranges_t = []
        for i, rt in enumerate(ranges_t):
            if i not in drop:
                dropped_ranges_t.append(rt)
        return np.concatenate(dropped_ranges_t)

    def __call__(self, total_frames: int, start_index: int = 0) -> np.ndarray:
        frame_inds = []

        for _ in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)


_TEST_1080P_PARAMS = {
    "fragments_h": 7,
    "fragments_w": 7,
    "fsize_h": 32,
    "fsize_w": 32,
    "aligned": 32,
    "clip_len": 32,
    "frame_interval": 2,
    "num_clips": 3,
}


class DOVERTechnicalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multi = False
        self.layer = -1
        self.technical_backbone = SwinTransformer3D()
        self.vqa_head = dict(
            in_channels=768,
            hidden_channels=64,
        )
        self.technical_head = VQAHead(pre_pool=False, **self.vqa_head)

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        feat = self.technical_backbone(clips, multi=self.multi, layer=self.layer)
        score = self.technical_head(feat)
        return score


class DOVERTechnical(nn.Module):
    def __init__(self, weights_name: str, utils_only: bool = False) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.sampler = UnifiedFrameSampler(
            _TEST_1080P_PARAMS["clip_len"],
            _TEST_1080P_PARAMS["num_clips"],
            _TEST_1080P_PARAMS["frame_interval"],
        )
        self.num_clips = _TEST_1080P_PARAMS["num_clips"]
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if not utils_only:
            self.model = DOVERTechnicalModel().to(self.device)
            self.model.load_state_dict(
                torch.load(weights_name, map_location=self.device), strict=False
            )
            self.eval()
        else:
            self.model = None

    def get_technical_view(self, video_file: bytes) -> torch.Tensor:
        # Seed both RNGs before sampling so spatial patch jitter is deterministic,
        # matching the imaginaire4 implementation. Save/restore global RNG state so
        # we don't affect downstream callers.
        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        try:
            torch.manual_seed(0)
            np.random.seed(0)
            video_data, _ = spatial_temporal_view_decomposition(video_file, self.sampler)
        finally:
            torch.set_rng_state(torch_rng_state)
            np.random.set_state(np_rng_state)
        video_data = ((video_data.permute(1, 2, 3, 0) - self.mean) / self.std).permute(
            3, 0, 1, 2
        )
        return video_data

    # new interface to support pipelined usage
    def generate_score(self, video_data: torch.Tensor) -> float:
        return float(self._batch_inference([video_data])[0])

    def __call__(self, video_files: list[bytes]) -> np.ndarray:
        batched_video_data = []
        for i in range(len(video_files)):
            batched_video_data.append(self.get_technical_view(video_files[i]))
        return self._batch_inference(batched_video_data)

    @torch.no_grad()
    def _batch_inference(self, batched_video_data: list[torch.Tensor]) -> np.ndarray:
        video_data = torch.stack(batched_video_data, dim=0)
        video_data = video_data.to(self.device)
        if len(video_data.shape) == 4:
            video_data = video_data.unsqueeze(
                0
            )  # Force unsqueeze for non-batched inference
        b, c, t, h, w = video_data.shape
        video_data = (
            video_data.reshape(b, c, self.num_clips, t // self.num_clips, h, w)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(b * self.num_clips, c, t // self.num_clips, h, w)
        )
        score = self.model(video_data)
        score = score.reshape(b, self.num_clips, -1)
        score = score.mean((1, 2))
        score = (score - 0.1107) / 0.07355
        score = torch.sigmoid(score)
        score = score.cpu().detach().numpy() * 100
        return score


class DOVERVideoTechnicalScorer(model_utils.ModelInterface):
    """DOVER: Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives

    DOVER is a Video Quality Assessment method, from ICCV 2023. For every video, DOVER produces two scores an
    aesthetic and a technical score. For our immediate purposes (as of 09/06/2024), we care only about the technical
    score. The implementation is based on this assumption -- it strips down to keep only the technical components of
    the DOVER pipeline.

    General overview of the DOVER pipeline:
    1. Take a clip, spatially subsample patches but preserve corresponding patches across time.
    2. This produces spatiotemporal cubes, called "fragments".
    3. Compute appropriate positional encoding -- interpolated to account for the fact that sub-sampled
        patches may not be spatial neighbors.
    4. Feed this to a 3D Swin transformer, trained to produce technical distortion scores
    5. Higher the score, better the perceptual quality of video footage.
    """

    def __init__(self, utils_only: bool = False):
        self._utils_only = utils_only

    @property
    def conda_env_name(self) -> str:
        return "paibench-conditional-generation"

    def setup(self) -> None:
        if not self._utils_only:
            self.download_weights()
        model_dir = model_utils.get_local_dir_for_weights_name(self.weights_names[0])
        self._model = DOVERTechnical(
            (model_dir / _DEFAULT_WEIGHTS_FILE).as_posix(), self._utils_only
        )

    @property
    def weights_names(self) -> list[str]:
        return [_DEFAULT_WEIGHTS_NAME]

    def __call__(self, video_files: list[bytes]) -> np.ndarray:
        return self._model(video_files)

    # new interface to support pipelined usage
    def get_technical_view(self, video_file: bytes) -> np.ndarray:
        return self._model.get_technical_view(video_file).cpu().numpy()

    def generate_score(self, video_data: np.ndarray) -> float:
        return self._model.generate_score(torch.from_numpy(video_data))
