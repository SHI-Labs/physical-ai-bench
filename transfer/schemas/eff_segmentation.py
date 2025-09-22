"""Schemas for the text segmentation v3p0 pipeline."""

import pickle
from typing import Any, Tuple

import attrs
import cattrs
import numpy as np

import pycocotools  # pyright: ignore[reportMissingModuleSource]
import pycocotools.mask  # pyright: ignore[reportMissingModuleSource]

from utils import cattrs_utils


@attrs.define
class RleMask:
    _data: dict[str, Any]

    @classmethod
    def encode(cls, mask: np.ndarray) -> "RleMask":
        return RleMask(pycocotools.mask.encode(np.array(mask, order="F")))  # type: ignore

    def decode(self) -> np.ndarray:
        return pycocotools.mask.decode(self._data).astype(bool)  # type: ignore


@attrs.define
class RleMaskSAMv2:
    data: dict[str, Any]  # data is public, since we may want to decode it with pycocotools outside of this class
    mask_shape: Tuple

    @classmethod
    def encode(cls, mask: np.ndarray) -> "RleMaskSAMv2":
        mask = np.array(mask, order="F")
        return RleMaskSAMv2(pycocotools.mask.encode(np.array(mask.reshape(-1, 1), order="F")), mask.shape)  # type: ignore

    def decode(self) -> np.ndarray:
        return pycocotools.mask.decode(self.data).astype(bool).reshape(self.mask_shape)  # type: ignore


@attrs.define
class Detection:
    # phrase: str
    confidence: float
    # grounding_dino_box: common.BoundingBox
    segmentation_mask_rle: RleMask
    # segmentation_mask_pixel_fraction: float


@attrs.define
class SAMV2Detection:
    phrase: str
    segmentation_mask_rle: RleMaskSAMv2

    def to_dict(self) -> dict[str, Any]:
        return cattrs.Converter().unstructure(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SAMV2Detection":
        return cattrs.structure(data, cls)


@attrs.define
class ResultsForCaption:
    caption: str
    detections: list[Detection]


@attrs.define
class Sample:
    key: str
    masks: np.ndarray

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        return cattrs_utils.get_custom_cattrs_converter().structure(data, cls)

    def to_dict(self) -> dict[str, Any]:
        return cattrs_utils.get_custom_cattrs_converter().unstructure(self)
