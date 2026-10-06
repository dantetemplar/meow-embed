"""Multimodal JSON transport: upload client files, never resolve server paths."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from meow_embed.types import EmbedInput, EmbedRequestPayload, MediaDataDict

MEDIA_MODALITIES = ("image", "video", "audio")


def validate_multimodal_input(item: dict[str, object]) -> None:
    unknown = item.keys() - {"text", *MEDIA_MODALITIES}
    if unknown:
        raise ValueError(f"Unknown input keys: {sorted(unknown)}")
    text = item.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Multimodal text must be a string.")
    counts: dict[str, int] = {}
    for modality in MEDIA_MODALITIES:
        if modality not in item:
            counts[modality] = 0
            continue
        value = item[modality]
        values = value if isinstance(value, (list, tuple)) else [value]
        if not values:
            raise ValueError(f"{modality} must contain at least one media item.")
        counts[modality] = len(values)
    if not text and not any(counts.values()):
        raise ValueError("Input must contain text or media.")
    if any(f"<|{modality}|>" in text for modality in MEDIA_MODALITIES):
        for modality, count in counts.items():
            if text.count(f"<|{modality}|>") != count:
                raise ValueError(
                    f"Number of <|{modality}|> placeholders "
                    f"must match {modality} items."
                )


def _media_data(value: object) -> MediaDataDict:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        return {
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "filename": path.name,
        }
    if isinstance(value, bytes):
        return {"data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        if value.keys() - {"data", "filename"} or not isinstance(
            value.get("data"), str
        ):
            raise ValueError("Media must contain base64 data and an optional filename.")
        if "filename" in value and not isinstance(value["filename"], str):
            raise ValueError("Media filename must be a string.")
        try:
            raw = base64.b64decode(value["data"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 media data.") from exc
        encoded = base64.b64encode(raw).decode("ascii")
        if "filename" in value:
            return {"data": encoded, "filename": value["filename"]}
        return {"data": encoded}
    raise ValueError("Media must be a file path, bytes, or a base64 data object.")


def normalize_input(item: EmbedInput) -> EmbedInput:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise ValueError("Each input must be a string or a multimodal dictionary.")
    normalized: dict[str, object] = dict(item)
    for modality in MEDIA_MODALITIES:
        if modality not in item:
            continue
        value = normalized[modality]
        values = (
            value
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            else [value]
        )
        normalized[modality] = [_media_data(media) for media in values]
    validate_multimodal_input(normalized)
    return cast(EmbedInput, normalized)


def normalize_embed_payload(payload: EmbedRequestPayload) -> EmbedRequestPayload:
    normalized = dict(payload)
    inputs = [normalize_input(item) for item in payload.get("texts", [])]
    normalized["texts"] = inputs
    if any(isinstance(item, dict) for item in inputs) and (
        not payload.get("dense_model_id")
        or payload.get("sparse_model_id")
        or payload.get("bge_model_id")
    ):
        raise ValueError(
            "Multimodal dictionaries require a dense model only; "
            "sparse and BGE-M3 accept text strings."
        )
    return cast(EmbedRequestPayload, normalized)
