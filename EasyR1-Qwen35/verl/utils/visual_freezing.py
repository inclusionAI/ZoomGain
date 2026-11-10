"""Utilities for configuring multimodal visual-module trainability."""

from __future__ import annotations

from typing import Any


def configure_visual_trainability(
    visual: Any,
    *,
    freeze_vision_tower: bool,
    keep_vision_merger_trainable: bool,
) -> dict[str, int | bool]:
    """Freeze a visual encoder while optionally leaving its merger trainable."""
    if keep_vision_merger_trainable and not freeze_vision_tower:
        raise ValueError(
            "keep_vision_merger_trainable=true requires freeze_vision_tower=true"
        )

    merger = getattr(visual, "merger", None)
    if freeze_vision_tower:
        visual.requires_grad_(False)
        if keep_vision_merger_trainable:
            if merger is None:
                raise RuntimeError(
                    "keep_vision_merger_trainable=true, but the visual tower exposes no merger"
                )
            merger.requires_grad_(True)

    visual_parameters = list(visual.parameters())
    merger_parameters = list(merger.parameters()) if merger is not None else []
    merger_parameter_ids = {id(parameter) for parameter in merger_parameters}
    encoder_parameters = [
        parameter
        for parameter in visual_parameters
        if id(parameter) not in merger_parameter_ids
    ]

    visual_total = sum(parameter.numel() for parameter in visual_parameters)
    visual_trainable = sum(
        parameter.numel() for parameter in visual_parameters if parameter.requires_grad
    )
    encoder_total = sum(parameter.numel() for parameter in encoder_parameters)
    encoder_trainable = sum(
        parameter.numel() for parameter in encoder_parameters if parameter.requires_grad
    )
    merger_total = sum(parameter.numel() for parameter in merger_parameters)
    merger_trainable = sum(
        parameter.numel() for parameter in merger_parameters if parameter.requires_grad
    )

    if freeze_vision_tower:
        expected_merger_trainable = merger_total if keep_vision_merger_trainable else 0
        if encoder_trainable != 0 or merger_trainable != expected_merger_trainable:
            raise RuntimeError(
                "unexpected visual trainability after freezing: "
                f"encoder={encoder_trainable}/{encoder_total} "
                f"merger={merger_trainable}/{merger_total}"
            )
    elif visual_trainable != visual_total:
        raise RuntimeError(
            "the visual tower was requested trainable but contains frozen parameters: "
            f"trainable={visual_trainable} total={visual_total}"
        )

    return {
        "freeze_vision_tower": freeze_vision_tower,
        "keep_vision_merger_trainable": keep_vision_merger_trainable,
        "visual_trainable": visual_trainable,
        "visual_total": visual_total,
        "encoder_trainable": encoder_trainable,
        "encoder_total": encoder_total,
        "merger_trainable": merger_trainable,
        "merger_total": merger_total,
    }
