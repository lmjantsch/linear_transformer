from __future__ import annotations

import torch.nn as nn
from nnsight import NNsight

from linear_transformer.patching.registry import get_registry


def patch_model_for_lvp(
    model: nn.Module,
    include: set[type] | None = None,
    exclude: set[type] | None = None,
    nnsight_wrapper: bool = True,
    dry_run: bool = False,
    **kwargs: dict | None
) -> nn.Module:
    registry = get_registry()
    replacements: list[tuple[str, type, type]] = []  # (path, old_cls, new_cls)

    def _walk(
        parent: nn.Module,
        prefix: str,
    ) -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            child_type = type(child)

            # Determine whether to patch this child
            in_registry = child_type in registry
            passes_include = include is None or child_type in include
            passes_exclude = exclude is None or child_type not in exclude

            if in_registry and passes_include and passes_exclude:
                new_cls = type(registry[child_type](child))  # peek at wrapper type
                replacements.append((full_name, child_type, new_cls))
                if not dry_run:
                    wrapper = registry[child_type](child, **kwargs)
                    setattr(parent, name, wrapper)
                # Do not recurse into replaced modules — the wrapper owns them
            else:
                _walk(child, full_name)

    _walk(model, "")

    if dry_run:
        _print_patch_plan(replacements, dry_run)

    if nnsight_wrapper:
        model = NNsight(model)

    return model


def _print_patch_plan(
    replacements: list[tuple[str, type, type]],
    dry_run: bool,
) -> None:
    label = "[dry-run] Would patch" if dry_run else "Patched"
    print(f"lvp — {label} {len(replacements)} module(s):")
    for path, old_cls, new_cls in replacements:
        print(f"  {path}: {old_cls.__name__} → {new_cls.__name__}")

