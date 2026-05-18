from __future__ import annotations

import torch.nn as nn
from nnsight import NNsight

from linear_transformer.patching.registry import get_registry
from linear_transformer.adapter.accessors import get_arch
from linear_transformer.adapter.adapters import ModelAdapter


def patch_model_for_lvp(
    model: nn.Module,
    model_id: str, # TODO: change to automatic mapping over huggingface registery
    include: set[type] | None = None,
    exclude: set[type] | None = None,
    nnsight_wrapper: bool = True,
    return_adapter: bool = True,
    dry_run: bool = False,
    **kwargs,
) -> nn.Module:
    registry = get_registry(model_id)
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
                    _walk(wrapper, full_name)  # patch nn.Linear etc. inside the wrapper
                else:
                    _walk(child, full_name)
            else:
                _walk(child, full_name)

    _walk(model, "")

    if dry_run:
        _print_patch_plan(replacements, dry_run)

    if nnsight_wrapper:
        model = NNsight(model)

    if return_adapter:
        arch = get_arch(model_id)
        model = ModelAdapter(model, arch)
    return model


def _print_patch_plan(
    replacements: list[tuple[str, type, type]],
    dry_run: bool,
) -> None:
    label = "[dry-run] Would patch" if dry_run else "Patched"
    print(f"lvp — {label} {len(replacements)} module(s):")
    for path, old_cls, new_cls in replacements:
        print(f"  {path}: {old_cls.__name__} → {new_cls.__name__}")

