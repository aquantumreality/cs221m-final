from __future__ import annotations

"""pyvene-based layer x token-position activation patching.

This is the **baseline** for the project: a vanilla ("hard") interchange
intervention that simply copies an activation at a single
``(layer, component, position)`` site from the source run into the base
run. We do *not* learn a rotation matrix here -- that's DAS, and lives
elsewhere.

API at a glance
---------------
- :func:`run_patching_sweep` -- **primary** API. Sweeps a grid of
  ``(layer, component, position)`` sites over a list of
  :class:`~src.data.counterfactual_pairs.CounterfactualExample`,
  returns a tidy :class:`pandas.DataFrame` with one row per
  ``(example, layer, component, position)`` cell.
- :func:`run_single_patch` -- patch a single ``(layer, position)`` site
  and return the logits + label predictions for the batch.
- :func:`run_activation_patching_sweep` -- *legacy* sweep that returns
  a :class:`PatchingResult` (layer x position grid). Kept for
  back-compat; prefer :func:`run_patching_sweep`.

Implementation notes
--------------------
- We use the pyvene "interchange intervention" pattern:

  .. code-block:: python

      config = IntervenableConfig(
          model_type=type(model),
          representations=[
              RepresentationConfig(layer, component, "pos", 1),
          ],
          intervention_types=VanillaIntervention,
      )
      intervenable = IntervenableModel(config, model)
      _, cf_out = intervenable(
          base_inputs, [source_inputs], {"sources->base": pos}
      )

  ``unit="pos"`` and ``max_number_of_units=1`` mean we address a single
  *token position* (as opposed to e.g. a single attention head).
- Valid ``component`` strings depend on the model type. For GPT-2 style
  models, the standard ones are ``"block_output"`` (post-block residual
  stream; default), ``"attention_input"``, ``"attention_output"``,
  ``"mlp_output"``, ``"mlp_input"``, etc.
- We use :class:`pyvene.VanillaIntervention` ("hard patch"); no
  learnable parameters.
"""


import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pyvene as pv

from .causal_models import ID2LABEL
from .data import CounterfactualExample
from .das import compute_iia
# inline:
from .das import (
    LabelVerbalizer,
    decode_label,
    label_logit_diff,
    logit_recovery,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PatchingResult:
    """Output of an activation patching sweep.

    Attributes
    ----------
    metric_name:
        Either ``"logit_recovery"`` or ``"iia"``.
    grid:
        Numpy array of shape ``[n_layers, n_positions]`` with the metric
        averaged across the evaluated batch.
    layers:
        Layer indices corresponding to the rows of ``grid``.
    positions:
        Token positions corresponding to the columns of ``grid``.
    position_labels:
        Optional list of human-readable strings (e.g. decoded tokens) of
        length ``n_positions``, used for axis labels in plots.
    clean_logit_diff:
        Mean logit difference on the *clean* (un-patched) base run.
    corrupted_logit_diff:
        Mean logit difference on the source run with the base's gold
        label -- the "noise baseline".
    base_accuracy:
        Optional: argmax accuracy of the unpatched base run.
    extras:
        Free-form dict for anything else the sweep wants to surface.
    """

    metric_name: str
    grid: np.ndarray
    layers: List[int]
    positions: List[int]
    position_labels: Optional[List[str]] = None
    clean_logit_diff: Optional[float] = None
    corrupted_logit_diff: Optional[float] = None
    base_accuracy: Optional[float] = None
    extras: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal: build an intervenable model for a given layer
# ---------------------------------------------------------------------------


def _build_intervenable(
    model: torch.nn.Module,
    layer: int,
    component: str = "block_output",
) -> pv.IntervenableModel:
    """Wrap ``model`` with a vanilla single-site interchange intervention."""
    config = pv.IntervenableConfig(
        model_type=type(model),
        representations=[
            pv.RepresentationConfig(
                layer,        # which transformer block
                component,    # "block_output" = post-block residual stream
            ),
        ],
        intervention_types=pv.VanillaIntervention,
    )
    intervenable = pv.IntervenableModel(config, model)
    intervenable.set_device(next(model.parameters()).device)
    intervenable.disable_model_gradients()
    return intervenable


# ---------------------------------------------------------------------------
# Single-site patch
# ---------------------------------------------------------------------------


def run_single_patch(
    model: torch.nn.Module,
    *,
    layer: int,
    position: int,
    base_input_ids: torch.Tensor,
    source_input_ids: torch.Tensor,
    base_attention_mask: Optional[torch.Tensor] = None,
    source_attention_mask: Optional[torch.Tensor] = None,
    component: str = "block_output",
    intervenable: Optional[pv.IntervenableModel] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a single interchange-intervention forward pass.

    Parameters
    ----------
    model:
        Any HuggingFace causal LM. Must already be on the right device.
    layer:
        Transformer block index whose ``component`` output we patch.
    position:
        Token position (single int -- *same* position for the entire
        batch). pyvene also supports per-example positions; for the
        simple position-aligned setting in this project a single int is
        sufficient.
    base_input_ids, source_input_ids:
        LongTensors of shape ``[B, T]``.
    base_attention_mask, source_attention_mask:
        Optional attention masks of shape ``[B, T]``.
    component:
        Which sub-module of the transformer block to intervene on.
        ``"block_output"`` is the post-block residual stream (default
        and recommended). pyvene also supports ``"attention_output"``,
        ``"mlp_output"``, etc.
    intervenable:
        Optionally reuse an :class:`IntervenableModel` already built for
        this layer. If ``None``, we build one on the fly.

    Returns
    -------
    base_logits, counterfactual_logits:
        Full LM logits ``[B, T, V]`` for the un-intervened base run and
        the patched run respectively.
    """
    own_intervenable = intervenable is None
    if own_intervenable:
        intervenable = _build_intervenable(model, layer=layer, component=component)

    base_inputs = {"input_ids": base_input_ids}
    if base_attention_mask is not None:
        base_inputs["attention_mask"] = base_attention_mask
    source_inputs = {"input_ids": source_input_ids}
    if source_attention_mask is not None:
        source_inputs["attention_mask"] = source_attention_mask

    base_out, cf_out = intervenable(
        base_inputs,
        [source_inputs],
        # Single-position patch: copy source position -> base position.
        {"sources->base": position},
    )

    # ``base_out`` may be ``None`` in some pyvene versions when only the
    # counterfactual is requested -- in that case do an explicit clean
    # forward pass.
    if base_out is None:
        with torch.no_grad():
            base_out = model(**base_inputs)

    return base_out.logits, cf_out.logits


# ---------------------------------------------------------------------------
# Layer x position sweep
# ---------------------------------------------------------------------------


def _detect_num_layers(model: torch.nn.Module) -> int:
    """Best-effort number of transformer blocks for HF causal LMs."""
    cfg = getattr(model, "config", None)
    for attr in ("n_layer", "num_hidden_layers", "num_layers"):
        if cfg is not None and hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError("Could not infer number of layers from model.config")


def run_activation_patching_sweep(
    model: torch.nn.Module,
    dataset,
    *,
    verbalizer: LabelVerbalizer,
    layers: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[int]] = None,
    component: str = "block_output",
    metric: str = "logit_recovery",
    batch_size: int = 8,
    position_labels: Optional[Sequence[str]] = None,
    progress: bool = True,
) -> PatchingResult:
    """Sweep activation patching over ``(layer, position)`` grid.

    For every pair (l, p) we patch the activation at block ``l``, token
    ``p`` from source into base for the whole dataset, and reduce the
    batch with the requested metric.

    Parameters
    ----------
    model:
        HuggingFace causal LM (already on device, in eval mode).
    dataset:
        A :class:`~src.data.counterfactual_pairs.CounterfactualDataset`
        (or any iterable of dicts with the same keys).
    verbalizer:
        :class:`LabelVerbalizer` matching the dataset's label space.
    layers:
        Layers to sweep. Defaults to ``range(num_layers)``.
    positions:
        Token positions to sweep. Defaults to ``range(seq_len)`` of the
        first batch.
    component:
        Which sub-module of the block to patch. See :func:`run_single_patch`.
    metric:
        Either ``"logit_recovery"`` (default, scalar in roughly [0, 1])
        or ``"iia"`` (interchange intervention accuracy in [0, 1]).
    batch_size:
        DataLoader batch size for the inner forward passes.
    position_labels:
        Optional human-readable labels for each position (e.g. decoded
        tokens) of length ``len(positions)``. Stored on the result for
        plotting.
    progress:
        If True, show a tqdm progress bar over ``layers``.

    Returns
    -------
    :class:`PatchingResult`
    """
    if metric not in ("logit_recovery", "iia"):
        raise ValueError(f"Unknown metric {metric!r}")

    device = next(model.parameters()).device

    # ---- inspect dataset shape ------------------------------------------
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    first = next(iter(loader))
    seq_len = int(first["base_input_ids"].shape[1])

    if layers is None:
        layers = list(range(_detect_num_layers(model)))
    else:
        layers = list(layers)
    if positions is None:
        positions = list(range(seq_len))
    else:
        positions = list(positions)

    if position_labels is not None and len(position_labels) != len(positions):
        raise ValueError("position_labels length must match positions")

    # ---- compute clean / corrupted baselines (once) ---------------------
    clean_diffs: List[float] = []
    corrupted_diffs: List[float] = []
    base_correct: List[int] = []
    base_total: List[int] = []

    with torch.no_grad():
        for batch in loader:
            b_ids = batch["base_input_ids"].to(device)
            s_ids = batch["source_input_ids"].to(device)
            b_attn = batch["base_attention_mask"].to(device)
            s_attn = batch["source_attention_mask"].to(device)
            base_label = batch["base_label_id"].to(device)
            cf_label = batch["counterfactual_label_id"].to(device)

            base_out = model(input_ids=b_ids, attention_mask=b_attn)
            src_out = model(input_ids=s_ids, attention_mask=s_attn)

            clean = label_logit_diff(
                base_out.logits, cf_label, verbalizer, attention_mask=b_attn,
                reduction="none",
            )
            corrupted = label_logit_diff(
                src_out.logits, cf_label, verbalizer, attention_mask=s_attn,
                reduction="none",
            )
            clean_diffs.append(clean.detach().cpu())
            corrupted_diffs.append(corrupted.detach().cpu())

            preds = decode_label(base_out.logits, verbalizer, attention_mask=b_attn)
            base_correct.append((preds == base_label).sum().item())
            base_total.append(b_ids.size(0))

    clean_diffs_t = torch.cat(clean_diffs)         # [N]
    corrupted_diffs_t = torch.cat(corrupted_diffs) # [N]
    clean_mean = float(clean_diffs_t.mean().item())
    corrupted_mean = float(corrupted_diffs_t.mean().item())
    base_acc = sum(base_correct) / max(1, sum(base_total))

    # ---- main sweep -----------------------------------------------------
    grid = np.zeros((len(layers), len(positions)), dtype=np.float64)
    layer_iter = tqdm(layers, desc="layers", disable=not progress)

    for li, layer in enumerate(layer_iter):
        intervenable = _build_intervenable(model, layer=layer, component=component)
        for pi, pos in enumerate(positions):
            patched_diff_chunks: List[torch.Tensor] = []
            patched_pred_chunks: List[torch.Tensor] = []
            cf_label_chunks: List[torch.Tensor] = []

            for batch in loader:
                b_ids = batch["base_input_ids"].to(device)
                s_ids = batch["source_input_ids"].to(device)
                b_attn = batch["base_attention_mask"].to(device)
                s_attn = batch["source_attention_mask"].to(device)
                cf_label = batch["counterfactual_label_id"].to(device)

                _, cf_logits = run_single_patch(
                    model,
                    layer=layer,
                    position=pos,
                    base_input_ids=b_ids,
                    source_input_ids=s_ids,
                    base_attention_mask=b_attn,
                    source_attention_mask=s_attn,
                    component=component,
                    intervenable=intervenable,
                )

                diff = label_logit_diff(
                    cf_logits, cf_label, verbalizer, attention_mask=b_attn,
                    reduction="none",
                )
                preds = decode_label(cf_logits, verbalizer, attention_mask=b_attn)

                patched_diff_chunks.append(diff.detach().cpu())
                patched_pred_chunks.append(preds.detach().cpu())
                cf_label_chunks.append(cf_label.detach().cpu())

            patched = torch.cat(patched_diff_chunks)
            preds_all = torch.cat(patched_pred_chunks)
            gold_all = torch.cat(cf_label_chunks)

            if metric == "logit_recovery":
                # We use *mean* clean / corrupted as denominators so the
                # recovery metric is comparable across the grid.
                grid[li, pi] = float(
                    logit_recovery(
                        patched_diff=float(patched.mean().item()),
                        clean_diff=clean_mean,
                        corrupted_diff=corrupted_mean,
                    )
                )
            else:  # "iia"
                grid[li, pi] = compute_iia(preds_all, gold_all)

    return PatchingResult(
        metric_name=metric,
        grid=grid,
        layers=list(layers),
        positions=list(positions),
        position_labels=list(position_labels) if position_labels is not None else None,
        clean_logit_diff=clean_mean,
        corrupted_logit_diff=corrupted_mean,
        base_accuracy=base_acc,
        extras={"component": component},
    )


# ---------------------------------------------------------------------------
# Primary API: tidy-DataFrame sweep
# ---------------------------------------------------------------------------


_DEFAULT_VERBALIZER: Dict[str, str] = {
    "entailment": " yes",
    "neutral": " maybe",
    "contradiction": " no",
}


def _resolve_device(device: Union[str, torch.device, None]) -> torch.device:
    """Coerce ``device`` to a real :class:`torch.device`, falling back to CPU
    if CUDA was requested but is unavailable. This keeps the function usable
    on laptops without GPUs (the user's stated debugging requirement).
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "mps" and not getattr(torch.backends, "mps", None) is not None:
        return torch.device("cpu")
    return dev


def _resolve_verbalizer(
    tokenizer,
    verbalizer: Optional[LabelVerbalizer],
) -> LabelVerbalizer:
    if verbalizer is not None:
        return verbalizer
    return LabelVerbalizer.from_tokenizer(tokenizer, _DEFAULT_VERBALIZER)


def _position_token_strings(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    example_index: int = 0,
) -> List[str]:
    """Decode each position of example ``example_index`` to a short string,
    so plots can label the x-axis by token rather than by integer index.
    Padded positions are rendered as ``"<pad>"``.
    """
    ids = input_ids[example_index].tolist()
    mask = attention_mask[example_index].tolist()
    out: List[str] = []
    for tok_id, m in zip(ids, mask):
        if not m:
            out.append("<pad>")
            continue
        tok = tokenizer.decode([tok_id])
        # Normalise whitespace so seaborn axis labels don't break.
        tok = tok.replace("\n", "\\n").replace("\t", "\\t")
        if not tok.strip():
            tok = repr(tok)
        out.append(tok)
    return out


def run_patching_sweep(
    model: torch.nn.Module,
    tokenizer,
    examples: Sequence[CounterfactualExample],
    layers: Sequence[int],
    components: Sequence[str] = ("mlp_output", "attention_input", "block_output"),
    positions: Union[str, Sequence[int]] = "all",
    metric: str = "logit_recovery",
    device: Union[str, torch.device] = "cuda",
    *,
    verbalizer: Optional[LabelVerbalizer] = None,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Sweep activation patching over ``(layer, component, position)``.

    For each base/source NLI pair we

    1. tokenize base and source (padded to a common length so positions align),
    2. run the original model on base and source to obtain the *clean*
       logit-difference scores w.r.t. the counterfactual label,
    3. for every ``(layer, component, position)`` site, wrap the model in
       a fresh :class:`pyvene.IntervenableModel` with a single
       :class:`pyvene.VanillaIntervention` and patch source into base,
    4. record per-example logit-difference recovery *and* whether the
       patched argmax equals the high-level counterfactual label.

    The returned DataFrame is tidy / long-form, with one row per
    ``(example, layer, component, position)``. This is convenient for
    pandas groupby, seaborn / matplotlib plotting, and downstream analysis.

    Parameters
    ----------
    model:
        HuggingFace causal LM. Must already be on ``device`` (we move it
        again defensively, but only model parameters move; the user's
        original handle stays valid).
    tokenizer:
        HuggingFace tokenizer matching ``model``.
    examples:
        List of :class:`~src.data.counterfactual_pairs.CounterfactualExample`.
        Use :func:`~src.data.counterfactual_pairs.build_counterfactual_dataset`
        to construct these.
    layers:
        Transformer-block indices to sweep.
    components:
        pyvene component names to sweep. Defaults to the three most
        informative residual-stream sites for GPT-2 family models.
    positions:
        Either the string ``"all"`` (default; iterate every token position
        of the padded sequence) or an explicit sequence of token indices.
    metric:
        Either ``"logit_recovery"`` or ``"iia"``. Both quantities are
        computed *and* stored on every row regardless of this choice; the
        ``metric`` value is attached to ``df.attrs["metric"]`` so plotting
        helpers know which column to pivot on. (Kept as an explicit arg
        because the user's spec asks for it.)
    device:
        ``"cuda"`` / ``"cpu"`` / ``torch.device``. Falls back to CPU if
        CUDA is requested but unavailable.
    verbalizer:
        Optional :class:`LabelVerbalizer`. Defaults to ``{"entailment": " yes",
        "neutral": " maybe", "contradiction": " no"}``.
    batch_size:
        Inner DataLoader batch size.
    max_length:
        Optional max-length passed to the tokenizer.
    progress:
        If ``True``, show a tqdm bar over ``(layer, component)`` pairs.

    Returns
    -------
    :class:`pandas.DataFrame`
        Columns:
        ``example_id, layer, component, position,
        base_label, source_label, cf_label,
        base_score, source_score, patched_score,
        recovery, iia_correct``.

        Additional metadata is stored on ``df.attrs``:
        ``metric``, ``verbalizer``, ``position_labels`` (decoded token
        strings, one per position), ``base_accuracy`` (argmax accuracy
        of the unpatched base run).
    """
    if metric not in ("logit_recovery", "iia"):
        raise ValueError(f"Unknown metric {metric!r}; expected 'logit_recovery' or 'iia'")

    if len(examples) == 0:
        raise ValueError("`examples` is empty")

    device = _resolve_device(device)
    model.to(device)
    model.eval()

    verbalizer = _resolve_verbalizer(tokenizer, verbalizer)

    # ----- 1. tokenize -----------------------------------------------------
    base_prompts = [ex.base.prompt for ex in examples]
    source_prompts = [ex.source.prompt for ex in examples]
    enc = tokenizer(
        base_prompts + source_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    N = len(examples)
    base_ids = enc["input_ids"][:N].to(device)
    src_ids = enc["input_ids"][N:].to(device)
    base_attn = enc["attention_mask"][:N].to(device)
    src_attn = enc["attention_mask"][N:].to(device)
    T = int(base_ids.shape[1])

    if positions == "all":
        position_list: List[int] = list(range(T))
    else:
        position_list = list(positions)  # type: ignore[arg-type]
        for p in position_list:
            if not (0 <= int(p) < T):
                raise ValueError(
                    f"position {p} out of range [0, {T}) given the padded sequence length"
                )

    layers_list = list(layers)

    # Pre-build label tensors.
    base_lab = torch.tensor([ex.base_label_id for ex in examples], dtype=torch.long, device=device)
    src_lab = torch.tensor([ex.source_label_id for ex in examples], dtype=torch.long, device=device)
    cf_lab = torch.tensor(
        [ex.counterfactual_label_id for ex in examples], dtype=torch.long, device=device
    )

    # ----- 2. baselines (clean base run + clean source run) ----------------
    with torch.no_grad():
        # Per-example logit-diff with cf_label as the "correct" token.
        base_score_chunks: List[torch.Tensor] = []
        source_score_chunks: List[torch.Tensor] = []
        base_pred_chunks: List[torch.Tensor] = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b_out = model(
                input_ids=base_ids[start:end],
                attention_mask=base_attn[start:end],
            )
            s_out = model(
                input_ids=src_ids[start:end],
                attention_mask=src_attn[start:end],
            )
            base_score_chunks.append(
                label_logit_diff(
                    b_out.logits,
                    cf_lab[start:end],
                    verbalizer,
                    attention_mask=base_attn[start:end],
                    reduction="none",
                ).detach().cpu()
            )
            source_score_chunks.append(
                label_logit_diff(
                    s_out.logits,
                    cf_lab[start:end],
                    verbalizer,
                    attention_mask=src_attn[start:end],
                    reduction="none",
                ).detach().cpu()
            )
            base_pred_chunks.append(
                decode_label(
                    b_out.logits, verbalizer, attention_mask=base_attn[start:end]
                ).detach().cpu()
            )

    base_score = torch.cat(base_score_chunks)         # [N]
    source_score = torch.cat(source_score_chunks)     # [N]
    base_pred = torch.cat(base_pred_chunks)           # [N]
    base_lab_cpu = base_lab.detach().cpu()
    cf_lab_cpu = cf_lab.detach().cpu()
    base_accuracy = float((base_pred == base_lab_cpu).float().mean().item())

    # ----- 3. sweep --------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    total = len(layers_list) * len(components) * len(position_list)
    pbar = tqdm(
        total=len(layers_list) * len(components),
        desc="layer x component",
        disable=not progress,
    )

    eps = 1e-8

    for layer in layers_list:
        for component in components:
            # Build one IntervenableModel per (layer, component) and reuse
            # it across all positions to amortise hook registration.
            config = pv.IntervenableConfig(
                model_type=type(model),
                representations=[
                    pv.RepresentationConfig(layer, component, "pos", 1),
                ],
                intervention_types=pv.VanillaIntervention,
            )
            intervenable = pv.IntervenableModel(config, model)
            intervenable.set_device(device)
            intervenable.disable_model_gradients()

            for pos in position_list:
                patched_score_chunks: List[torch.Tensor] = []
                patched_pred_chunks: List[torch.Tensor] = []

                with torch.no_grad():
                    for start in range(0, N, batch_size):
                        end = min(start + batch_size, N)
                        base_inputs = {
                            "input_ids": base_ids[start:end],
                            "attention_mask": base_attn[start:end],
                        }
                        source_inputs = {
                            "input_ids": src_ids[start:end],
                            "attention_mask": src_attn[start:end],
                        }
                        _, cf_out = intervenable(
                            base_inputs,
                            [source_inputs],
                            {"sources->base": int(pos)},
                        )

                        patched_score_chunks.append(
                            label_logit_diff(
                                cf_out.logits,
                                cf_lab[start:end],
                                verbalizer,
                                attention_mask=base_attn[start:end],
                                reduction="none",
                            ).detach().cpu()
                        )
                        patched_pred_chunks.append(
                            decode_label(
                                cf_out.logits,
                                verbalizer,
                                attention_mask=base_attn[start:end],
                            ).detach().cpu()
                        )

                patched_score = torch.cat(patched_score_chunks)   # [N]
                patched_pred = torch.cat(patched_pred_chunks)     # [N]

                # recovery = fraction of the (source - base) gap that the
                # patch closes, when scored against cf_label.
                denom = source_score - base_score
                # Avoid division by zero per-example; rows where |denom| < eps
                # get NaN recovery (more honest than a huge number).
                recovery = torch.where(
                    denom.abs() < eps,
                    torch.full_like(denom, float("nan")),
                    (patched_score - base_score) / denom,
                )
                iia_correct = (patched_pred == cf_lab_cpu).to(torch.int64)

                for i, ex in enumerate(examples):
                    rows.append(
                        {
                            "example_id": i,
                            "layer": int(layer),
                            "component": component,
                            "position": int(pos),
                            "base_label": ID2LABEL[int(base_lab_cpu[i].item())],
                            "source_label": ID2LABEL[int(src_lab[i].item())],
                            "cf_label": ID2LABEL[int(cf_lab_cpu[i].item())],
                            "base_score": float(base_score[i].item()),
                            "source_score": float(source_score[i].item()),
                            "patched_score": float(patched_score[i].item()),
                            "recovery": float(recovery[i].item()),
                            "iia_correct": int(iia_correct[i].item()),
                        }
                    )

            pbar.update(1)

            # Help GC drop the hooks held by the wrapper before we register
            # new ones for the next (layer, component).
            del intervenable

    pbar.close()

    df = pd.DataFrame(rows)
    df.attrs["metric"] = metric
    df.attrs["verbalizer"] = dict(verbalizer.label_to_string)
    df.attrs["base_accuracy"] = base_accuracy
    df.attrs["position_labels"] = _position_token_strings(
        tokenizer, base_ids.detach().cpu(), base_attn.detach().cpu(), example_index=0
    )
    df.attrs["sequence_length"] = T
    return df
