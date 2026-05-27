from __future__ import annotations

# ===========================================================================
# Logit-difference + verbalizer helpers (from src/metrics/logits.py)
# ===========================================================================
"""Logit-difference and logit-recovery metrics.

For a 3-way NLI verbalizer (e.g. " yes" / " maybe" / " no") we read the
next-token logits at the final position of the prompt and reduce them to:

- ``label_logit_diff``: ``logit[correct_label] - logit[wrong_label]``
  averaged across distractors. This is the workhorse metric of
  Wang et al. 2022 ("Interpretability in the Wild") and the activation
  patching literature.
- ``logit_recovery``: how much of the *clean* logit-diff is recovered by
  a patched run, normalised to ``[0, 1]``:

    recovery = (LD_patched - LD_corrupted) / (LD_clean - LD_corrupted)

  recovery = 1 means the patch fully recovers the clean behaviour;
  recovery = 0 means it does nothing; values outside [0, 1] are possible
  (and informative) when patches over- or under-shoot.
"""


from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------


@dataclass
class LabelVerbalizer:
    """Maps NLI labels to single-token ids in a given tokenizer.

    Examples
    --------
    >>> v = LabelVerbalizer.from_tokenizer(tok,
    ...     {"entailment": " yes", "neutral": " maybe", "contradiction": " no"})
    >>> v.token_ids
    {'entailment': 3763, 'neutral': 14373, 'contradiction': 645}
    """

    label_to_string: Dict[str, str]
    token_ids: Dict[str, int]

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        label_to_string: Dict[str, str],
    ) -> "LabelVerbalizer":
        token_ids: Dict[str, int] = {}
        for label, surface in label_to_string.items():
            ids = tokenizer(surface, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(
                    f"Verbalizer string {surface!r} for label {label!r} "
                    f"tokenizes to {len(ids)} tokens ({ids!r}); it must "
                    f"map to a single token. Try a different surface form."
                )
            token_ids[label] = int(ids[0])
        return cls(label_to_string=dict(label_to_string), token_ids=token_ids)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self.label_to_string.keys())

    def token_id_tensor(self, device=None) -> torch.Tensor:
        """Return token ids in canonical label order as a LongTensor."""
        ids = [self.token_ids[l] for l in self.labels]
        t = torch.tensor(ids, dtype=torch.long)
        return t.to(device) if device is not None else t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _final_logits(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return logits at the last non-pad position. Shape ``[B, V]``."""
    if logits.dim() != 3:
        raise ValueError(f"Expected logits of shape [B, T, V], got {tuple(logits.shape)}")
    if attention_mask is None:
        return logits[:, -1, :]
    # Last index where attention_mask == 1.
    last_idx = attention_mask.long().sum(dim=-1) - 1
    last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_idx, last_idx, :]


def decode_label(
    logits: torch.Tensor,
    verbalizer: LabelVerbalizer,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Argmax-decode an NLI label from full LM logits.

    Restricts the argmax to the verbalizer tokens, so the prediction is
    always a valid label id (entailment=0 / neutral=1 / contradiction=2,
    matching :data:`src.data.causal_model.LABEL2ID`).

    Returns
    -------
    LongTensor of shape ``[B]`` with label ids in the verbalizer's order.
    """
    final = _final_logits(logits, attention_mask)
    ids = verbalizer.token_id_tensor(device=final.device)  # [L]
    restricted = final[:, ids]  # [B, L]
    return restricted.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Logit difference
# ---------------------------------------------------------------------------


def label_logit_diff(
    logits: torch.Tensor,
    correct_label_ids: torch.Tensor,
    verbalizer: LabelVerbalizer,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Logit difference between correct and average-wrong label tokens.

    For each example ``i``,

        diff_i = logit[i, correct_i] - mean_{j != correct_i} logit[i, j]

    Parameters
    ----------
    logits:
        Full LM logits, shape ``[B, T, V]``.
    correct_label_ids:
        LongTensor of shape ``[B]`` with the gold label id in the
        *verbalizer* index space (0..len(labels)-1), matching the order
        of :attr:`LabelVerbalizer.labels`.
    verbalizer:
        :class:`LabelVerbalizer` for converting labels to token ids.
    attention_mask:
        Optional attention mask used to find the final non-pad position.
    reduction:
        ``"none"`` returns the per-example tensor;
        ``"mean"`` (default) returns a scalar tensor.

    Returns
    -------
    Tensor of shape ``[B]`` (reduction="none") or scalar (reduction="mean").
    """
    final = _final_logits(logits, attention_mask)  # [B, V]
    verb_ids = verbalizer.token_id_tensor(device=final.device)  # [L]
    restricted = final[:, verb_ids]  # [B, L]

    B, L = restricted.shape
    correct = correct_label_ids.to(restricted.device).long()
    if correct.shape != (B,):
        raise ValueError(
            f"correct_label_ids must have shape [B={B}], got {tuple(correct.shape)}"
        )

    correct_logit = restricted.gather(1, correct.unsqueeze(1)).squeeze(1)  # [B]

    # Mask out the correct entry, average the rest.
    mask = torch.ones_like(restricted, dtype=torch.bool)
    mask.scatter_(1, correct.unsqueeze(1), False)
    wrong_logits = restricted.masked_select(mask).view(B, L - 1)
    wrong_avg = wrong_logits.mean(dim=-1)

    diff = correct_logit - wrong_avg

    if reduction == "none":
        return diff
    if reduction == "mean":
        return diff.mean()
    raise ValueError(f"Unknown reduction {reduction!r}")


# ---------------------------------------------------------------------------
# Logit recovery
# ---------------------------------------------------------------------------


def logit_recovery(
    patched_diff: Union[torch.Tensor, float, np.ndarray],
    clean_diff: Union[torch.Tensor, float, np.ndarray],
    corrupted_diff: Union[torch.Tensor, float, np.ndarray],
    eps: float = 1e-8,
) -> Union[torch.Tensor, float]:
    """Normalised logit-difference recovery.

    .. math::
        \\mathrm{recovery}
            = \\frac{\\mathrm{LD}_{\\mathrm{patched}}
                    - \\mathrm{LD}_{\\mathrm{corrupted}}}
                   {\\mathrm{LD}_{\\mathrm{clean}}
                    - \\mathrm{LD}_{\\mathrm{corrupted}} + \\epsilon}

    A value of 1.0 means the patch fully restored the clean-run logit
    difference; 0.0 means it did nothing. The metric can fall outside
    ``[0, 1]`` (e.g. if the patch over-shoots, or if the model was
    *more* confident in the corrupted run than the clean run).

    Inputs can be scalars, NumPy arrays, or torch Tensors and are
    broadcast together. The return type matches the inputs (Tensor if
    any input is a Tensor).
    """

    def _is_tensor(x):
        return isinstance(x, torch.Tensor)

    if any(_is_tensor(x) for x in (patched_diff, clean_diff, corrupted_diff)):
        # Promote everything to tensors.
        def _t(x):
            if _is_tensor(x):
                return x
            return torch.as_tensor(x)

        p = _t(patched_diff)
        c = _t(clean_diff)
        cor = _t(corrupted_diff)
        return (p - cor) / (c - cor + eps)

    p = np.asarray(patched_diff, dtype=np.float64)
    c = np.asarray(clean_diff, dtype=np.float64)
    cor = np.asarray(corrupted_diff, dtype=np.float64)
    out = (p - cor) / (c - cor + eps)
    if out.ndim == 0:
        return float(out)
    return out


# ===========================================================================
# IIA metrics (from src/metrics/iia.py)
# ===========================================================================
"""Interchange Intervention Accuracy (IIA).

IIA is the main metric advocated by the DAS / causal-abstraction line of
work (Geiger et al. 2021, 2023). Given a batch of interchange
interventions, it asks: *after patching the relevant low-level
representation from source into base, does the model's prediction match
the gold counterfactual label produced by the symbolic high-level model?*

If IIA is high, the low-level neural network is implementing the
high-level algorithm (at least for the chosen alignment). IIA == 1.0
means perfect causal abstraction; chance-level IIA means the alignment
explains nothing.
"""


from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch


ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[int]]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_iia(
    preds: ArrayLike,
    gold_labels: ArrayLike,
    *,
    mask: Optional[ArrayLike] = None,
) -> float:
    """Compute interchange intervention accuracy.

    Parameters
    ----------
    preds:
        Model predictions after the intervention. Shape ``[N]`` of int
        label ids.
    gold_labels:
        Gold counterfactual labels from the high-level causal model.
        Shape ``[N]`` of int label ids.
    mask:
        Optional boolean mask of shape ``[N]``. If given, only entries
        where the mask is true contribute to the metric. Useful to
        exclude examples whose base prediction was already wrong.

    Returns
    -------
    float
        IIA in ``[0, 1]``. Returns 0.0 if there are no valid examples.
    """
    preds = _to_numpy(preds).astype(np.int64).ravel()
    gold = _to_numpy(gold_labels).astype(np.int64).ravel()
    if preds.shape != gold.shape:
        raise ValueError(
            f"preds and gold_labels must have the same shape, "
            f"got {preds.shape} vs {gold.shape}"
        )

    if mask is not None:
        m = _to_numpy(mask).astype(bool).ravel()
        if m.shape != preds.shape:
            raise ValueError("mask must have the same shape as preds")
        preds = preds[m]
        gold = gold[m]

    if preds.size == 0:
        return 0.0
    return float((preds == gold).mean())


def compute_iia_per_class(
    preds: ArrayLike,
    gold_labels: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Per-class IIA.

    Returns a dict mapping class name (or stringified id) to its accuracy.
    Classes with zero gold occurrences map to ``float('nan')``.
    """
    preds = _to_numpy(preds).astype(np.int64).ravel()
    gold = _to_numpy(gold_labels).astype(np.int64).ravel()
    if preds.shape != gold.shape:
        raise ValueError("preds and gold_labels must have the same shape")

    classes = np.unique(gold)
    out: Dict[str, float] = {}
    for c in classes:
        m = gold == c
        if m.sum() == 0:
            acc = float("nan")
        else:
            acc = float((preds[m] == gold[m]).mean())
        name = class_names[int(c)] if class_names is not None else str(int(c))
        out[name] = acc
    return out


# ===========================================================================
# DAS IntervenableConfig builder (from src/interventions/das_config.py)
# ===========================================================================
"""Build a Distributed Alignment Search (DAS) ``IntervenableConfig``.

This is the minimal "learned rotation" intervention used by DAS
(Geiger et al. 2023). Compared to the vanilla activation patching
baseline in :mod:`src.interventions.patching`, here we learn an
*orthogonal rotation* of the hidden-state subspace at a single
``(layer, component, position)`` site; the rotated subspace is what
gets swapped between source and base, so training pressure is what
finds the dimensions of the hidden state that causally implement the
target high-level variable.

We default to :class:`pyvene.LowRankRotatedSpaceIntervention` because
its API is the cleanest fit for "patch a ``d``-dim subspace at one
token":

- the rotation matrix is a single ``[hidden_size, low_rank_dimension]``
  parameter (a slice of an orthogonal matrix),
- *all* of the rotated subspace is swapped (no extra
  ``subspace_partition`` plumbing),
- the only trainable parameter is the rotation itself.

If a particular pyvene build doesn't expose ``LowRankRotatedSpaceIntervention``
we fall back to the full-rank :class:`pyvene.RotatedSpaceIntervention`
with a binary subspace partition ``[[0, d], [d, hidden]]`` and
intervene on subspace ``0``. The two formulations are mathematically
equivalent for our use case.
"""


from typing import Optional

import pyvene as pv


def _get_hidden_size(model) -> int:
    """Best-effort hidden size lookup across HF causal LM configs."""
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        if cfg is not None and hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError("Could not infer hidden size from model.config")


def _pick_intervention_class():
    """Return ``(cls, uses_low_rank_dimension_kwarg)``.

    We prefer :class:`LowRankRotatedSpaceIntervention` if pyvene exposes
    it; otherwise we fall back to the full-rank rotation.
    """
    if hasattr(pv, "LowRankRotatedSpaceIntervention"):
        return pv.LowRankRotatedSpaceIntervention, True
    if hasattr(pv, "RotatedSpaceIntervention"):
        return pv.RotatedSpaceIntervention, False
    raise ImportError(
        "Neither LowRankRotatedSpaceIntervention nor RotatedSpaceIntervention "
        "is available in your installed pyvene. Please upgrade pyvene."
    )


def make_das_config(
    model,
    layer: int,
    component: str = "block_output",
    intervention_dim: int = 4,
    unit: str = "pos",
    *,
    max_number_of_units: int = 1,
    force_full_rank: bool = False,
) -> pv.IntervenableConfig:
    """Build an :class:`IntervenableConfig` for DAS.

    Parameters
    ----------
    model:
        The HuggingFace causal LM that the intervention will be attached
        to. We need it only to infer ``hidden_size`` and ``type(model)``.
    layer:
        Transformer-block index at which to intervene.
    component:
        pyvene component name (default ``"block_output"`` -- the
        post-block residual stream, which is the canonical DAS site).
    intervention_dim:
        Dimensionality of the rotated subspace that DAS will swap. For
        our lexical-NLI task we have 4 lexical relations, so 4 is a
        sensible default; in general pick the smallest ``d`` that
        seems to work.
    unit:
        pyvene unit type. ``"pos"`` (default) addresses a single token
        position; ``"h"`` would address an attention head.
    max_number_of_units:
        Number of units swapped per intervention (default 1 -- patch a
        single token).
    force_full_rank:
        If True, use :class:`RotatedSpaceIntervention` with a binary
        subspace partition even if the low-rank variant is available.
        Useful when comparing implementations.

    Returns
    -------
    :class:`pyvene.IntervenableConfig`
        Configuration ready to be passed to
        :class:`pyvene.IntervenableModel`.

    Notes
    -----
    The trainable parameters created by pyvene live under
    ``intervenable.interventions[k].rotate_layer`` for every key ``k``.
    :func:`src.interventions.train_das.train_das_alignment` knows how to
    pull them out for the optimizer.
    """
    hidden = _get_hidden_size(model)
    if intervention_dim < 1 or intervention_dim > hidden:
        raise ValueError(
            f"intervention_dim={intervention_dim} must be in [1, hidden_size={hidden}]"
        )

    intervention_cls, low_rank_supported = _pick_intervention_class()
    if force_full_rank:
        if not hasattr(pv, "RotatedSpaceIntervention"):
            raise ImportError("RotatedSpaceIntervention not available in this pyvene build")
        intervention_cls = pv.RotatedSpaceIntervention
        low_rank_supported = False

    if low_rank_supported:
        rep = pv.RepresentationConfig(
            layer,
            component,
            unit,
            max_number_of_units,
            low_rank_dimension=intervention_dim,
        )
    else:
        # Full-rank rotation: split into a "target" subspace of size
        # ``intervention_dim`` and a "rest" subspace, and intervene on
        # subspace 0 (the target).
        rep = pv.RepresentationConfig(
            layer,
            component,
            unit,
            max_number_of_units,
            subspace_partition=[[0, intervention_dim], [intervention_dim, hidden]],
        )

    config = pv.IntervenableConfig(
        model_type=type(model),
        representations=[rep],
        intervention_types=intervention_cls,
    )
    # Stash some bookkeeping so downstream code can introspect without
    # having to re-derive everything.
    config.__dict__["_das_meta"] = {
        "layer": int(layer),
        "component": str(component),
        "intervention_dim": int(intervention_dim),
        "unit": str(unit),
        "max_number_of_units": int(max_number_of_units),
        "hidden_size": int(hidden),
        "intervention_class": intervention_cls.__name__,
        "low_rank_supported": bool(low_rank_supported),
    }
    return config


def das_config_meta(config: pv.IntervenableConfig) -> dict:
    """Return the bookkeeping dict :func:`make_das_config` attached to ``config``.

    Returns an empty dict if ``config`` wasn't built by ``make_das_config``.
    """
    return dict(config.__dict__.get("_das_meta", {}))


# ===========================================================================
# DAS training loop (from src/interventions/train_das.py)
# ===========================================================================
"""Train a DAS (Distributed Alignment Search) rotation on counterfactual NLI data.

The training loop is intentionally minimal: we freeze the base LM and
learn *only* the orthogonal rotation parameters of the
:class:`pyvene.LowRankRotatedSpaceIntervention` (or
:class:`RotatedSpaceIntervention`) at the chosen site, with a vanilla
cross-entropy objective on the next-token logits at the answer
position. The "label" is the verbalizer token id corresponding to the
**high-level counterfactual label** computed by
:mod:`src.data.causal_model`.

Training signal in plain English
--------------------------------
For each batch we

1. take a base sequence (the prompt we want to intervene on),
2. take a source sequence (the prompt whose value of the target
   variable we want to import),
3. forward-pass both through the model, *swap* the rotated subspace at
   the configured site from source into base,
4. read the final-position logits, restrict to verbalizer tokens, and
   minimise CE against the gold counterfactual token.

If the rotated subspace really does encode the target variable, this
loss is minimisable; if it doesn't (wrong site, too-small ``d``, etc.),
the loss plateaus near chance and IIA stays at chance.

The function returns the wrapped :class:`pyvene.IntervenableModel` and
a list of ``{"epoch", "loss", "train_iia"}`` records. If ``log_path``
is given, the history is also dumped to JSON for later plotting.
"""


import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pyvene as pv

# LabelVerbalizer, _final_logits, decode_label, make_das_config,
# das_config_meta are all defined above in this same file (merged from
# src/metrics/logits.py and src/interventions/das_config.py).
from .patching import _resolve_device, _resolve_verbalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_das_device(device: Union[str, torch.device, None]) -> torch.device:
    """Resolve the DAS device, avoiding Apple MPS for rotated interventions.

    pyvene's rotated-space interventions call ``torch.linalg.householder_product``
    through the rotation layer. That operator is not implemented on MPS as of
    current PyTorch releases, so DAS must run on CUDA or CPU. Activation
    patching can still use MPS; only learned rotated interventions need this
    guard.
    """
    resolved = _resolve_device(device)
    if resolved.type == "mps":
        warnings.warn(
            "DAS rotated interventions require an operator that is not "
            "implemented on Apple MPS. Falling back to CPU for DAS training.",
            RuntimeWarning,
        )
        return torch.device("cpu")
    return resolved


def _format_unit_locations(positions: torch.Tensor) -> Union[int, List[int]]:
    """Convert a per-example LongTensor ``[B]`` of positions to the format
    accepted by :meth:`pyvene.IntervenableModel.forward`.

    - If all elements are equal, return a single int (most efficient).
    - Otherwise return a Python list of ints; pyvene broadcasts each
      element to the corresponding base example.
    """
    if not isinstance(positions, torch.Tensor):
        positions = torch.as_tensor(positions, dtype=torch.long)
    pl = positions.tolist()
    if len(pl) == 0:
        return 0
    if all(p == pl[0] for p in pl):
        return int(pl[0])
    return [int(p) for p in pl]


def infer_fixed_position(dataset, fixed_position: Optional[int] = None) -> int:
    """Return the single token position DAS should use for a dataset.

    DAS in this project is a *fixed-site* alignment: one layer, one
    component, one token position. pyvene treats a Python list of positions as
    "intervene on all of these positions" for each example, not "one position
    per example". Passing per-example lists can therefore multiply the hidden
    dimension (e.g. ``8 positions * 768 hidden = 6144``) and break low-rank
    rotations. We instead use one fixed integer position for the whole run.

    If ``fixed_position`` is not provided, we use the mode of the dataset's
    ``intervention_pos`` values and warn if positions are not uniform.
    """
    if fixed_position is not None:
        return int(fixed_position)

    positions: List[int] = []
    if hasattr(dataset, "examples"):
        for ex in dataset.examples:
            if getattr(ex, "intervention_pos", None) is not None:
                positions.append(int(ex.intervention_pos))

    if not positions:
        for i in range(len(dataset)):
            item = dataset[i]
            pos = item.get("intervention_pos", 0)
            if isinstance(pos, torch.Tensor):
                pos = int(pos.item())
            positions.append(int(pos))

    if not positions:
        return 0

    counts = Counter(positions)
    mode_pos, mode_count = counts.most_common(1)[0]
    if len(counts) > 1:
        warnings.warn(
            "DAS is a fixed-position intervention, but the dataset contains "
            f"multiple intervention positions {dict(counts)}. Using the most "
            f"common position {mode_pos} ({mode_count}/{len(positions)} examples). "
            "For stricter experiments, use a single template or pass "
            "`fixed_position=` explicitly.",
            RuntimeWarning,
        )
    return int(mode_pos)


def _collect_optim_params(intervenable: pv.IntervenableModel) -> List[Dict[str, Any]]:
    """Return optimizer param groups holding only intervention weights.

    We rely on pyvene's :meth:`get_trainable_parameters` when available;
    otherwise we hand-walk ``intervenable.interventions`` and pull each
    rotation's ``rotate_layer.parameters()`` (the canonical DAS rotation
    object). This matches the pattern in the pyvene Boundless-DAS tutorial.
    """
    trainable = list(intervenable.get_trainable_parameters())
    if trainable:
        return [{"params": trainable}]

    groups: List[Dict[str, Any]] = []
    for _, v in intervenable.interventions.items():
        # rotate_layer is the orthogonal-matrix wrapper used by
        # RotatedSpaceIntervention and LowRankRotatedSpaceIntervention.
        if hasattr(v, "rotate_layer"):
            groups.append({"params": list(v.rotate_layer.parameters())})
        else:
            groups.append({"params": list(v.parameters())})
    return groups


# ---------------------------------------------------------------------------
# Evaluation helper (used internally for val-loss tracking)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_loss_iia(
    intervenable: pv.IntervenableModel,
    dataset,
    *,
    verbalizer: LabelVerbalizer,
    verb_ids: torch.Tensor,
    device: torch.device,
    fixed_position: int,
    batch_size: int,
) -> Tuple[float, float]:
    """Compute mean CE loss and IIA on a held-out CF dataset.

    Returns ``(val_loss, val_iia)`` over the entire dataset, using the
    same intervention site as training. Does not modify the intervenable
    (intervenable is set back to its caller-side mode after running).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_n = 0
    correct = 0

    for batch in loader:
        base_inputs = {
            "input_ids": batch["base_input_ids"].to(device),
            "attention_mask": batch["base_attention_mask"].to(device),
        }
        source_inputs = {
            "input_ids": batch["source_input_ids"].to(device),
            "attention_mask": batch["source_attention_mask"].to(device),
        }
        cf_labels = batch["counterfactual_label_id"].to(device)

        _, cf_out = intervenable(
            base_inputs,
            [source_inputs],
            {"sources->base": int(fixed_position)},
        )
        final_logits = _final_logits(cf_out.logits, base_inputs["attention_mask"])
        target_token_ids = verb_ids[cf_labels]
        loss = F.cross_entropy(final_logits, target_token_ids, reduction="sum")
        total_loss += float(loss.item())
        total_n += int(cf_labels.size(0))

        preds = decode_label(
            cf_out.logits, verbalizer,
            attention_mask=base_inputs["attention_mask"],
        )
        correct += int((preds == cf_labels).sum().item())

    mean_loss = total_loss / max(1, total_n)
    iia = correct / max(1, total_n)
    return mean_loss, iia


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class DASTrainOutput:
    """Lightweight container returned by :func:`train_das_alignment`."""

    intervenable: pv.IntervenableModel
    history: List[Dict[str, float]]
    meta: Dict[str, Any]


def train_das_alignment(
    model,
    tokenizer,
    train_cf_dataset,
    config: pv.IntervenableConfig,
    num_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: Union[str, torch.device, None] = None,
    *,
    verbalizer: Optional[LabelVerbalizer] = None,
    log_path: Optional[str] = None,
    weight_decay: float = 0.0,
    progress: bool = True,
    fixed_position: Optional[int] = None,
    eval_cf_dataset: Optional[Any] = None,
) -> DASTrainOutput:
    """Train a DAS rotation on a counterfactual NLI dataset.

    Parameters
    ----------
    model:
        HuggingFace causal LM. The base parameters are frozen
        (``disable_model_gradients``); only the intervention's rotation
        matrix gets gradient updates.
    tokenizer:
        Tokenizer matching ``model``. Used only to build a default
        :class:`LabelVerbalizer` if ``verbalizer`` is not supplied.
    train_cf_dataset:
        A :class:`~src.data.counterfactual_pairs.CounterfactualDataset`,
        or any Dataset whose items expose the same keys
        (``base_input_ids``, ``source_input_ids``,
        ``base_attention_mask``, ``source_attention_mask``,
        ``counterfactual_label_id``, ``intervention_pos``).
    config:
        :class:`pyvene.IntervenableConfig` -- typically built by
        :func:`src.interventions.das_config.make_das_config`.
    num_epochs, lr, batch_size:
        Standard optimiser settings.
    device:
        Target device. Falls back to CPU if CUDA is requested but
        unavailable.
    verbalizer:
        Optional :class:`LabelVerbalizer`. Defaults to ``" yes" / " maybe"
        / " no"`` constructed from ``tokenizer``.
    log_path:
        If given, the training history is dumped to JSON at this path
        once training finishes. The parent directory is created if
        needed.
    weight_decay:
        AdamW weight decay (default 0.0 -- rotations don't usually want
        decay).
    progress:
        Show tqdm progress bars if True.
    fixed_position:
        Token position to intervene on for every example. DAS here is a
        fixed-site alignment, so this should usually be copied from the best
        activation-patching heatmap cell. If ``None``, we use the mode of
        ``train_cf_dataset``'s ``intervention_pos`` values.
    eval_cf_dataset:
        Optional held-out :class:`CounterfactualDataset`. If provided, we
        compute val loss and val IIA at the end of each epoch and record
        them in ``history``. Highly recommended for small datasets where
        overfitting is fast.

    Returns
    -------
    :class:`DASTrainOutput`
        ``intervenable`` (trained), ``history`` (list of
        ``{"epoch", "loss", "train_iia", "val_loss", "val_iia"}``
        records; ``val_*`` only present if ``eval_cf_dataset`` given),
        and ``meta`` (a dict of config + training hyperparameters,
        useful for logging).
    """
    device = _resolve_das_device(device)
    model.to(device)
    model.eval()  # we never train the base LM

    verbalizer = _resolve_verbalizer(tokenizer, verbalizer)
    fixed_position = infer_fixed_position(train_cf_dataset, fixed_position)

    intervenable = pv.IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()

    param_groups = _collect_optim_params(intervenable)
    if not param_groups:
        raise RuntimeError(
            "No trainable intervention parameters found. Did you pass a config "
            "for a learnable intervention (e.g. RotatedSpaceIntervention)?"
        )

    optimizer = torch.optim.Adam(
        [
            {"params": pg["params"], "lr": pg.get("lr", lr),
             "weight_decay": pg.get("weight_decay", weight_decay)}
            for pg in param_groups
        ]
    )

    # Cache the verbalizer token ids once.
    verb_ids = verbalizer.token_id_tensor(device=device)  # [num_labels]

    loader = DataLoader(train_cf_dataset, batch_size=batch_size, shuffle=True)

    history: List[Dict[str, float]] = []
    epoch_iter = range(num_epochs)
    if progress:
        epoch_iter = tqdm(epoch_iter, desc="epochs")

    for epoch in epoch_iter:
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        epoch_correct = 0
        epoch_total = 0

        step_iter = loader
        if progress:
            step_iter = tqdm(
                loader, desc=f"epoch {epoch}", leave=False, position=1
            )

        for batch in step_iter:
            base_inputs = {
                "input_ids": batch["base_input_ids"].to(device),
                "attention_mask": batch["base_attention_mask"].to(device),
            }
            source_inputs = {
                "input_ids": batch["source_input_ids"].to(device),
                "attention_mask": batch["source_attention_mask"].to(device),
            }
            cf_labels = batch["counterfactual_label_id"].to(device)        # [B]
            pos_arg = int(fixed_position)

            _, cf_out = intervenable(
                base_inputs,
                [source_inputs],
                {"sources->base": pos_arg},
            )

            # Cross-entropy on the full vocabulary, with the gold
            # verbalizer-token id as the target. Using full-vocab CE
            # (rather than CE restricted to the 3 verbalizer tokens)
            # keeps gradient signal on the rest of the vocab, which
            # tends to be more stable and matches the Boundless-DAS
            # tutorial.
            final_logits = _final_logits(
                cf_out.logits, base_inputs["attention_mask"]
            )  # [B, V]
            target_token_ids = verb_ids[cf_labels]  # [B]
            loss = F.cross_entropy(final_logits, target_token_ids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = decode_label(
                    cf_out.logits, verbalizer,
                    attention_mask=base_inputs["attention_mask"],
                )
                epoch_correct += int((preds == cf_labels).sum().item())
                epoch_total += int(cf_labels.size(0))
                epoch_loss_sum += float(loss.item()) * int(cf_labels.size(0))
                epoch_loss_n += int(cf_labels.size(0))

            if progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "iia": f"{epoch_correct / max(1, epoch_total):.2f}",
                })

        avg_loss = epoch_loss_sum / max(1, epoch_loss_n)
        train_iia = epoch_correct / max(1, epoch_total)
        record: Dict[str, float] = {
            "epoch": int(epoch),
            "loss": float(avg_loss),
            "train_iia": float(train_iia),
        }

        if eval_cf_dataset is not None:
            val_loss, val_iia = _eval_loss_iia(
                intervenable,
                eval_cf_dataset,
                verbalizer=verbalizer,
                verb_ids=verb_ids,
                device=device,
                fixed_position=int(fixed_position),
                batch_size=batch_size,
            )
            record["val_loss"] = float(val_loss)
            record["val_iia"] = float(val_iia)

        history.append(record)

    meta = {
        **das_config_meta(config),
        "num_epochs": int(num_epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "fixed_position": int(fixed_position),
        "verbalizer": dict(verbalizer.label_to_string),
        "n_train_examples": int(len(train_cf_dataset)),
        "device": str(device),
    }

    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"history": history, "meta": meta}, f, indent=2)

    return DASTrainOutput(intervenable=intervenable, history=history, meta=meta)


# ===========================================================================
# DAS / IIA evaluation (from src/interventions/eval_iia.py)
# ===========================================================================
"""Evaluate a trained DAS intervenable on counterfactual NLI data.

Reports three things:

- ``factual_accuracy`` -- argmax accuracy of the *un-intervened* base
  run against the base example's gold (factual) NLI label. This is a
  sanity check: if the model can't even produce the base label
  unprompted, IIA numbers are hard to interpret.
- ``iia`` -- interchange-intervention accuracy: fraction of held-out
  pairs where the *patched* argmax matches the high-level counterfactual
  label predicted by the symbolic causal model.
- ``confusion`` -- 3x3 confusion matrix (``true_cf_label`` x
  ``pred_cf_label``) as a :class:`pandas.DataFrame`, useful for spotting
  asymmetric errors (e.g. "neutral collapses to contradiction").

Optionally, per-relation IIA is reported too -- handy because in our
lexical-NLI setup the four high-level relations (EQUIV / FORWARD /
REVERSE / DISJOINT) have systematically different difficulty.
"""


from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import pyvene as pv

from .causal_models import ID2LABEL
# iia helpers defined in this file
# logits helpers defined in this file
# train_das helpers defined in this file
from .patching import _resolve_device, _resolve_verbalizer


def _confusion_matrix(
    gold: np.ndarray,
    pred: np.ndarray,
    labels: list,
) -> pd.DataFrame:
    """Counts-only 3x3 confusion table (no scikit-learn dependency)."""
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for g, p in zip(gold.astype(np.int64), pred.astype(np.int64)):
        if 0 <= g < n and 0 <= p < n:
            cm[g, p] += 1
    return pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )


def _transition_matrix(
    base_labels: np.ndarray,
    cf_labels: np.ndarray,
    patched_preds: np.ndarray,
    labels: list,
) -> pd.DataFrame:
    """Per-(base_label -> cf_label) IIA cell.

    Cell ``[i, j]`` is the IIA over examples whose base label was ``i``
    and whose cf label was ``j``. Useful for spotting asymmetric failure
    modes (e.g. "the rotation flips entail->contradiction reliably but
    not entail->neutral"). Cells with no examples are ``NaN``.
    """
    n = len(labels)
    grid = np.full((n, n), np.nan, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)
    base = base_labels.astype(np.int64)
    cf = cf_labels.astype(np.int64)
    pred = patched_preds.astype(np.int64)
    for i in range(n):
        for j in range(n):
            m = (base == i) & (cf == j)
            if m.sum() == 0:
                continue
            grid[i, j] = float((pred[m] == cf[m]).mean())
            counts[i, j] = int(m.sum())
    df = pd.DataFrame(
        grid,
        index=[f"base_{l}" for l in labels],
        columns=[f"cf_{l}" for l in labels],
    )
    df.attrs["counts"] = pd.DataFrame(
        counts,
        index=[f"base_{l}" for l in labels],
        columns=[f"cf_{l}" for l in labels],
    )
    return df


def evaluate_das_iia(
    intervenable: pv.IntervenableModel,
    dataset,
    tokenizer,
    device: Union[str, torch.device, None] = None,
    *,
    verbalizer: Optional[LabelVerbalizer] = None,
    batch_size: int = 8,
    label_names: Optional[list] = None,
    fixed_position: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a (trained or untrained) DAS intervenable on a CF dataset.

    Parameters
    ----------
    intervenable:
        :class:`pyvene.IntervenableModel`. Both trained (DAS) and
        untrained (vanilla patching) intervenables work -- the function
        simply runs forward passes.
    dataset:
        :class:`~src.data.counterfactual_pairs.CounterfactualDataset` or
        compatible iterable of dicts.
    tokenizer:
        Used to build a default verbalizer if ``verbalizer`` is not
        supplied.
    device:
        Target device. Falls back to CPU when CUDA is unavailable.
    verbalizer:
        Optional :class:`LabelVerbalizer`.
    batch_size:
        Inner DataLoader batch size.
    label_names:
        Optional list of label-name strings in canonical order (length =
        number of labels). Defaults to ``verbalizer.labels``.
    fixed_position:
        Token position to intervene on for every example. If ``None``, use the
        mode of ``dataset``'s ``intervention_pos`` values. This must match the
        fixed position used during DAS training for a clean evaluation.

    Returns
    -------
    dict
        Keys:
        - ``factual_accuracy`` (float in [0, 1])
        - ``iia`` (float in [0, 1])
        - ``iia_per_class`` (dict label -> float)
        - ``confusion`` (:class:`pandas.DataFrame`, gold x pred counts)
        - ``n_examples`` (int)
        - ``base_preds`` / ``patched_preds`` / ``gold_cf_labels``
          (1-D numpy arrays of label ids, in dataset order; handy for
          downstream analysis and plotting).
    """
    device = _resolve_das_device(device)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    intervenable.model.eval()

    verbalizer = _resolve_verbalizer(tokenizer, verbalizer)
    if label_names is None:
        label_names = list(verbalizer.labels)
    fixed_position = infer_fixed_position(dataset, fixed_position)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    base_preds_all: list = []           # verbalizer-restricted argmax
    base_preds_unrestricted_all: list = []  # full-vocab argmax (compared to verb ids)
    base_labels_all: list = []
    patched_preds_all: list = []
    cf_labels_all: list = []

    verb_ids = verbalizer.token_id_tensor(device=device)  # [L]

    with torch.no_grad():
        for batch in loader:
            base_inputs = {
                "input_ids": batch["base_input_ids"].to(device),
                "attention_mask": batch["base_attention_mask"].to(device),
            }
            source_inputs = {
                "input_ids": batch["source_input_ids"].to(device),
                "attention_mask": batch["source_attention_mask"].to(device),
            }
            base_labels = batch["base_label_id"].to(device)
            cf_labels = batch["counterfactual_label_id"].to(device)
            pos_arg = int(fixed_position)

            # Factual: run the underlying (un-intervened) model directly.
            base_out = intervenable.model(**base_inputs)
            base_preds = decode_label(
                base_out.logits, verbalizer,
                attention_mask=base_inputs["attention_mask"],
            )
            # Unrestricted: take full-vocab argmax, check whether the top
            # token happens to be one of the verbalizer tokens AND matches
            # the gold label. This is a stricter test of factual ability:
            # is the model actually trying to emit " yes" / " no" / " maybe"
            # at the answer slot, or did the restricted decode just pick
            # the lesser of three near-zero logits?
            final = _final_logits(base_out.logits, base_inputs["attention_mask"])  # [B, V]
            top_tokens = final.argmax(dim=-1)  # [B]
            # For each example, find which verbalizer index (if any) the
            # top token corresponds to; -1 means "top token isn't a label
            # token at all".
            verb_match = (top_tokens.unsqueeze(1) == verb_ids.unsqueeze(0))  # [B, L]
            has_match = verb_match.any(dim=1)
            matched_label = verb_match.float().argmax(dim=1)  # [B]
            unrestricted_pred = torch.where(
                has_match, matched_label, torch.full_like(matched_label, -1)
            )

            # Counterfactual: run the intervened model.
            _, cf_out = intervenable(
                base_inputs,
                [source_inputs],
                {"sources->base": pos_arg},
            )
            patched_preds = decode_label(
                cf_out.logits, verbalizer,
                attention_mask=base_inputs["attention_mask"],
            )

            base_preds_all.append(base_preds.detach().cpu().numpy())
            base_preds_unrestricted_all.append(unrestricted_pred.detach().cpu().numpy())
            base_labels_all.append(base_labels.detach().cpu().numpy())
            patched_preds_all.append(patched_preds.detach().cpu().numpy())
            cf_labels_all.append(cf_labels.detach().cpu().numpy())

    base_preds_np = np.concatenate(base_preds_all) if base_preds_all else np.array([], dtype=np.int64)
    base_preds_unrestricted_np = (
        np.concatenate(base_preds_unrestricted_all)
        if base_preds_unrestricted_all
        else np.array([], dtype=np.int64)
    )
    base_labels_np = np.concatenate(base_labels_all) if base_labels_all else np.array([], dtype=np.int64)
    patched_preds_np = np.concatenate(patched_preds_all) if patched_preds_all else np.array([], dtype=np.int64)
    cf_labels_np = np.concatenate(cf_labels_all) if cf_labels_all else np.array([], dtype=np.int64)

    factual_acc = compute_iia(base_preds_np, base_labels_np)
    # Unrestricted: example counts only if (a) top token is a verbalizer
    # token AND (b) it matches the gold label. Examples whose top token is
    # something else (e.g. punctuation) get 0 credit.
    if base_labels_np.size:
        unrestricted_correct = (
            (base_preds_unrestricted_np == base_labels_np)
            & (base_preds_unrestricted_np >= 0)
        ).mean()
        verbalizer_hit_rate = (base_preds_unrestricted_np >= 0).mean()
    else:
        unrestricted_correct = 0.0
        verbalizer_hit_rate = 0.0
    iia = compute_iia(patched_preds_np, cf_labels_np)
    iia_per_class = compute_iia_per_class(
        patched_preds_np, cf_labels_np, class_names=label_names
    )
    confusion = _confusion_matrix(cf_labels_np, patched_preds_np, labels=label_names)
    transition = _transition_matrix(
        base_labels_np, cf_labels_np, patched_preds_np, labels=label_names
    )

    return {
        "factual_accuracy": float(factual_acc),
        "factual_accuracy_unrestricted": float(unrestricted_correct),
        "verbalizer_hit_rate": float(verbalizer_hit_rate),
        "iia": float(iia),
        "iia_per_class": iia_per_class,
        "confusion": confusion,
        "transition_iia": transition,
        "n_examples": int(cf_labels_np.shape[0]),
        "fixed_position": int(fixed_position),
        "base_preds": base_preds_np,
        "base_preds_unrestricted": base_preds_unrestricted_np,
        "patched_preds": patched_preds_np,
        "gold_cf_labels": cf_labels_np,
        "base_labels": base_labels_np,
    }
