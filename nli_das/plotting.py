from __future__ import annotations

"""Heatmap plotting for activation patching sweeps.

We deliberately keep the dependency surface tiny: only matplotlib is
required. ``seaborn`` is used opportunistically if installed (nicer
default styles), but the module degrades gracefully without it.
"""


import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    import seaborn as _sns  # type: ignore
    _HAS_SEABORN = True
except Exception:  # pragma: no cover -- seaborn is optional
    _HAS_SEABORN = False

from .patching import PatchingResult


def _pretty_metric_name(name: str) -> str:
    return {
        "logit_recovery": "Logit-diff recovery",
        "iia": "Interchange intervention accuracy (IIA)",
    }.get(name, name)


def plot_patching_heatmap(
    result: Union[PatchingResult, np.ndarray],
    *,
    layers: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[int]] = None,
    position_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    metric_name: Optional[str] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = 0.0,
    annot: bool = False,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Render a layer x position heatmap and return the matplotlib Figure.

    Either pass a :class:`PatchingResult` (and we'll pull all the axis
    info from it) or a raw numpy array plus explicit ``layers`` /
    ``positions`` / ``position_labels``.

    The default colormap is the standard diverging red/blue used in the
    activation-patching literature, centered at 0 so positive recovery
    is red and negative is blue.

    For IIA grids you'll usually want ``cmap="viridis"``,
    ``center=None``, ``vmin=0, vmax=1`` -- you can pass those overrides
    yourself.
    """
    if isinstance(result, PatchingResult):
        grid = result.grid
        layers = layers if layers is not None else result.layers
        positions = positions if positions is not None else result.positions
        position_labels = (
            position_labels if position_labels is not None else result.position_labels
        )
        metric_name = metric_name or result.metric_name
        if title is None:
            base_acc = result.base_accuracy
            clean = result.clean_logit_diff
            cor = result.corrupted_logit_diff
            extra = []
            if base_acc is not None:
                extra.append(f"base_acc={base_acc:.2f}")
            if clean is not None and cor is not None:
                extra.append(f"LD clean={clean:.2f} / corr={cor:.2f}")
            suffix = f"  ({', '.join(extra)})" if extra else ""
            title = f"Activation patching: {_pretty_metric_name(metric_name)}{suffix}"
    else:
        grid = np.asarray(result)
        if layers is None or positions is None:
            raise ValueError(
                "When passing a raw array, you must also supply ``layers`` and ``positions``."
            )

    if grid.shape != (len(layers), len(positions)):
        raise ValueError(
            f"grid shape {grid.shape} does not match (len(layers)={len(layers)}, "
            f"len(positions)={len(positions)})"
        )

    fig, ax = plt.subplots(figsize=figsize)

    if _HAS_SEABORN:
        _sns.heatmap(
            grid,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=annot,
            fmt=".2f" if annot else "",
            cbar_kws={"label": _pretty_metric_name(metric_name or "")},
            xticklabels=position_labels if position_labels is not None else positions,
            yticklabels=layers,
        )
    else:
        # Pure-matplotlib fallback.
        if vmin is None:
            vmin = float(np.nanmin(grid))
        if vmax is None:
            vmax = float(np.nanmax(grid))
        im = ax.imshow(
            grid,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(_pretty_metric_name(metric_name or ""))
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(
            position_labels if position_labels is not None else positions,
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        if annot:
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def save_patching_heatmap(
    result: Union[PatchingResult, np.ndarray],
    path: str,
    **kwargs,
) -> str:
    """Plot a heatmap and save it to ``path``.

    Any extra ``**kwargs`` are forwarded to :func:`plot_patching_heatmap`.
    Returns the (absolute) path that was written.
    """
    fig = plot_patching_heatmap(result, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# DataFrame -> heatmap helpers
# ---------------------------------------------------------------------------


def df_to_heatmap_grid(
    df: pd.DataFrame,
    value_col: str,
    *,
    component: Optional[str] = None,
    agg: str = "mean",
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Pivot a tidy patching DataFrame into a ``(layer, position)`` grid.

    Parameters
    ----------
    df:
        Output of :func:`src.interventions.run_patching_sweep`.
    value_col:
        Column to aggregate -- typically ``"recovery"`` or ``"iia_correct"``.
    component:
        If given, filter the DataFrame to a single component first.
    agg:
        Pandas aggregation name (``"mean"``, ``"median"``, ...).

    Returns
    -------
    grid, layers, positions
        Numpy array of shape ``[len(layers), len(positions)]``, the sorted
        list of layer indices for the rows, and the sorted list of token
        positions for the columns.
    """
    if component is not None:
        df = df[df["component"] == component]
        if df.empty:
            raise ValueError(f"No rows for component {component!r}")
    pivot = df.pivot_table(
        index="layer", columns="position", values=value_col, aggfunc=agg
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot.to_numpy(), pivot.index.tolist(), pivot.columns.tolist()


def save_patching_heatmap_from_df(
    df: pd.DataFrame,
    path: str,
    *,
    value_col: str = "recovery",
    component: Optional[str] = None,
    agg: str = "mean",
    position_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    annot: bool = False,
    figsize: Tuple[float, float] = (10, 6),
) -> str:
    """Pivot ``df`` to a heatmap grid, render it, and save to ``path``.

    Sensible defaults for color scales:

    - ``value_col="recovery"``  -> diverging ``RdBu_r``, centered at 0.
    - ``value_col="iia_correct"`` -> sequential ``viridis``, fixed ``[0, 1]``.

    Override any of those via the explicit ``cmap`` / ``vmin`` / ``vmax`` /
    ``center`` kwargs. ``position_labels`` defaults to ``df.attrs["position_labels"]``
    if present (which :func:`run_patching_sweep` populates with decoded tokens).
    """
    grid, layers, positions = df_to_heatmap_grid(
        df, value_col=value_col, component=component, agg=agg
    )

    if cmap is None:
        cmap = "viridis" if value_col == "iia_correct" else "RdBu_r"
    if value_col == "iia_correct":
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0
        center = None  # don't center a [0,1] sequential map

    if position_labels is None:
        position_labels = df.attrs.get("position_labels")
        if position_labels is not None:
            # Restrict / reorder to the positions actually present in the pivot.
            try:
                position_labels = [position_labels[p] for p in positions]
            except (IndexError, TypeError):
                position_labels = None

    if title is None:
        comp_str = f" [{component}]" if component else ""
        base_acc = df.attrs.get("base_accuracy")
        suffix = f"  (base_acc={base_acc:.2f})" if base_acc is not None else ""
        title = f"Activation patching: {value_col}{comp_str}{suffix}"

    fig = plot_patching_heatmap(
        grid,
        layers=layers,
        positions=positions,
        position_labels=position_labels,
        title=title,
        metric_name=value_col,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        annot=annot,
        figsize=figsize,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)
