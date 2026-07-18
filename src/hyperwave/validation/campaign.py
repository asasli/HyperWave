"""Drive an injection-and-recovery campaign and aggregate it into a PP-test.

The expensive loop -- draw a truth from the prior, inject, recover -- is supplied
by the caller as a ``run_injection(index, rng) -> Result`` callable, so the same
harness works for any likelihood/sampler. :func:`run_pp_campaign` runs the loop
(optionally resuming from already-saved results), persists each
:class:`hyperwave.Result`, and writes a PP-plot plus a JSON summary.

See ``examples/validation/pp_fast.py`` for a concrete fast-PE campaign.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np

from ..result import Result
from .pp import credible_levels, make_pp_plot, pp_pvalues

__all__ = ["run_pp_campaign"]


def run_pp_campaign(
    run_injection: Callable[[int, np.random.Generator], Result],
    n_injections: int,
    outdir: str,
    *,
    seed: int = 0,
    resume: bool = True,
    make_plot: bool = True,
    title: Optional[str] = None,
) -> dict:
    """Run ``n_injections`` injection/recovery trials and summarize calibration.

    Parameters
    ----------
    run_injection:
        ``run_injection(index, rng) -> Result``. Must draw the truth from the
        prior using ``rng`` (so the PP-test is valid) and return a
        :class:`hyperwave.Result` with its ``injection`` set.
    n_injections:
        Number of injections.
    outdir:
        Directory for per-injection result files and the summary/plot.
    seed:
        Base seed; injection ``i`` uses an independent child stream so results are
        reproducible and resumable.
    resume:
        Skip injections whose result file already exists (re-loaded instead).

    Returns
    -------
    summary:
        ``{"pvalues": {...}, "n_injections": int, "results": [paths]}``.
    """
    os.makedirs(outdir, exist_ok=True)
    seeds = np.random.SeedSequence(seed).spawn(n_injections)
    results: list[Result] = []
    paths: list[str] = []

    for i in range(n_injections):
        path = os.path.join(outdir, f"inj_{i:04d}.npz")
        if resume and os.path.exists(path):
            results.append(Result.load(path))
            paths.append(path)
            continue
        rng = np.random.default_rng(seeds[i])
        res = run_injection(i, rng)
        if res.injection is None:
            raise ValueError(f"injection {i}: run_injection returned a Result without an injection")
        res.metadata.setdefault("injection_index", i)
        res.save(path)
        results.append(res)
        paths.append(path)
        print(f"[pp] injection {i + 1}/{n_injections} done -> {path}")

    levels, names = credible_levels(results)
    pvalues = pp_pvalues(levels, names)
    summary = {"pvalues": pvalues, "n_injections": len(results),
               "parameters": names, "results": paths}
    with open(os.path.join(outdir, "pp_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if make_plot:
        fig, _ = make_pp_plot(results, names, title=title)
        plot_path = os.path.join(outdir, "pp_plot.png")
        fig.savefig(plot_path, dpi=140, bbox_inches="tight")
        summary["plot"] = plot_path
        print(f"[pp] PP-plot -> {plot_path}  (combined p={pvalues['combined']:.3f})")

    return summary
