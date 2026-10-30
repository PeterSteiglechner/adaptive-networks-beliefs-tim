"""`make_all`: every figure and table that a simulation folder's runs allow, and the report of what was made."""

import sys
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt

from .bnmetrics import fig4, fig5, figA4
from .index import index_folder
from .load import Loader
from .output import FigureResult, checked_formats, save_table
from .responses import fig3, figA3
from .select import PAPER, Baseline, FigureFailed, FigureNotApplicable, runs
from .sensitivity import figA8
from .social import fig6, figA7
from .tables import table3, tableA5, tableA6, tableA7
from .timeseries import fig2

FIGURES = ("fig2", "fig3", "figA3", "fig4", "fig5", "figA4", "fig6", "figA7", "figA8")
TABLES = ("table3", "tableA5", "tableA6", "tableA7")
FIG2_CONFIGS = ("adaptive", "staticlow", "statichigh")  # Fig. 2, A1, A2
MUS = (0.0, 0.005, 0.05)  # Fig. 4, A5, A6 (and their tables)


@dataclass(frozen=True)
class _Item:
    """One call of `make_all`: `name` (a figure or table function) with the `shown` kwargs that make up its key."""

    name: str
    func: Callable
    shown: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)  # passed on, not in the key

    @property
    def key(self) -> str:
        """`name` and the shown kwargs, e.g. `fig4 mu=0.005`, `table3 s=4 mu=0.0`."""
        return " ".join([self.name, *(f"{k}={_shown(k, v)}" for k, v in self.shown.items())])


def _shown(name, value) -> str:
    if isinstance(value, str):
        return value
    return repr(float(value)) if name == "mu" else f"{value:g}"


def _level(s):
    """A pressure from the index as the user would write it: `4`, not `4.0`."""
    return int(s) if float(s).is_integer() else float(s)


def _no_pressures(sim_dir, *, mu, **_):
    warnings.warn(f"table3 not applicable: no ensemble runs with s > 0 at μ={mu:g}", FigureNotApplicable,
                  stacklevel=2)


def _items(index, base: Baseline, mus, min_seeds: int) -> list[_Item]:
    """Every item `make_all` attempts, in report order: the figures, then the tables (`table3` per s > 0 present).

    A level with fewer than `min_seeds` runs is still an item, so that `table3` itself reports it as skipped and why.
    """
    few = dict(min_seeds=min_seeds)
    items = [*(_Item("fig2", fig2, dict(config=config)) for config in FIG2_CONFIGS),
             _Item("fig3", fig3, kwargs=few), _Item("figA3", figA3)]
    for mu in mus:
        items += [_Item(name, func, dict(mu=float(mu)), few) for name, func in (("fig4", fig4), ("fig5", fig5),
                                                                                  ("figA4", figA4))]
    items += [_Item("fig6", fig6, kwargs=few), _Item("figA7", figA7), _Item("figA8", figA8, kwargs=few)]
    for mu in mus:
        mu = float(mu)
        levels = sorted({_level(s) for s in runs(index, replace(base, mu=mu)).s if s > 0})
        items += [_Item("table3", table3, dict(s=s, mu=mu), few) for s in levels] or \
            [_Item("table3", _no_pressures, dict(mu=mu))]
        items += [_Item(name, func, dict(mu=mu), few) for name, func in (("tableA5", tableA5), ("tableA6", tableA6),
                                                                          ("tableA7", tableA7))]
    return items


def _chosen(items: list[_Item], figs, tables: bool) -> list[_Item]:
    """The items named in `figs` (all for `None`), without the tables unless `tables`."""
    if figs is not None:
        figs = [figs] if isinstance(figs, str) else list(figs)
        unknown = [name for name in figs if name not in {*FIGURES, *TABLES, *(item.key for item in items)}]
        if unknown:
            raise ValueError(f"unknown item(s) {', '.join(unknown)}; choose from {', '.join((*FIGURES, *TABLES))} "
                             f"or a report key such as 'fig4 mu=0.005'")
        items = [item for item in items if item.name in figs or item.key in figs]
    return [item for item in items if tables or item.name not in TABLES]


def make_all(sim_dir, out_dir, *, base: Baseline = PAPER, formats=("png",), figs=None, tables=True, mus=MUS,
             min_seeds=1, strict=False, loader=None, index=None) -> dict[str, list[Path] | str]:
    """Make and save every applicable figure and table of `sim_dir`'s runs into `out_dir`; the report.

    Attempts `fig2` for the adaptive, static-low and static-high configs (Fig. 2, A1, A2), `fig3`, `figA3`, `fig4` /
    `fig5` / `figA4` for each μ in `mus`, `fig6`, `figA7` and `figA8`, then (with `tables`) `table3` for each s > 0
    with runs, `tableA5`, `tableA6` and `tableA7` for each μ in `mus`, all for `base`. `figs` restricts them to those
    item names (`"fig4"`, `"table3"`) or report keys (`"fig4 mu=0.005"`). Figures are saved in `formats`, tables to
    `out_dir/tables/`. The folder is indexed once (unless `index` is given) and all items share one whole-reading
    `Loader` (unless `loader` is given), so each CSV is read once.

    The report maps each item's key to its saved paths, `"skipped: <reason>"` (the `FigureNotApplicable` messages)
    or `"error: <message>"`, and is printed as a summary. Warnings are still emitted. An unexpected error in one
    item is warned about with its traceback (`FigureFailed`) and the other items still run; `strict=True` raises it.
    """
    formats = checked_formats(formats)
    out_dir = Path(out_dir)
    index = index_folder(sim_dir) if index is None else index
    loader = Loader(whole=True) if loader is None else loader
    report = {}
    for item in _chosen(_items(index, base, mus, min_seeds), figs, tables):
        report[item.key] = _attempt(item, sim_dir, out_dir, formats, strict,
                                    dict(base=base, loader=loader, index=index, **item.shown, **item.kwargs))
    print(format_report(report, out_dir))
    return report


def _attempt(item: _Item, sim_dir, out_dir: Path, formats, strict: bool, kwargs) -> list[Path] | str:
    """Make and save one item; its report entry.

    The item's warnings are all recorded (for the skip reason, whatever the caller's filters) and then emitted again
    through the caller's filters, so a warning those turn into an error is raised after the item.
    """
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            outcome = _save(item.func(sim_dir, **kwargs), out_dir, formats)
        except Exception as raised:
            error, trace = raised, traceback.format_exc()
    _reemit(caught)
    if error is not None:
        if strict:
            raise error
        warnings.warn(f"{item.key} failed, carrying on with the other items:\n{trace}", FigureFailed, stacklevel=3)
        return f"error: {type(error).__name__}: {error}"
    if outcome is None:
        reasons = [str(w.message) for w in caught if issubclass(w.category, FigureNotApplicable)]
        outcome = "skipped: " + ("; ".join(reasons) or "not applicable")
    return outcome


def _save(result, out_dir: Path, formats) -> list[Path] | None:
    if result is None:
        return None
    if isinstance(result, FigureResult):
        try:
            return result.save(out_dir, formats)
        finally:
            plt.close(result.figure)
    return [save_table(result, out_dir)]


def _reemit(caught):
    """Emit recorded warnings again, through the caller's filters, as from the module that warned.

    A record keeps only the file, so the module is the loaded one of that file: its name for the filters' `module=`,
    its registry for the repeats, as for a direct call. Without one, the module is derived from the file name and an
    item's repeats are emitted once, as by default.
    """
    if not caught:
        return
    modules = {getattr(module, "__file__", None): module for module in list(sys.modules.values())}
    registry = {}
    for w in caught:
        module = modules.get(w.filename)
        where = dict(registry=registry) if module is None else \
            dict(module=module.__name__, registry=vars(module).setdefault("__warningregistry__", {}))
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source, **where)


def format_report(report: dict, out_dir=None) -> str:
    """The report as `make_all` prints it: a count line, then one line per item (paths relative to `out_dir`)."""
    counts = {kind: sum(isinstance(v, str) and v.startswith(kind) for v in report.values())
              for kind in ("skipped", "error")}
    written = len(report) - counts["skipped"] - counts["error"]
    head = f"make_all: {written} written, {counts['skipped']} skipped, {counts['error']} failed"
    width = max((len(key) for key in report), default=0)
    lines = [head + (f" (in {out_dir})" if out_dir is not None else "")]
    for key, value in report.items():
        if not isinstance(value, str):
            value = ", ".join(str(_relative(path, out_dir)) for path in value)
        lines.append(f"  {key:<{width}}  {' '.join(value.split())}")
    return "\n".join(lines)


def _relative(path: Path, out_dir) -> Path:
    try:
        return path.relative_to(out_dir) if out_dir is not None else path
    except ValueError:
        return path
