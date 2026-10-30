"""`python -m anb.analysis SIM_DIR OUT_DIR [...]`: `make_all` from the shell (see `--help`); it prints the report.

Exits with 1 if an item failed (with `--strict`, the error is raised instead), with 0 otherwise.
"""

import argparse
import sys
from dataclasses import fields

import matplotlib

from .runner import MUS, make_all
from .select import Baseline


def _names(text: str) -> list[str]:
    return [name.strip() for name in text.split(",") if name.strip()]


def _floats(text: str) -> list[float]:
    return [float(x) for x in _names(text)]


def _field(text: str) -> tuple[str, object]:
    """`FIELD=VALUE` of `Baseline`, the value parsed as that field's type."""
    name, sep, value = text.partition("=")
    types = {f.name: f.type for f in fields(Baseline)}
    if not sep or name not in types:
        raise argparse.ArgumentTypeError(f"expected FIELD=VALUE with FIELD one of {', '.join(types)}, got {text!r}")
    if types[name] is bool:
        if value.lower() not in ("true", "false", "1", "0"):
            raise argparse.ArgumentTypeError(f"{name} is true or false, got {value!r}")
        return name, value.lower() in ("true", "1")
    return name, types[name](value)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m anb.analysis",
                                description="Make every applicable figure and table of a simulation folder.")
    p.add_argument("sim_dir", help="the simulation folder (ensemble CSVs, detailed runs in its detailed/)")
    p.add_argument("out_dir", help="where the figures go (tables into its tables/); created if missing")
    p.add_argument("--figs", type=_names, help="comma-separated item names (fig2, …, figA8, table3, …, tableA7) or "
                                               "report keys ('fig4 mu=0.005'); default: all")
    p.add_argument("--formats", type=_names, default=["png"], help="comma-separated, png and/or pdf (default png)")
    p.add_argument("--mus", type=_floats, default=list(MUS),
                   help=f"comma-separated μ values of the Fig. 4 family and the tables (default {','.join(map(str, MUS))})")
    p.add_argument("--min-seeds", type=int, default=1, help="runs a cell needs to count as present (default 1)")
    p.add_argument("--base", type=_field, nargs="+", default=[], metavar="FIELD=VALUE",
                   help="base config fields that differ from the paper's, e.g. link_prob=0.2 M=15")
    p.add_argument("--no-tables", dest="tables", action="store_false", help="make no tables")
    p.add_argument("--strict", action="store_true", help="raise an item's unexpected error instead of carrying on")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    matplotlib.use("Agg")  # nothing is shown
    report = make_all(args.sim_dir, args.out_dir, base=Baseline(**dict(args.base)), formats=args.formats,
                      figs=args.figs, tables=args.tables, mus=args.mus, min_seeds=args.min_seeds, strict=args.strict)
    return int(any(isinstance(value, str) and value.startswith("error:") for value in report.values()))


if __name__ == "__main__":
    sys.exit(main())
