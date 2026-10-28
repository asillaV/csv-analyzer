"""
Benchmark del caricamento CSV: percorso attuale vs nuovo ingest (Parte 1, F0).

Ogni misura gira in un processo Python separato, così il picco di RAM di un
loader non contamina l'altro.

Uso:
    python scripts/bench_load.py tests_csv/bench_large_500k.csv
    python scripts/bench_load.py FILE --loaders reference new --repeat 3
    python scripts/bench_load.py FILE --compare          # differenze vecchio/nuovo
    python scripts/bench_load.py BIG.csv --slice-mb 300  # crea e misura una fetta
    python scripts/bench_load.py FILE --json risultati.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOADERS = ("reference", "new")


# ---------------------------------------------------------------------------
# Fetta di un file grande, rispettando encoding e fine riga
# ---------------------------------------------------------------------------
def make_slice(src: Path, size_mb: int, out_dir: Path) -> Path:
    """Copia i primi `size_mb` MB di `src` tagliando sull'ultima riga completa."""
    n = size_mb * 1024 * 1024
    with open(src, "rb") as f:
        data = f.read(n)
    if data.startswith(b"\xff\xfe"):
        newline, step = b"\n\x00", 2
    elif data.startswith(b"\xfe\xff"):
        newline, step = b"\x00\n", 2
    else:
        newline, step = b"\n", 1
    cut = data.rfind(newline)
    while cut > 0 and cut % step:
        cut = data.rfind(newline, 0, cut)
    if cut <= 0:
        raise ValueError(f"nessuna riga completa nei primi {size_mb} MB di {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_{size_mb}mb{src.suffix}"
    dst.write_bytes(data[: cut + len(newline)])
    return dst


# ---------------------------------------------------------------------------
# Misura singola (eseguita nel processo figlio)
# ---------------------------------------------------------------------------
def _measure(loader: str, path: Path) -> Dict[str, Any]:
    import psutil

    logging.disable(logging.CRITICAL)
    proc = psutil.Process()
    rss_start = proc.memory_info().rss
    peak = [rss_start]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            stop.wait(0.05)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        if loader == "reference":
            from core.analyzer import analyze_csv
            from core.loader_optimized import load_csv

            meta = analyze_csv(str(path))
            timings["analyze_s"] = time.perf_counter() - t0
            df, _report = load_csv(
                str(path),
                encoding=meta.get("encoding"),
                delimiter=meta.get("delimiter"),
                header=meta.get("header"),
                apply_cleaning=True,
                return_details=True,
                use_fast_ingest=False,
            )
        else:
            from core import ingest

            result = ingest.load(path)
            df = result.df
            timings.update(getattr(result, "timings", {}) or {})
        error = None
    except Exception as exc:  # misura comunque tempo e RAM fino all'errore
        df, error = None, repr(exc)
    total = time.perf_counter() - t0
    stop.set()
    thread.join()
    peak[0] = max(peak[0], proc.memory_info().rss)

    out: Dict[str, Any] = {
        "loader": loader,
        "file": str(path),
        "file_mb": path.stat().st_size / 2**20,
        "total_s": total,
        "peak_rss_mb": peak[0] / 2**20,
        "peak_delta_mb": (peak[0] - rss_start) / 2**20,
        "timings": timings,
        "error": error,
    }
    if df is not None:
        out["rows"], out["cols"] = df.shape
        out["df_mb"] = df.memory_usage(deep=True).sum() / 2**20
    return out


def _run_isolated(loader: str, path: Path) -> Dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", loader, str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"loader": loader, "file": str(path), "error": (res.stderr or res.stdout).strip()[-500:]}


# ---------------------------------------------------------------------------
# Confronto vecchio/nuovo
# ---------------------------------------------------------------------------
def _compare(path: Path) -> List[str]:
    logging.disable(logging.CRITICAL)
    from tests.parity import LoadOutcome, apply_msvc_rule, compare_outcomes, load_reference
    from core import ingest

    ref = apply_msvc_rule(load_reference(path), path)
    try:
        r = ingest.load(path)
        new = LoadOutcome(df=r.df, report=r.report, meta=dict(r.meta))
    except Exception as exc:
        new = LoadOutcome(error=exc)
    return compare_outcomes(ref, new)


def _ingest_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("core.ingest") is not None


def _fmt_row(r: Dict[str, Any]) -> str:
    if r.get("error") and "total_s" not in r:
        return f"{r['loader']:10s} ERRORE {r['error'][:100]}"
    shape = f"{r.get('rows', '?')}x{r.get('cols', '?')}"
    err = f"  ERRORE {r['error'][:60]}" if r.get("error") else ""
    phases = " ".join(f"{k}={v:.2f}" for k, v in r.get("timings", {}).items())
    return (
        f"{r['loader']:10s} {r['total_s']:8.2f}s  picco {r['peak_rss_mb']:7.0f} MB "
        f"(+{r['peak_delta_mb']:.0f})  df {r.get('df_mb', 0):6.0f} MB  {shape:>12s}  {phases}{err}"
    )


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--_worker":
        print(json.dumps(_measure(argv[1], Path(argv[2]))))
        return 0

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--loaders", nargs="+", choices=LOADERS, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--slice-mb", type=int, default=None, help="misura una fetta dei primi N MB")
    ap.add_argument("--compare", action="store_true", help="elenca le differenze vecchio/nuovo")
    ap.add_argument("--json", type=Path, default=None, help="salva i risultati in JSON")
    args = ap.parse_args(argv)

    loaders = args.loaders or (["reference", "new"] if _ingest_available() else ["reference"])
    if "new" in loaders and not _ingest_available():
        print("core.ingest non disponibile: misuro solo 'reference'.")
        loaders = [x for x in loaders if x != "new"]

    slice_dir = Path(tempfile.gettempdir()) / "analizzatore_bench_slices"
    results: List[Dict[str, Any]] = []
    for src in args.files:
        path = make_slice(src, args.slice_mb, slice_dir) if args.slice_mb else src
        print(f"\n== {path.name} ({path.stat().st_size / 2**20:.0f} MB)")
        for loader in loaders:
            for i in range(args.repeat):
                r = _run_isolated(loader, path)
                r["run"] = i + 1
                results.append(r)
                print("  " + _fmt_row(r))
        if args.compare and _ingest_available():
            diffs = _compare(path)
            print(f"  parità: {'OK' if not diffs else f'{len(diffs)} differenze'}")
            for d in diffs[:20]:
                print(f"    - {d}")
        if args.slice_mb:
            os.remove(path)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nRisultati salvati in {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
