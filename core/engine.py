"""
Motore di analisi senza interfaccia (Parte 1, F5).

Raccoglie in `core/` la pipeline che `web_app.py` esegue per ogni colonna Y:
X convertita una volta -> fs risolta una volta (`resolve_fs`) -> trasformazioni
-> filtro -> FFT, con le cache dei risultati e i controlli qualità. Nessun
import di Streamlit: la stessa analisi è usabile da script, CLI o Claude Code.

- `parse_x`, `numeric_series`: conversione della X e delle Y come nell'app.
- `ResultCache`: cache LRU di trasformazioni, filtri, FFT e qualità (prima in
  `st.session_state`), con le stesse chiavi e dimensioni.
- `Recipe`: configurazione di analisi serializzabile in JSON, compatibile con i
  campi dei preset (`manual_fs`, `filter`, `fft`).
- `AnalysisSession`: un dataset (DataFrame o voce di `dataset_cache`) + una
  ricetta; carica dalla cache Parquet solo le colonne che servono.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, astuple, dataclass, field
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .quality import DataQualityReport, run_quality_checks
from .signal_tools import (
    FFTSpec,
    FilterSpec,
    FsInfo,
    apply_filter,
    compute_fft,
    resolve_fs,
    validate_filter_spec,
)
from .signal_transforms import TransformSpec, apply_transform_pipeline, transform_spec_cache_key

RECIPE_VERSION = "1"

MAX_FILTER_CACHE_SIZE = 32
MAX_FFT_CACHE_SIZE = 16
MAX_TRANSFORM_CACHE_SIZE = 32
MAX_QUALITY_CACHE_SIZE = 4

DEFAULT_QUALITY_CONFIG: Dict[str, Any] = {
    "gap_factor_k": 5.0,
    "spike_z": 4.0,
    "min_points": 20,
    "max_examples": 5,
}

TransformResultTuple = Tuple[pd.Series, Optional[pd.Series], List[str], bool]


# ---------------------------------------------------------------------------
# Conversioni (spostate da web_app.py)
# ---------------------------------------------------------------------------
def to_datetime_flexible(values: Any) -> pd.Series:
    """Datetime con `format="mixed"` quando disponibile (meno avvisi di inferenza)."""
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")  # type: ignore[arg-type]
    except TypeError:
        return pd.to_datetime(values, errors="coerce")  # type: ignore[arg-type]


def parse_x(raw: Optional[pd.Series]) -> Optional[pd.Series]:
    """Asse X più utile: datetime se già tale, numerico se >= 80% valido, altrimenti datetime."""
    if raw is None:
        return None
    if pd.api.types.is_datetime64_any_dtype(raw) or pd.api.types.is_timedelta64_dtype(raw):
        return to_datetime_flexible(raw)
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        return numeric
    try:
        return to_datetime_flexible(raw)
    except Exception:
        return None


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    y = pd.to_numeric(df[column], errors="coerce")
    y.name = column
    return y


# ---------------------------------------------------------------------------
# Cache dei risultati
# ---------------------------------------------------------------------------
class _LRU:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._items: "OrderedDict[Hashable, Any]" = OrderedDict()

    def get(self, key: Hashable) -> Any:
        return self._items.get(key)

    def put(self, key: Hashable, value: Any) -> None:
        if key not in self._items and len(self._items) >= self.max_size:
            # Come prima in web_app: si elimina la voce inserita per prima.
            self._items.popitem(last=False)
        self._items[key] = value

    def pop(self, key: Hashable) -> None:
        self._items.pop(key, None)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class ResultCache:
    """Cache di trasformazioni, filtri, FFT e controlli qualità per un'analisi.

    Le chiavi includono la firma del dataset (`signature`), la colonna, i
    parametri e lo stato del riempimento (`fill_stamp`): cambiando uno di questi
    il risultato viene ricalcolato.
    """

    def __init__(self) -> None:
        self._filter = _LRU(MAX_FILTER_CACHE_SIZE)
        self._fft = _LRU(MAX_FFT_CACHE_SIZE)
        self._transform = _LRU(MAX_TRANSFORM_CACHE_SIZE)
        self._quality = _LRU(MAX_QUALITY_CACHE_SIZE)
        self.stats: Dict[str, int] = {"filter_hits": 0, "filter_misses": 0, "fft_hits": 0, "fft_misses": 0}

    def clear(self) -> None:
        for cache in (self._filter, self._fft, self._transform, self._quality):
            cache.clear()

    def transform(
        self,
        series: pd.Series,
        x: Optional[pd.Series],
        specs: Sequence[TransformSpec],
        fs_info: Optional[FsInfo],
        signature: Hashable,
        column: str,
        fill_stamp: Hashable,
    ) -> TransformResultTuple:
        """Pipeline di trasformazioni; solleva ValueError se un passo non è valido."""
        specs_key = tuple(transform_spec_cache_key(s) for s in specs)
        key = (
            column,
            signature,
            specs_key,
            fs_info.value if fs_info else None,
            fs_info.source if fs_info else None,
            fill_stamp,
        )
        cached = self._transform.get(key)
        if cached is not None:
            return cached
        y_out, x_out, descriptions, changed = apply_transform_pipeline(
            series, x, list(specs), fs_info, col_name=column
        )
        result: TransformResultTuple = (
            y_out.copy(),
            x_out.copy() if x_out is not None else None,
            descriptions,
            changed,
        )
        self._transform.put(key, result)
        return result

    def filter(
        self,
        series: pd.Series,
        x: Optional[pd.Series],
        fspec: FilterSpec,
        fs: Optional[float],
        fs_source: Optional[str],
        signature: Hashable,
        column: str,
        fill_stamp: Hashable,
    ) -> Optional[pd.Series]:
        """Serie filtrata, o None se il filtro fallisce."""
        key = (column, signature, astuple(fspec), fs, fs_source, fill_stamp)
        cached = self._filter.get(key)
        if cached is not None:
            if cached.index.equals(series.index):
                self.stats["filter_hits"] += 1
                return cached
            # Indici cambiati (es. decimazione): la voce non vale più.
            self._filter.pop(key)
        self.stats["filter_misses"] += 1
        try:
            filtered, _ = apply_filter(series, x, fspec, fs_override=fs)
        except Exception:
            return None
        self._filter.put(key, filtered.copy())
        return filtered

    def fft(
        self,
        series: pd.Series,
        fs: float,
        fs_source: Optional[str],
        fftspec: FFTSpec,
        signature: Hashable,
        column: str,
        is_filtered: bool,
        fill_stamp: Hashable,
        fspec: Optional[FilterSpec] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # La firma del filtro conta solo per lo spettro del segnale filtrato.
        filter_sig = astuple(fspec) if (is_filtered and fspec is not None) else None
        key = (column, signature, is_filtered, astuple(fftspec), fs, fs_source, fill_stamp, filter_sig)
        cached = self._fft.get(key)
        if cached is not None:
            self.stats["fft_hits"] += 1
            return cached
        self.stats["fft_misses"] += 1
        freqs, amp = compute_fft(series, fs, detrend=fftspec.detrend, window=fftspec.window)
        self._fft.put(key, (freqs.copy(), amp.copy()))
        return freqs, amp

    def quality(
        self, df: pd.DataFrame, signature: Hashable, config: Optional[Dict[str, Any]] = None
    ) -> DataQualityReport:
        cfg = {**DEFAULT_QUALITY_CONFIG, **(config or {})}
        columns = list(df.columns)
        key = (signature, tuple(columns), (cfg["gap_factor_k"], cfg["spike_z"], cfg["min_points"]))
        cached = self._quality.get(key)
        if cached is not None:
            return cached
        report = run_quality_checks(
            df=df,
            x_col=None,
            y_cols=columns,
            gap_factor_k=cfg["gap_factor_k"],
            spike_z=cfg["spike_z"],
            min_points=cfg["min_points"],
            max_examples=cfg["max_examples"],
        )
        self._quality.put(key, report)
        return report


# ---------------------------------------------------------------------------
# Ricetta di analisi
# ---------------------------------------------------------------------------
@dataclass
class RecipeTransform:
    spec: TransformSpec
    targets: List[str] = field(default_factory=list)  # vuoto = tutte le colonne Y

    def applies_to(self, column: str) -> bool:
        return not self.targets or column in self.targets


@dataclass
class Recipe:
    """Configurazione completa di un'analisi, salvabile in JSON."""

    x: Optional[str] = None
    manual_fs: Optional[float] = None
    fill_missing: bool = False
    interpolate_x: bool = False
    transforms: List[RecipeTransform] = field(default_factory=list)
    filter: FilterSpec = field(default_factory=lambda: FilterSpec(kind="ma"))
    fft: FFTSpec = field(default_factory=FFTSpec)
    fft_use_filtered: bool = True

    @property
    def fill_stamp(self) -> Tuple[bool, bool]:
        return (bool(self.fill_missing), bool(self.interpolate_x))

    def transforms_for(self, column: str) -> List[TransformSpec]:
        return [t.spec for t in self.transforms if t.applies_to(column)]

    def to_dict(self) -> Dict[str, Any]:
        filter_dict = asdict(self.filter)
        if isinstance(filter_dict.get("cutoff"), tuple):
            filter_dict["cutoff"] = list(filter_dict["cutoff"])
        return {
            "recipe_version": RECIPE_VERSION,
            "x": self.x,
            "manual_fs": self.manual_fs,
            "fill_missing": self.fill_missing,
            "interpolate_x": self.interpolate_x,
            "transforms": [{"spec": asdict(t.spec), "targets": list(t.targets)} for t in self.transforms],
            "filter": filter_dict,
            "fft": asdict(self.fft),
            "fft_use_filtered": self.fft_use_filtered,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recipe":
        version = str(data.get("recipe_version", RECIPE_VERSION))
        if version != RECIPE_VERSION:
            raise ValueError(f"Versione ricetta non supportata: {version}")
        filter_data = dict(data.get("filter") or {"kind": "ma"})
        cutoff = filter_data.get("cutoff")
        if isinstance(cutoff, list):
            filter_data["cutoff"] = (tuple(cutoff) + (None,))[:2] if cutoff else None
        manual_fs = data.get("manual_fs")
        return cls(
            x=data.get("x"),
            manual_fs=float(manual_fs) if manual_fs else None,
            fill_missing=bool(data.get("fill_missing", False)),
            interpolate_x=bool(data.get("interpolate_x", False)),
            transforms=[
                RecipeTransform(spec=TransformSpec(**t["spec"]), targets=list(t.get("targets") or []))
                for t in data.get("transforms", [])
            ],
            filter=FilterSpec(**filter_data),
            fft=FFTSpec(**(data.get("fft") or {})),
            fft_use_filtered=bool(data.get("fft_use_filtered", True)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Recipe":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Sessione di analisi
# ---------------------------------------------------------------------------
@dataclass
class FilterResult:
    series: Optional[pd.Series]  # None se il filtro non è attivo o non è applicabile
    applied: bool
    message: str = ""


@dataclass
class FFTResult:
    freqs: np.ndarray
    amp: np.ndarray
    used_filtered: bool
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.freqs.size > 0


class AnalysisSession:
    """Un dataset e una ricetta: serie elaborate, filtri, FFT, qualità e statistiche."""

    def __init__(
        self,
        source: Union[pd.DataFrame, Any],
        recipe: Optional[Recipe] = None,
        signature: Optional[Hashable] = None,
        cache: Optional[ResultCache] = None,
    ) -> None:
        self._dataset = None if isinstance(source, pd.DataFrame) else source
        self._frame: Optional[pd.DataFrame] = source if isinstance(source, pd.DataFrame) else None
        self._loaded = pd.DataFrame(index=pd.RangeIndex(self.num_rows))
        self.recipe = recipe or Recipe()
        self.signature: Hashable = signature if signature is not None else self._default_signature()
        self.cache = cache or ResultCache()
        self._prepared_key: Optional[Tuple] = None
        self._prepared: Optional[pd.DataFrame] = None
        self._fs_key: Optional[Tuple] = None
        self._fs_info: Optional[FsInfo] = None

    @classmethod
    def open(
        cls,
        path: Union[str, Path],
        apply_cleaning: bool = True,
        cache_dir: Optional[Path] = None,
        recipe: Optional[Recipe] = None,
    ) -> "AnalysisSession":
        """Apre un CSV tramite la cache Parquet (conversione solo la prima volta)."""
        from . import dataset_cache

        dataset = dataset_cache.open_dataset(path, apply_cleaning=apply_cleaning, directory=cache_dir)
        return cls(dataset, recipe=recipe)

    # --- dataset ---------------------------------------------------------
    def _default_signature(self) -> Hashable:
        if self._dataset is not None:
            return ("dataset", self._dataset.directory.name)
        return ("frame", id(self._frame))

    @property
    def columns(self) -> List[str]:
        return list(self._dataset.columns) if self._dataset is not None else [str(c) for c in self._frame.columns]

    @property
    def num_rows(self) -> int:
        return int(self._dataset.num_rows) if self._dataset is not None else len(self._frame)

    @property
    def report(self) -> Any:
        return self._dataset.report if self._dataset is not None else None

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._dataset.meta) if self._dataset is not None else {}

    @property
    def numeric_columns(self) -> List[str]:
        if self._dataset is not None:
            return list(self._dataset.numeric_columns)
        return [str(c) for c in self._frame.columns if pd.api.types.is_numeric_dtype(self._frame[c])]

    def frame(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Colonne originali richieste (dalla cache Parquet solo quelle mancanti)."""
        wanted = list(self.columns if columns is None else columns)
        unknown = [c for c in wanted if c not in self.columns]
        if unknown:
            raise KeyError(f"Colonne non presenti: {unknown}")
        if self._frame is not None:
            return self._frame[wanted]
        missing = [c for c in wanted if c not in self._loaded.columns]
        if missing:
            loaded = self._dataset.load(missing)
            for name in missing:
                self._loaded[name] = loaded[name].array
        return self._loaded[wanted]

    # --- preparazione: interpolazione X e riempimento --------------------
    def _needed(self, columns: Sequence[str]) -> List[str]:
        names = ([self.recipe.x] if self.recipe.x else []) + list(columns)
        seen: set = set()
        return [c for c in names if not (c in seen or seen.add(c))]

    def prepared(self, columns: Sequence[str]) -> pd.DataFrame:
        """X + colonne richieste dopo interpolazione X e forward fill (come l'app)."""
        needed = self._needed(columns)
        key = (tuple(needed), self.recipe.x, self.recipe.fill_stamp)
        if self._prepared_key == key and self._prepared is not None:
            return self._prepared
        df = self.frame(needed).copy()
        x = self.recipe.x
        if self.recipe.interpolate_x and x and x in df.columns and df[x].isna().any():
            df[x] = df[x].interpolate(method="index")
        if self.recipe.fill_missing:
            df = df.ffill()
        self._prepared_key, self._prepared = key, df
        return df

    def x_values(self) -> Optional[pd.Series]:
        if not self.recipe.x:
            return None
        return parse_x(self.prepared([])[self.recipe.x])

    def fs_info(self) -> FsInfo:
        """`resolve_fs` calcolata una volta per X, fs manuale e riempimento."""
        manual = self.recipe.manual_fs if self.recipe.manual_fs and self.recipe.manual_fs > 0 else None
        key = (self.recipe.x, manual, self.recipe.fill_stamp)
        if self._fs_key != key or self._fs_info is None:
            self._fs_info = resolve_fs(self.x_values(), manual)
            self._fs_key = key
        return self._fs_info

    @property
    def fs(self) -> Optional[float]:
        info = self.fs_info()
        return info.value if info.value and info.value > 0 else None

    # --- pipeline per colonna --------------------------------------------
    def raw_series(self, column: str) -> Tuple[pd.Series, Optional[pd.Series]]:
        df = self.prepared([column])
        x = parse_x(df[self.recipe.x]) if self.recipe.x else None
        return numeric_series(df, column), x

    def series(self, column: str) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Serie Y (e X) dopo le trasformazioni che si applicano alla colonna."""
        y, x = self.raw_series(column)
        specs = self.recipe.transforms_for(column)
        if not specs:
            return y, x
        y_out, x_out, _, _ = self.cache.transform(
            y, x, specs, self.fs_info(), self.signature, column, self.recipe.fill_stamp
        )
        return y_out, x_out

    def filtered(self, column: str) -> FilterResult:
        fspec = self.recipe.filter
        if not fspec.enabled:
            return FilterResult(None, False, "Filtro non attivo.")
        ok, message = validate_filter_spec(fspec, self.fs)
        if not ok:
            return FilterResult(None, False, message)
        y, x = self.series(column)
        result = self.cache.filter(
            y, x, fspec, self.fs, self.fs_info().source, self.signature, column, self.recipe.fill_stamp
        )
        if result is None:
            return FilterResult(None, False, "Errore nel calcolo del filtro.")
        return FilterResult(result, True, message)

    def fft(self, column: str) -> FFTResult:
        empty = np.array([])
        info = self.fs_info()
        if not self.fs:
            return FFTResult(empty, empty, False, "fs non disponibile.")
        if not info.is_uniform:
            detail = "; ".join(info.warnings) if info.warnings else "campionamento irregolare."
            return FFTResult(empty, empty, False, detail)
        y, _ = self.series(column)
        use_filtered = False
        if self.recipe.fft_use_filtered:
            filtered = self.filtered(column)
            if filtered.applied:
                y, use_filtered = filtered.series, True
        freqs, amp = self.cache.fft(
            y,
            self.fs,
            info.source,
            self.recipe.fft,
            self.signature,
            column,
            use_filtered,
            self.recipe.fill_stamp,
            self.recipe.filter,
        )
        message = "" if freqs.size else "Serie troppo corta o parametri non validi."
        return FFTResult(freqs, amp, use_filtered, message)

    # --- riepiloghi ------------------------------------------------------
    def _y_columns(self, columns: Optional[Sequence[str]]) -> List[str]:
        if columns is not None:
            return list(columns)
        return [c for c in self.numeric_columns if c != self.recipe.x]

    def quality(self, columns: Optional[Sequence[str]] = None, config: Optional[Dict[str, Any]] = None) -> DataQualityReport:
        cols = self._y_columns(columns)
        return self.cache.quality(self.frame(cols), self.signature, config)

    def stats(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Statistiche delle serie elaborate (dopo trasformazioni e filtro, se attivo)."""
        rows = []
        for column in self._y_columns(columns):
            filtered = self.filtered(column)
            y = filtered.series if filtered.applied else self.series(column)[0]
            values = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            values = values[np.isfinite(values)]
            rows.append({
                "colonna": column,
                "count": int(values.size),
                "mean": float(values.mean()) if values.size else np.nan,
                "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "min": float(values.min()) if values.size else np.nan,
                "max": float(values.max()) if values.size else np.nan,
                "rms": float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan,
            })
        return pd.DataFrame(rows).set_index("colonna") if rows else pd.DataFrame()

    # --- ricetta ---------------------------------------------------------
    def to_recipe(self) -> Dict[str, Any]:
        return self.recipe.to_dict()

    def from_recipe(self, data: Union[Recipe, Dict[str, Any], str]) -> "AnalysisSession":
        if isinstance(data, Recipe):
            self.recipe = data
        elif isinstance(data, str):
            self.recipe = Recipe.from_json(data)
        else:
            self.recipe = Recipe.from_dict(data)
        return self
