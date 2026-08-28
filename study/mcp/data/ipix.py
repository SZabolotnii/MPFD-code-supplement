"""McMaster IPIX 1993 Dartmouth sea-clutter files with a floating target (netCDF classic).

Fourteen stare-mode files; each carries 131072 sweeps x 2 TX polarisations x 14 range cells x
4 ADC channels of 8-bit samples. The primary and secondary target cells are the ones the
source page documents (1-indexed). Lake slug: ipix-dartmouth-1993.

The ADC bytes are unsigned. scipy's netcdf reader hands them back as int8, and read that way
pure clutter shows kurtosis ~ 114 and skewness ~ 11 -- wrap-around, not sea clutter. View as
uint8 and centre at 127.5.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

RAW = Path(os.environ.get("MPFD_IPIX_RAW",
                          Path(os.environ.get("RESEARCH_DATA_ROOT", Path.home() / "Project/datasets"))
                          / "store/ipix-dartmouth-1993/data"))

#: (file stem, primary cell, secondary cells) exactly as listed on
#: https://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html
FILES = [("19931107_135603_starea", 9, range(8, 12)), ("19931107_141630_starea", 9, range(8, 12)),
         ("19931107_145028_starea", 8, range(7, 10)), ("19931108_213827_starea", 7, range(6, 9)),
         ("19931108_220902_starea", 7, range(6, 9)), ("19931109_191449_starea", 7, range(6, 9)),
         ("19931109_202217_starea", 7, range(6, 10)), ("19931110_001635_starea", 7, range(5, 9)),
         ("19931111_163625_starea", 8, range(7, 11)), ("19931118_023604_stareC0000", 8, range(7, 11)),
         ("19931118_035737_stareC0000", 10, range(8, 13)), ("19931118_162155_stareC0000", 7, range(6, 10)),
         ("19931118_162658_stareC0000", 7, range(6, 10)), ("19931118_174259_stareC0000", 7, range(6, 10))]
N_CELLS = 14


def available() -> bool:
    return all((RAW / f"{stem}.cdf").exists() for stem, _, _ in FILES)


def like_i(stem: str, txpol: int = 0) -> np.ndarray:
    """Like-polarised I channel, all range cells: (nsweep, nrange), centred unsigned bytes."""
    f = netcdf_file(RAW / f"{stem}.cdf", "r", mmap=False)
    a = f.variables["adc_data"][:]
    i = int(f.variables["adc_like_I"].data)
    return a[:, txpol, :, i].view(np.uint8).astype(float) - 127.5


def windows(col: np.ndarray, N: int) -> np.ndarray:
    return col[: len(col) // N * N].reshape(-1, N)


def h0_h1(stem: str, prim: int, sec, N: int = 1024, txpol: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Clutter-only windows (every cell outside the secondary set) and primary-cell windows."""
    x = like_i(stem, txpol)
    clutter = [c for c in range(1, N_CELLS + 1) if c not in sec]
    h0 = np.concatenate([windows(x[:, c - 1], N) for c in clutter])
    h1 = windows(x[:, prim - 1], N)
    return h0, h1, clutter
