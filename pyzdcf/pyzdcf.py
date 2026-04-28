#!/usr/bin/env python

__author__ = "Isidora Jankov"
__credits__ = [
    "Andjelka Kovacevic",
    "Dragana Ilic",
    "Paula Sanchez Saez",
    "Robert Nikutta",
]
__maintainer__ = "Isidora Jankov"
__contact__ = "isidora_jankov@matf.bg.ac.rs"
__license__ = "MIT License"
__version__ = "1.0.3"

from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


ENOUGH = 11
ISEED = 123456
RSLUTN = 0.001
EPS = 1e-7
STREAMING_THRESHOLD = 3000


@dataclass(frozen=True)
class ZDCFConfig:
    autocf: bool
    uniform_sampling: bool
    omit_zero_lags: bool
    minpts: int


@dataclass(frozen=True)
class ZDCFBin:
    tau: float
    sigtm: float
    sigtp: float
    a_idx: np.ndarray
    b_idx: np.ndarray


def _build_path(dir_path, filename):
    return Path(dir_path) / filename


def _clamp_corr(value, eps=EPS):
    return np.clip(value, -1 + eps, 1 - eps)


def _validate_bool(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"Invalid {name}. Please enter True/False.")


def _validate_sparse_mode(sparse):
    if sparse in ("auto", True, False):
        return sparse
    raise ValueError("Invalid sparse mode. Use 'auto', True, or False.")


def _validate_text(value, name):
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError(f"Invalid {name}. Please enter a non-empty string.")
    value = value.strip()
    if not value:
        raise ValueError(f"Invalid {name}. Please enter a non-empty string.")
    return value


def _prepare_light_curve(lc):
    if not isinstance(lc, pd.DataFrame):
        raise TypeError("Light curve input must be a pandas.DataFrame.")

    required = {"t", "flux", "err"}
    missing = required.difference(lc.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Light curve is missing required columns: {missing_str}")

    lc = lc.loc[:, ["t", "flux", "err"]].copy()
    lc = lc.apply(pd.to_numeric, errors="coerce")
    values = lc[["t", "flux", "err"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Light curve contains NaN or non-finite values.")
    if (lc["err"] < 0).any():
        raise ValueError("Flux uncertainties must be non-negative.")

    lc.sort_values(by="t", ascending=True, inplace=True)
    lc = _combine_duplicate_times(lc)
    lc.reset_index(drop=True, inplace=True)

    return lc


def _combine_duplicate_times(lc):
    rows = []
    for t, group in lc.groupby("t", sort=True):
        flux = group["flux"].to_numpy(dtype=float)
        err = group["err"].to_numpy(dtype=float)

        if len(group) == 1:
            rows.append((float(t), float(flux[0]), float(err[0])))
            continue

        zero_err = err == 0
        if zero_err.any():
            rows.append((float(t), float(np.mean(flux[zero_err])), 0.0))
            continue

        weights = 1.0 / np.square(err)
        rows.append(
            (
                float(t),
                float(np.average(flux, weights=weights)),
                float(np.sqrt(1.0 / np.sum(weights))),
            )
        )

    return pd.DataFrame(rows, columns=["t", "flux", "err"])


def _count_zero_pairs(a_t, b_t):
    return int(np.intersect1d(a_t, b_t, assume_unique=True).size)


def _total_possible_pairs(a_t, b_t, config):
    a_n = len(a_t)
    if config.autocf:
        if config.omit_zero_lags:
            return (a_n * (a_n - 1)) // 2
        return (a_n * (a_n + 1)) // 2

    total = a_n * len(b_t)
    if config.omit_zero_lags:
        total -= _count_zero_pairs(a_t, b_t)
    return total


def _unused_pairs_from_bins(bins, a_t, b_t, config):
    used_pairs = int(sum(len(zbin.a_idx) for zbin in bins))
    total_pairs = _total_possible_pairs(a_t, b_t, config)
    return total_pairs - used_pairs


def _lag_tolerance(a_t, b_t, config, sorted_lags=None):
    if sorted_lags is not None:
        if len(sorted_lags) == 0:
            return 0.0
        return float((sorted_lags[-1] - sorted_lags[0]) * RSLUTN)

    if len(a_t) == 0 or len(b_t) == 0:
        return 0.0

    if config.autocf:
        max_tau = float(a_t[-1] - a_t[0])
        if config.omit_zero_lags:
            if len(a_t) < 2:
                return 0.0
            min_tau = float(np.min(np.diff(a_t)))
        else:
            min_tau = 0.0
    else:
        min_tau = float(b_t[0] - a_t[-1])
        max_tau = float(b_t[-1] - a_t[0])

    return float((max_tau - min_tau) * RSLUTN)


def _dense_sorted_pairs(a_t, b_t, config):
    total_pairs = _total_possible_pairs(a_t, b_t, config)
    if total_pairs == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    waidx = np.empty(total_pairs, dtype=np.int32)
    wbidx = np.empty(total_pairs, dtype=np.int32)
    wtau = np.empty(total_pairs, dtype=np.float64)

    l1 = 0
    b_all = np.arange(len(b_t), dtype=np.int32)

    if config.autocf:
        for i in range(len(a_t)):
            start = i + int(config.omit_zero_lags)
            if start >= len(b_t):
                continue
            tij = b_t[start:] - b_t[i]
            count = len(tij)
            l2 = l1 + count
            wtau[l1:l2] = tij
            waidx[l1:l2] = i
            wbidx[l1:l2] = np.arange(start, len(b_t), dtype=np.int32)
            l1 = l2
    else:
        for i in range(len(a_t)):
            tij = b_t - a_t[i]
            if config.omit_zero_lags:
                mask = tij != 0
                tij = tij[mask]
                wb = b_all[mask]
            else:
                wb = b_all
            count = len(tij)
            if count == 0:
                continue
            l2 = l1 + count
            wtau[l1:l2] = tij
            waidx[l1:l2] = i
            wbidx[l1:l2] = wb
            l1 = l2

    wtau = wtau[:l1]
    waidx = waidx[:l1]
    wbidx = wbidx[:l1]
    idx = np.argsort(wtau, kind="stable")

    return wtau[idx], waidx[idx], wbidx[idx]


def _iter_dense_indices(sorted_lags, a_idx, b_idx, indices):
    for idx in indices:
        yield float(sorted_lags[idx]), int(a_idx[idx]), int(b_idx[idx])


def _iter_ccf_pairs_forward_stream(a_t, b_t, omit_zero_lags, lower_bound=None):
    heap = []
    for i, at in enumerate(a_t):
        if lower_bound is None:
            j = 0
        else:
            j = int(np.searchsorted(b_t, at + lower_bound, side="left"))
        while j < len(b_t) and omit_zero_lags and b_t[j] == at:
            j += 1
        if j < len(b_t):
            heapq.heappush(heap, (float(b_t[j] - at), i, j))

    while heap:
        lag, i, j = heapq.heappop(heap)
        yield lag, i, j

        j += 1
        while j < len(b_t) and omit_zero_lags and b_t[j] == a_t[i]:
            j += 1
        if j < len(b_t):
            heapq.heappush(heap, (float(b_t[j] - a_t[i]), i, j))


def _iter_ccf_pairs_backward_stream(a_t, b_t, omit_zero_lags, upper_bound):
    heap = []
    for i, at in enumerate(a_t):
        j = int(np.searchsorted(b_t, at + upper_bound, side="left")) - 1
        while j >= 0 and omit_zero_lags and b_t[j] == at:
            j -= 1
        if j >= 0:
            lag = float(b_t[j] - at)
            if lag < upper_bound:
                heapq.heappush(heap, (-lag, i, j))

    while heap:
        neg_lag, i, j = heapq.heappop(heap)
        lag = -neg_lag
        yield lag, i, j

        j -= 1
        while j >= 0 and omit_zero_lags and b_t[j] == a_t[i]:
            j -= 1
        if j >= 0:
            next_lag = float(b_t[j] - a_t[i])
            if next_lag < upper_bound:
                heapq.heappush(heap, (-next_lag, i, j))


def _iter_acf_pairs_forward_stream(a_t, omit_zero_lags):
    heap = []
    for i, at in enumerate(a_t):
        j = i + int(omit_zero_lags)
        if j < len(a_t):
            heapq.heappush(heap, (float(a_t[j] - at), i, j))

    while heap:
        lag, i, j = heapq.heappop(heap)
        yield lag, i, j

        j += 1
        if j < len(a_t):
            heapq.heappush(heap, (float(a_t[j] - a_t[i]), i, j))


def _count_pairs_leq_ccf(a_t, b_t, x):
    return int(np.searchsorted(b_t, a_t + x, side="right").sum())


def _min_lag_geq_ccf(a_t, b_t, x, omit_zero_lags):
    best = None
    b_n = len(b_t)
    starts = np.searchsorted(b_t, a_t + x, side="left")
    for i, j in enumerate(starts):
        j = int(j)
        while j < b_n and omit_zero_lags and b_t[j] == a_t[i]:
            j += 1
        if j >= b_n:
            continue
        lag = float(b_t[j] - a_t[i])
        if best is None or lag < best:
            best = lag
    if best is None:
        raise RuntimeError("Unable to determine streaming median lag.")
    return best


def _median_lag_value_streaming(a_t, b_t, omit_zero_lags):
    zero_pairs = _count_zero_pairs(a_t, b_t) if omit_zero_lags else 0
    total_pairs = len(a_t) * len(b_t) - zero_pairs
    if total_pairs <= 0:
        raise ValueError(
            "No valid pairs available for binning. "
            "Check the input light curves and omit_zero_lags setting."
        )

    low = float(b_t[0] - a_t[-1])
    high = float(b_t[-1] - a_t[0])
    target = total_pairs // 2

    for _ in range(60):
        mid = (low + high) / 2.0
        count = _count_pairs_leq_ccf(a_t, b_t, mid)
        if omit_zero_lags and mid >= 0:
            count -= zero_pairs
        if count > target:
            high = mid
        else:
            low = mid

    return _min_lag_geq_ccf(a_t, b_t, high, omit_zero_lags)


def _build_bin(lags, a_idx, b_idx, uniform_sampling, minpts):
    n = len(lags)
    if uniform_sampling:
        if n < minpts:
            return None
        tau = float(np.mean(lags))
        return ZDCFBin(
            tau=tau,
            sigtm=0.0,
            sigtp=0.0,
            a_idx=np.asarray(a_idx, dtype=np.int32),
            b_idx=np.asarray(b_idx, dtype=np.int32),
        )

    if n < 2:
        return None

    lags_arr = np.sort(np.asarray(lags, dtype=np.float64))
    tau = float(np.mean(lags_arr))
    j = int(np.searchsorted(lags_arr, tau, side="left"))
    if j >= n:
        j = n - 1
    p = max(j - round(float(j) * 0.3413 * 2), 0)
    sigtm = max(tau - float(lags_arr[p]), 0.0)
    p = min(j + round(float((n - 1) - j) * 0.3413 * 2), n - 1)
    sigtp = max(float(lags_arr[p]) - tau, 0.0)

    return ZDCFBin(
        tau=tau,
        sigtm=sigtm,
        sigtp=sigtp,
        a_idx=np.asarray(a_idx, dtype=np.int32),
        b_idx=np.asarray(b_idx, dtype=np.int32),
    )


def _consume_bins(pair_iter, a_n, b_n, config, tolrnc):
    bins = []
    pending = next(pair_iter, None)

    while pending is not None:
        a_used = np.zeros(a_n, dtype=bool)
        b_used = np.zeros(b_n, dtype=bool)
        lags = []
        a_idx = []
        b_idx = []
        last_lag = None

        while pending is not None:
            lag, ai, bi = pending
            if (
                last_lag is not None
                and abs(lag - last_lag) > tolrnc
                and (len(lags) >= config.minpts or config.uniform_sampling)
            ):
                break

            pending = next(pair_iter, None)

            if a_used[ai] or b_used[bi]:
                continue

            a_used[ai] = True
            b_used[bi] = True
            lags.append(float(lag))
            a_idx.append(int(ai))
            b_idx.append(int(bi))
            last_lag = float(lag)

        zdcf_bin = _build_bin(
            lags=lags,
            a_idx=a_idx,
            b_idx=b_idx,
            uniform_sampling=config.uniform_sampling,
            minpts=config.minpts,
        )
        if zdcf_bin is not None:
            bins.append(zdcf_bin)

        if pending is None:
            break

    return bins


def _use_streaming_binner(sparse, a_n, b_n):
    if sparse is True:
        return True
    if sparse is False:
        return False
    return max(a_n, b_n) > STREAMING_THRESHOLD


def alcbin(a, b, config, sparse="auto", verbose=True):
    a_n = len(a.t)
    b_n = len(b.t)
    a_t = a["t"].to_numpy(dtype=np.float64)
    b_t = b["t"].to_numpy(dtype=np.float64)

    total_pairs = _total_possible_pairs(a_t, b_t, config)
    if total_pairs == 0:
        raise ValueError(
            "No valid pairs available for binning. "
            "Check the input light curves and omit_zero_lags setting."
        )

    sparse = _validate_sparse_mode(sparse)
    use_streaming = _use_streaming_binner(sparse, a_n, b_n)

    if verbose:
        print(
            f"\nBinning with minimum of {config.minpts} points "
            f"per bin and resolution of {RSLUTN}\n"
        )
        if use_streaming:
            print("using streaming lag iterator...")

    if use_streaming:
        tolrnc = _lag_tolerance(a_t, b_t, config)
        if config.autocf:
            pair_iter = _iter_acf_pairs_forward_stream(
                a_t, omit_zero_lags=config.omit_zero_lags
            )
            bins = _consume_bins(pair_iter, a_n, b_n, config, tolrnc)
        elif config.uniform_sampling:
            pair_iter = _iter_ccf_pairs_forward_stream(
                a_t,
                b_t,
                omit_zero_lags=config.omit_zero_lags,
                lower_bound=None,
            )
            bins = _consume_bins(pair_iter, a_n, b_n, config, tolrnc)
        else:
            median_tau = _median_lag_value_streaming(
                a_t, b_t, omit_zero_lags=config.omit_zero_lags
            )
            neg_iter = _iter_ccf_pairs_backward_stream(
                a_t,
                b_t,
                omit_zero_lags=config.omit_zero_lags,
                upper_bound=median_tau,
            )
            pos_iter = _iter_ccf_pairs_forward_stream(
                a_t,
                b_t,
                omit_zero_lags=config.omit_zero_lags,
                lower_bound=median_tau,
            )
            neg_bins = _consume_bins(neg_iter, a_n, b_n, config, tolrnc)
            pos_bins = _consume_bins(pos_iter, a_n, b_n, config, tolrnc)
            bins = list(reversed(neg_bins)) + pos_bins
    else:
        lags, a_idx, b_idx = _dense_sorted_pairs(a_t, b_t, config)
        tolrnc = _lag_tolerance(a_t, b_t, config, sorted_lags=lags)
        if config.autocf or config.uniform_sampling:
            pair_iter = _iter_dense_indices(
                lags, a_idx, b_idx, range(0, len(lags))
            )
            bins = _consume_bins(pair_iter, a_n, b_n, config, tolrnc)
        else:
            median_tau = lags[len(lags) // 2]
            split = int(np.searchsorted(lags, median_tau, side="left"))
            neg_iter = _iter_dense_indices(
                lags, a_idx, b_idx, range(split - 1, -1, -1)
            )
            pos_iter = _iter_dense_indices(
                lags, a_idx, b_idx, range(split, len(lags))
            )
            neg_bins = _consume_bins(neg_iter, a_n, b_n, config, tolrnc)
            pos_bins = _consume_bins(pos_iter, a_n, b_n, config, tolrnc)
            bins = list(reversed(neg_bins)) + pos_bins

    if not bins:
        raise RuntimeError("alcbin: nbins = 0")

    return bins


def dcf_pairs(dcf_inbin_or_bins, dcf_nbins, config, a_n, b_n):
    if (
        isinstance(dcf_inbin_or_bins, (list, tuple))
        and len(dcf_inbin_or_bins) > 0
        and isinstance(dcf_inbin_or_bins[0], ZDCFBin)
    ):
        used_pairs = int(sum(len(zbin.a_idx) for zbin in dcf_inbin_or_bins))
    else:
        used_pairs = int(np.sum(dcf_inbin_or_bins[:dcf_nbins]))

    if config.autocf:
        if config.omit_zero_lags:
            total_pairs = (a_n * (a_n - 1)) // 2
        else:
            total_pairs = (a_n * (a_n + 1)) // 2
    else:
        total_pairs = a_n * b_n

    return total_pairs - used_pairs


def read_obs(
    fname,
    input_dir="./",
    output_dir="./",
    out_name="light_curve.lc",
    sep=",",
    savelc=False,
):
    input_path = _build_path(input_dir, fname)
    lc = pd.read_table(input_path, header=None, names=["t", "flux", "err"], sep=sep)
    if lc.empty:
        raise ValueError(f"No data found in {input_path}.")

    lc = _prepare_light_curve(lc)

    if savelc:
        if output_dir is None:
            raise ValueError("output_dir must be provided when savelc=True.")
        output_path = _build_path(output_dir, out_name)
        lc.to_csv(output_path, index=False)
        print(f"{out_name} written (contains {len(lc.t)} points)")

    return lc


def simerr(flux, err, rng=None):
    return flux - (err * rndnrm(len(err), rng=rng))


def rndnrm(n, rng=None):
    if rng is None:
        return np.random.standard_normal(size=n)
    if hasattr(rng, "standard_normal"):
        return rng.standard_normal(size=n)
    return np.random.standard_normal(size=n)


def fishs(r, n):
    fish = (
        1.0
        / (n - 1)
        * (
            1
            + (4 - r**2) / 2.0 / (n - 1)
            + (22 - (6 * r**2) - 3 * (r**4)) / 6 / (n - 1) ** 2
        )
    )
    return np.sqrt(np.maximum(fish, 0))


def fishe(r, n):
    return np.log((1 + r) / (1 - r)) / 2.0 + r / 2 / (n - 1) * (
        1
        + (5 + r**2) / 4 / (n - 1)
        + (11 + 2 * r**2 + 3 * r**4) / 8 / (n - 1) ** 2
    )


def clcdcf(a, b, bins, MC=False):
    dcf_nbins = len(bins)
    dcf_t = np.asarray([zbin.tau for zbin in bins], dtype=np.float64)
    dcf_sigtm = np.asarray([zbin.sigtm for zbin in bins], dtype=np.float64)
    dcf_sigtp = np.asarray([zbin.sigtp for zbin in bins], dtype=np.float64)
    dcf_inbin = np.asarray([len(zbin.a_idx) for zbin in bins], dtype=np.int32)

    dcf_r = np.zeros(dcf_nbins, dtype=np.float64)
    dcf_sigrm = np.zeros(dcf_nbins, dtype=np.float64)
    dcf_sigrp = np.zeros(dcf_nbins, dtype=np.float64)

    if MC:
        a_flux = a["MC"].to_numpy(dtype=np.float64)
        b_flux = b["MC"].to_numpy(dtype=np.float64)
    else:
        a_flux = a["flux"].to_numpy(dtype=np.float64)
        b_flux = b["flux"].to_numpy(dtype=np.float64)

    for ibin, zbin in enumerate(bins):
        n = len(zbin.a_idx)
        if n < 2:
            continue

        wa_x = a_flux[zbin.a_idx]
        wb_x = b_flux[zbin.b_idx]

        wa_centered = wa_x - np.mean(wa_x)
        wb_centered = wb_x - np.mean(wb_x)
        vara = float(np.dot(wa_centered, wa_centered) / (n - 1))
        varb = float(np.dot(wb_centered, wb_centered) / (n - 1))
        vnorm = vara * varb

        if vnorm <= 0:
            expbin = 0.0
            dcf_r[ibin] = 0.0
        else:
            cov = float(np.dot(wa_centered, wb_centered) / (n - 1))
            expbin = _clamp_corr(cov / np.sqrt(vnorm))
            dcf_r[ibin] = expbin

        sigz = fishs(expbin, n)
        expz = fishe(expbin, n)
        dcf_sigrm[ibin] = expbin - np.tanh(expz - sigz)
        dcf_sigrp[ibin] = np.tanh(expz + sigz) - expbin

    dcf_taus = (dcf_t, dcf_sigtm, dcf_sigtp)
    dcf_rs = (dcf_r, dcf_sigrm, dcf_sigrp)
    dcf_other = (dcf_inbin, dcf_nbins)

    return dcf_taus, dcf_rs, dcf_other


def check_user_input(usr_input, name="value", allow_float=False):
    if usr_input is None:
        raise ValueError(f"Invalid {name}. Please enter a number.")

    if isinstance(usr_input, bool):
        raise ValueError(f"Invalid {name}. Please enter a number.")

    if isinstance(usr_input, int):
        return usr_input

    if isinstance(usr_input, float):
        if allow_float:
            return usr_input
        if usr_input.is_integer():
            return int(usr_input)
        raise ValueError(f"Invalid {name}. Please enter a whole number.")

    try:
        return int(str(usr_input).strip())
    except (ValueError, TypeError):
        try:
            val = float(str(usr_input).strip())
        except (ValueError, TypeError):
            raise ValueError(f"Invalid {name}. Please enter a number.") from None
        if allow_float:
            return val
        if val.is_integer():
            return int(val)
        raise ValueError(f"Invalid {name}. Please enter a whole number.") from None


def user_input(interactive=True, verbose=True, parameters=None):
    if parameters is None:
        parameters = {}

    if verbose:
        print("\npyZDCF begins:\n")

    if interactive:
        def prompt_choice(prompt, choices, name):
            while True:
                value = input(prompt).strip()
                if value in choices:
                    return choices[value]
                print(
                    f"Invalid {name}. Please enter one of: {', '.join(choices.keys())}."
                )

        def prompt_yes_no(prompt, name):
            while True:
                value = input(prompt).strip().lower()
                if value in ("y", "yes"):
                    return True
                if value in ("n", "no"):
                    return False
                print(f"Invalid {name}. Please enter y/n.")

        def prompt_int(prompt, name):
            while True:
                try:
                    return check_user_input(input(prompt), name=name)
                except ValueError as exc:
                    print(exc)

        Autocf = prompt_choice(
            "Auto-correlation or cross-correlation? (1/2): ",
            {"1": True, "2": False},
            "correlation mode",
        )
        Prefix = _validate_text(input("Enter output files prefix: "), "prefix")
        UniSample = prompt_yes_no("Uniform sampling of light curve? (y/n): ", "sampling")
        Minpts = prompt_int(
            "Enter minimal number of points per bin (0 for default): ", "minpts"
        )
        if Minpts <= 0:
            Minpts = ENOUGH
        if Minpts < 2:
            raise ValueError("Invalid minpts. Please enter a whole number >= 2.")
        NoZeroLag = prompt_yes_no("Omit zero-lag points? (y/n): ", "zero-lag option")
        nMC = prompt_int("How many Monte Carlo runs for error estimation? ", "num_MC")
        if nMC <= 1:
            nMC = 0
        Name1 = _validate_text(input("Enter name of 1st light curve file: "), "lc1_name")
        if Autocf:
            Name2 = Name1
        else:
            Name2 = _validate_text(
                input("Enter name of 2nd light curve file: "), "lc2_name"
            )
    else:
        input_keys = [
            "autocf",
            "prefix",
            "uniform_sampling",
            "omit_zero_lags",
            "minpts",
            "num_MC",
            "lc1_name",
        ]

        for key in input_keys:
            if key not in parameters:
                raise KeyError(f"Missing key in input dictionary 'parameters': {key}")

        Autocf = _validate_bool(parameters["autocf"], "autocf")
        Prefix = _validate_text(parameters["prefix"], "prefix")
        UniSample = _validate_bool(parameters["uniform_sampling"], "uniform_sampling")
        NoZeroLag = _validate_bool(parameters["omit_zero_lags"], "omit_zero_lags")
        Minpts = check_user_input(parameters["minpts"], name="minpts")
        nMC = check_user_input(parameters["num_MC"], name="num_MC")
        Name1 = _validate_text(parameters["lc1_name"], "lc1_name")

        if Autocf:
            Name2 = Name1
        else:
            if "lc2_name" not in parameters:
                raise KeyError("Missing key in input dictionary 'parameters': lc2_name")
            Name2 = _validate_text(parameters["lc2_name"], "lc2_name")

        if Minpts <= 0:
            Minpts = ENOUGH
        if Minpts < 2:
            raise ValueError("Invalid minpts. Please enter a whole number >= 2.")
        if nMC <= 1:
            nMC = 0

    if verbose:
        print(30 * "=")
        print("pyZDCF PARAMETERS:\n")
        print("Autocorrelation?  ", Autocf)
        print("Uniform sampling? ", UniSample)
        print("Omit zero lags?   ", NoZeroLag)
        print("Minimal # in bin: ", Minpts)
        print("# of Monte Carlo: ", nMC)
        print("Monte Carlo seed: ", ISEED)
        print(30 * "=")

    return Autocf, Prefix, UniSample, NoZeroLag, Minpts, nMC, Name1, Name2


def _compute_zdcf(a, b, config, nMC, verbose=True, sparse="auto", output_path=None):
    a_n = len(a.t)
    b_n = len(b.t)
    bins = alcbin(a, b, config, sparse=sparse, verbose=verbose)

    rng = np.random.RandomState(ISEED)

    if nMC > 1:
        dcf_unused = _unused_pairs_from_bins(
            bins,
            a["t"].to_numpy(dtype=np.float64),
            b["t"].to_numpy(dtype=np.float64),
            config,
        )
        dcf_avz = None

        for i in range(nMC):
            a["MC"] = simerr(a.flux.to_numpy(dtype=np.float64), a.err.to_numpy(dtype=np.float64), rng=rng)
            b["MC"] = simerr(b.flux.to_numpy(dtype=np.float64), b.err.to_numpy(dtype=np.float64), rng=rng)
            dcf_taus, dcf_rs, dcf_other = clcdcf(a, b, bins, MC=True)
            dcf_r = _clamp_corr(dcf_rs[0])
            dcf_z = np.arctanh(dcf_r)
            if dcf_avz is None:
                dcf_avz = np.zeros_like(dcf_z)
                if verbose:
                    print(
                        f"{dcf_other[1]} bins actually used, {dcf_unused} inter-dependent pairs discarded."
                    )
            dcf_avz = dcf_avz + dcf_z

        dcf_avz = dcf_avz / nMC
        dcf_r = _clamp_corr(np.tanh(dcf_avz))

        dcf_inbin = dcf_other[0]
        dcf_expz = fishe(dcf_r, dcf_inbin)
        dcf_sigz = fishs(dcf_r, dcf_inbin)
        dcf_sigrm = dcf_r - np.tanh(dcf_expz - dcf_sigz)
        dcf_sigrp = np.tanh(dcf_expz + dcf_sigz) - dcf_r
        dcf_t, dcf_sigtm, dcf_sigtp = dcf_taus
    else:
        dcf_taus, dcf_rs, dcf_other = clcdcf(a, b, bins)
        dcf_t, dcf_sigtm, dcf_sigtp = dcf_taus
        dcf_r, dcf_sigrm, dcf_sigrp = dcf_rs
        if verbose:
            dcf_unused = _unused_pairs_from_bins(
                bins,
                a["t"].to_numpy(dtype=np.float64),
                b["t"].to_numpy(dtype=np.float64),
                config,
            )
            print(
                f"{dcf_other[1]} bins actually used, {int(dcf_unused)} inter-dependent pairs discarded."
            )

    d = np.column_stack(
        (dcf_t, dcf_sigtm, dcf_sigtp, dcf_r, dcf_sigrm, dcf_sigrp, dcf_other[0])
    )

    if output_path is not None:
        np.savetxt(
            output_path,
            d,
            fmt=[
                "%11.3e",
                "%11.3e",
                "%11.3e",
                "%11.3e",
                "%11.3e",
                "%11.3e",
                "%5.1i",
            ],
        )

    dcf_df = pd.DataFrame(
        data=d,
        columns=[
            "tau",
            "-sig(tau)",
            "+sig(tau)",
            "dcf",
            "-err(dcf)",
            "+err(dcf)",
            "#bin",
        ],
    )

    return dcf_df


def pyzdcf_from_dataframes(
    light_curve1,
    light_curve2=None,
    *,
    autocf=False,
    prefix="ccf",
    uniform_sampling=False,
    omit_zero_lags=True,
    minpts=0,
    num_MC=100,
    verbose=True,
    sparse="auto",
    output_path=None,
):
    autocf = _validate_bool(autocf, "autocf")
    uniform_sampling = _validate_bool(uniform_sampling, "uniform_sampling")
    omit_zero_lags = _validate_bool(omit_zero_lags, "omit_zero_lags")
    sparse = _validate_sparse_mode(sparse)
    _validate_text(prefix, "prefix")

    minpts = check_user_input(minpts, name="minpts")
    num_MC = check_user_input(num_MC, name="num_MC")
    if minpts <= 0:
        minpts = ENOUGH
    if minpts < 2:
        raise ValueError("Invalid minpts. Please enter a whole number >= 2.")
    if num_MC <= 1:
        num_MC = 0

    a = _prepare_light_curve(light_curve1)
    if autocf:
        b = a.copy()
    else:
        if light_curve2 is None:
            raise ValueError("light_curve2 must be provided when autocf=False.")
        b = _prepare_light_curve(light_curve2)

    config = ZDCFConfig(
        autocf=autocf,
        uniform_sampling=uniform_sampling,
        omit_zero_lags=omit_zero_lags,
        minpts=minpts,
    )

    return _compute_zdcf(
        a=a,
        b=b,
        config=config,
        nMC=num_MC,
        verbose=verbose,
        sparse=sparse,
        output_path=output_path,
    )


def pyzdcf(
    input_dir,
    output_dir=None,
    intr=True,
    sep=",",
    verbose=True,
    sparse="auto",
    savelc=False,
    parameters=None,
):
    sparse = _validate_sparse_mode(sparse)

    if parameters is None:
        parameters = {
            "autocf": False,
            "prefix": "ccf",
            "uniform_sampling": False,
            "omit_zero_lags": True,
            "minpts": 0,
            "num_MC": 100,
            "lc1_name": "lc1_example",
            "lc2_name": "lc2_example",
        }

    if intr:
        Autocf, Prefix, UniSample, NoZeroLag, Minpts, nMC, Name1, Name2 = user_input(
            interactive=True, verbose=verbose
        )
    else:
        Autocf, Prefix, UniSample, NoZeroLag, Minpts, nMC, Name1, Name2 = user_input(
            interactive=False, parameters=parameters, verbose=verbose
        )

    config = ZDCFConfig(
        autocf=Autocf,
        uniform_sampling=UniSample,
        omit_zero_lags=NoZeroLag,
        minpts=Minpts,
    )

    a = read_obs(
        Name1, input_dir, output_dir, out_name=Prefix + ".lc1", sep=sep, savelc=savelc
    )
    if Autocf:
        b = a.copy()
    else:
        b = read_obs(
            Name2,
            input_dir,
            output_dir,
            out_name=Prefix + ".lc2",
            sep=sep,
            savelc=savelc,
        )

    output_path = None
    if output_dir is not None:
        output_path = _build_path(output_dir, Prefix + ".dcf")

    dcf_df = _compute_zdcf(
        a=a,
        b=b,
        config=config,
        nMC=nMC,
        verbose=verbose,
        sparse=sparse,
        output_path=output_path,
    )

    if verbose and output_path is not None:
        print("\n")
        print(Prefix + ".dcf written...")
        print("\npyZDCF ended.")
        print("")

    return dcf_df


if __name__ == "__main__":
    input_dir = input("Enter the path to the directory containing input data: ")
    output_dir = input("Enter the path to the directory for storing the results: ")
    dcf_df = pyzdcf(input_dir, output_dir, intr=True, sep=",")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(dcf_df)
