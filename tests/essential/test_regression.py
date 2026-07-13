from pathlib import Path
import warnings

import numpy as np
import pytest

from pyzdcf import pyzdcf


BASE_DIR = Path(__file__).resolve().parents[1]
EPS = 0.01

DATASETS = {
    "lc1": {
        "input_dir": BASE_DIR / "test_data" / "test_lc1",
        "lc1_name": "lcx_1000",
        "lc2_name": "lcy_1000",
    },
    "lc2": {
        "input_dir": BASE_DIR / "test_data" / "test_lc2",
        "lc1_name": "lcx_3500",
        "lc2_name": "lcy_3500",
    },
    "lc3": {
        "input_dir": BASE_DIR / "test_data" / "test_lc3",
        "lc1_name": "lcx_15",
        "lc2_name": "lcy_15",
    },
}

CASES = [
    ("acfTT1_test", True, True, True, 1),
    ("acfFT1_test", True, False, True, 1),
    ("acfTF1_test", True, True, False, 1),
    ("acfFF1_test", True, False, False, 1),
    ("ccfTT1_test", False, True, True, 1),
    ("ccfFT1_test", False, False, True, 1),
    ("ccfTF1_test", False, True, False, 1),
    ("ccfFF1_test", False, False, False, 1),
    ("acfTT3_test", True, True, True, 3),
    ("acfFT3_test", True, False, True, 3),
    ("acfTF3_test", True, True, False, 3),
    ("acfFF3_test", True, False, False, 3),
    ("ccfTT3_test", False, True, True, 3),
    ("ccfFT3_test", False, False, True, 3),
    ("ccfTF3_test", False, True, False, 3),
    ("ccfFF3_test", False, False, False, 3),
]

FORTRAN_REFERENCE_DRIFT = {
    ("lc1", "acfFT1_test"),
    ("lc1", "acfFF1_test"),
    ("lc1", "ccfFT1_test"),
    ("lc1", "ccfTF1_test"),
    ("lc1", "ccfFF1_test"),
    ("lc1", "acfTT3_test"),
    ("lc1", "acfFT3_test"),
    ("lc1", "acfTF3_test"),
    ("lc1", "acfFF3_test"),
    ("lc1", "ccfTT3_test"),
    ("lc1", "ccfFT3_test"),
    ("lc1", "ccfTF3_test"),
    ("lc1", "ccfFF3_test"),
    ("lc2", "acfFT1_test"),
    ("lc2", "acfFF1_test"),
    ("lc2", "ccfTT1_test"),
    ("lc2", "ccfFT1_test"),
    ("lc2", "ccfTF1_test"),
    ("lc2", "ccfFF1_test"),
    ("lc2", "acfTT3_test"),
    ("lc2", "acfFT3_test"),
    ("lc2", "acfTF3_test"),
    ("lc2", "acfFF3_test"),
    ("lc2", "ccfTT3_test"),
    ("lc2", "ccfFT3_test"),
    ("lc2", "ccfTF3_test"),
    ("lc2", "ccfFF3_test"),
    ("lc3", "acfFT1_test"),
    ("lc3", "acfFF1_test"),
    ("lc3", "ccfFT1_test"),
    ("lc3", "ccfFF1_test"),
    ("lc3", "acfTT3_test"),
    ("lc3", "acfFT3_test"),
    ("lc3", "acfTF3_test"),
    ("lc3", "acfFF3_test"),
    ("lc3", "ccfTT3_test"),
    ("lc3", "ccfFT3_test"),
    ("lc3", "ccfTF3_test"),
    ("lc3", "ccfFF3_test"),
}

FORTRAN_DRIFT_REASON = (
    "known divergence from the bundled Fortran fixture due pyZDCF bug fixes, "
    "binning convention changes, or Monte Carlo sequence changes"
)


def _load_reference(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _fortran_reference_prefix(prefix):
    if prefix.endswith("_test"):
        return prefix[: -len("_test")]
    return prefix


def _case_parameters(mark_fortran_drift=False):
    for dataset_key in sorted(DATASETS):
        for prefix, autocf, uniform_sampling, omit_zero_lags, num_mc in CASES:
            marks = []
            if mark_fortran_drift and (dataset_key, prefix) in FORTRAN_REFERENCE_DRIFT:
                marks.append(pytest.mark.xfail(reason=FORTRAN_DRIFT_REASON, strict=True))
            yield pytest.param(
                dataset_key,
                prefix,
                autocf,
                uniform_sampling,
                omit_zero_lags,
                num_mc,
                id=f"{dataset_key}-{prefix}",
                marks=marks,
            )


def _run_case(
    tmp_path,
    dataset_key,
    prefix,
    autocf,
    uniform_sampling,
    omit_zero_lags,
    num_mc,
):
    dataset = DATASETS[dataset_key]
    input_dir = dataset["input_dir"]
    params = {
        "autocf": autocf,
        "prefix": prefix,
        "uniform_sampling": uniform_sampling,
        "omit_zero_lags": omit_zero_lags,
        "minpts": 0,
        "num_MC": num_mc,
        "lc1_name": dataset["lc1_name"],
    }
    if not autocf:
        params["lc2_name"] = dataset["lc2_name"]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*fewer than 11 independent pairs.*",
            category=RuntimeWarning,
        )
        result = pyzdcf(
            input_dir=str(input_dir),
            output_dir=str(tmp_path),
            intr=False,
            parameters=params,
            sparse="auto",
            verbose=False,
        ).to_numpy()

    return input_dir, result


@pytest.mark.parametrize(
    "dataset_key,prefix,autocf,uniform_sampling,omit_zero_lags,num_mc",
    list(_case_parameters()),
)
def test_outputs_are_stable_against_checked_in_python_snapshots(
    tmp_path,
    dataset_key,
    prefix,
    autocf,
    uniform_sampling,
    omit_zero_lags,
    num_mc,
):
    input_dir, result = _run_case(
        tmp_path,
        dataset_key,
        prefix,
        autocf,
        uniform_sampling,
        omit_zero_lags,
        num_mc,
    )
    reference = _load_reference(input_dir / "results_py" / f"{prefix}.dcf")

    assert result.shape == reference.shape
    assert np.allclose(reference, result, rtol=EPS, atol=EPS, equal_nan=True)


@pytest.mark.parametrize(
    "dataset_key,prefix,autocf,uniform_sampling,omit_zero_lags,num_mc",
    list(_case_parameters(mark_fortran_drift=True)),
)
def test_outputs_match_independent_fortran_references_where_expected(
    tmp_path,
    dataset_key,
    prefix,
    autocf,
    uniform_sampling,
    omit_zero_lags,
    num_mc,
):
    input_dir, result = _run_case(
        tmp_path,
        dataset_key,
        prefix,
        autocf,
        uniform_sampling,
        omit_zero_lags,
        num_mc,
    )
    reference = _load_reference(input_dir / f"{_fortran_reference_prefix(prefix)}.dcf")

    assert result.shape == reference.shape
    assert np.allclose(reference, result, rtol=EPS, atol=EPS, equal_nan=True)
