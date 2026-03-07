from pathlib import Path

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


def _load_reference(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


@pytest.mark.parametrize("dataset_key", sorted(DATASETS))
@pytest.mark.parametrize(
    "prefix,autocf,uniform_sampling,omit_zero_lags,num_mc",
    CASES,
)
def test_reference_outputs_match_checked_in_python_results(
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
    reference = _load_reference(input_dir / "results_py" / f"{prefix}.dcf")

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

    result = pyzdcf(
        input_dir=str(input_dir),
        output_dir=str(tmp_path),
        intr=False,
        parameters=params,
        sparse="auto",
        verbose=False,
    ).to_numpy()

    assert result.shape == reference.shape
    assert np.allclose(reference, result, rtol=EPS, atol=EPS)
