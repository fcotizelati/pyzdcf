import warnings

import numpy as np
import pytest

from pyzdcf.pyzdcf import (
    ZDCFConfig,
    check_user_input,
    dcf_pairs,
    pyzdcf,
    read_obs,
    user_input,
)


def write_lc(path, rows):
    path.write_text("\n".join(rows) + "\n")


def test_check_user_input_rejects_non_integer():
    with pytest.raises(ValueError):
        check_user_input("1.5", name="minpts")


def test_check_user_input_accepts_integer_float_string():
    assert check_user_input("2.0", name="num_MC") == 2


def test_read_obs_rejects_all_nan(tmp_path):
    data_path = tmp_path / "nan_lc.csv"
    write_lc(data_path, ["0,nan,0.1", "1,nan,0.1"])

    with pytest.raises(ValueError):
        read_obs("nan_lc.csv", input_dir=tmp_path, sep=",")


def test_read_obs_rejects_partial_nan(tmp_path):
    data_path = tmp_path / "nan_lc.csv"
    write_lc(data_path, ["0,1.0,0.1", "1,2.0,nan"])

    with pytest.raises(ValueError):
        read_obs("nan_lc.csv", input_dir=tmp_path, sep=",")


def test_read_obs_accepts_dir_without_trailing_slash(tmp_path):
    data_path = tmp_path / "lc.csv"
    write_lc(data_path, ["0,1.0,0.1", "1,2.0,0.1"])

    df = read_obs("lc.csv", input_dir=str(tmp_path), sep=",")

    assert len(df) == 2


def test_user_input_rejects_non_integer_minpts():
    params = {
        "autocf": True,
        "prefix": "out",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": "1.5",
        "num_MC": 1,
        "lc1_name": "lc.csv",
    }

    with pytest.raises(ValueError):
        user_input(interactive=False, parameters=params, verbose=False)


def test_user_input_rejects_missing_num_mc():
    params = {
        "autocf": True,
        "prefix": "out",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 2,
        "lc1_name": "lc.csv",
    }

    with pytest.raises(KeyError, match="num_MC"):
        user_input(interactive=False, parameters=params, verbose=False)


def test_dcf_pairs_autocf_includes_zero_lag():
    config = ZDCFConfig(
        autocf=True, uniform_sampling=False, omit_zero_lags=False, minpts=2
    )

    # For ACF with zero-lag pairs included, total pairs are n*(n+1)/2.
    assert dcf_pairs(np.array([0], dtype=np.int32), 1, config, 3, 3) == 6


def test_pyzdcf_writes_output_without_trailing_slash(tmp_path):
    data_path = tmp_path / "lc.csv"
    write_lc(
        data_path,
        [
            "0,1.0,0.1",
            "1,1.1,0.1",
            "2,0.9,0.1",
            "3,1.2,0.1",
            "4,1.0,0.1",
            "5,0.95,0.1",
        ],
    )

    params = {
        "autocf": True,
        "prefix": "out",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 2,
        "num_MC": 1,
        "lc1_name": "lc.csv",
    }

    df = pyzdcf(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        intr=False,
        parameters=params,
        sparse=False,
        verbose=False,
    )

    assert (tmp_path / "out.dcf").exists()
    assert not df.empty


def test_pyzdcf_handles_lag_equal_negative_sentinel_value(tmp_path):
    write_lc(tmp_path / "a.csv", ["9999,1.0,0.1", "10000,1.1,0.1"])
    write_lc(tmp_path / "b.csv", ["1,2.0,0.1"])

    params = {
        "autocf": False,
        "prefix": "sentinel",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 1,
        "num_MC": 1,
        "lc1_name": "a.csv",
        "lc2_name": "b.csv",
    }

    df = pyzdcf(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        intr=False,
        parameters=params,
        sparse=False,
        verbose=False,
    )

    assert not df.empty
    assert np.isclose(df["tau"].iloc[0], -9999.0)


def test_pyzdcf_raises_clear_error_when_no_valid_pairs(tmp_path):
    write_lc(tmp_path / "lc.csv", ["0,1.0,0.1"])

    params = {
        "autocf": True,
        "prefix": "nopairs",
        "uniform_sampling": False,
        "omit_zero_lags": True,
        "minpts": 2,
        "num_MC": 0,
        "lc1_name": "lc.csv",
    }

    with pytest.raises(ValueError, match="No valid pairs available for binning"):
        pyzdcf(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
            intr=False,
            parameters=params,
            sparse=False,
            verbose=False,
        )


def test_pyzdcf_avoids_divide_by_zero_warning_on_empty_last_bin(tmp_path):
    write_lc(tmp_path / "a.csv", ["0,1.0,0.1"])
    write_lc(tmp_path / "b.csv", ["1,2.0,0.1"])

    params = {
        "autocf": False,
        "prefix": "warn",
        "uniform_sampling": False,
        "omit_zero_lags": False,
        "minpts": 2,
        "num_MC": 0,
        "lc1_name": "a.csv",
        "lc2_name": "b.csv",
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="alcbin: nbins = 0"):
            pyzdcf(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path),
                intr=False,
                parameters=params,
                sparse=False,
                verbose=False,
            )

    assert not any(
        "invalid value encountered in scalar divide" in str(w.message)
        for w in caught
    )


def test_pyzdcf_does_not_modify_global_numpy_rng_state(tmp_path):
    write_lc(
        tmp_path / "lc.csv",
        [
            "0,1.0,0.1",
            "1,1.1,0.1",
            "2,0.9,0.1",
            "3,1.2,0.1",
            "4,1.0,0.1",
            "5,0.95,0.1",
        ],
    )

    params = {
        "autocf": True,
        "prefix": "rng",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 2,
        "num_MC": 3,
        "lc1_name": "lc.csv",
    }

    np.random.seed(987654)
    expected_first = np.random.random()
    expected_second = np.random.random()

    np.random.seed(987654)
    observed_first = np.random.random()
    pyzdcf(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        intr=False,
        parameters=params,
        sparse=False,
        verbose=False,
    )
    observed_second = np.random.random()

    assert np.isclose(observed_first, expected_first)
    assert np.isclose(observed_second, expected_second)
