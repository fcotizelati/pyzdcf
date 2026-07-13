import warnings

import numpy as np
import pandas as pd
import pytest

from pyzdcf.pyzdcf import (
    ZDCFConfig,
    alcbin,
    check_user_input,
    clcdcf,
    dcf_pairs,
    pyzdcf,
    pyzdcf_from_dataframes,
    read_obs,
    user_input,
    ZDCFBin,
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


def test_read_obs_combines_duplicate_times_with_inverse_variance_weights(tmp_path):
    data_path = tmp_path / "lc.csv"
    write_lc(data_path, ["0,10.0,1.0", "0,20.0,2.0", "1,30.0,1.0"])

    df = read_obs("lc.csv", input_dir=tmp_path, sep=",")

    assert len(df) == 2
    assert np.isclose(df.loc[0, "flux"], 12.0)
    assert np.isclose(df.loc[0, "err"], np.sqrt(1 / 1.25))


def test_read_obs_rejects_negative_uncertainty(tmp_path):
    data_path = tmp_path / "bad_err.csv"
    write_lc(data_path, ["0,1.0,-0.1", "1,2.0,0.1"])

    with pytest.raises(ValueError, match="non-negative"):
        read_obs("bad_err.csv", input_dir=tmp_path, sep=",")


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


def test_user_input_rejects_minpts_below_two():
    params = {
        "autocf": True,
        "prefix": "out",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 1,
        "num_MC": 0,
        "lc1_name": "lc.csv",
    }

    with pytest.raises(ValueError, match=">= 2"):
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


def test_user_input_rejects_non_boolean_flags():
    params = {
        "autocf": "False",
        "prefix": "out",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 2,
        "num_MC": 0,
        "lc1_name": "lc.csv",
    }

    with pytest.raises(ValueError, match="True/False"):
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

    with pytest.warns(RuntimeWarning, match="below the ZDCF recommended minimum"):
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
    write_lc(tmp_path / "a.csv", ["9999,1.0,0.1", "10000,1.1,0.1", "10001,1.2,0.1"])
    write_lc(tmp_path / "b.csv", ["1,2.0,0.1", "2,2.1,0.1"])

    params = {
        "autocf": False,
        "prefix": "sentinel",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 2,
        "num_MC": 1,
        "lc1_name": "a.csv",
        "lc2_name": "b.csv",
    }

    with pytest.warns(RuntimeWarning, match="below the ZDCF recommended minimum"):
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


def test_pyzdcf_rejects_minpts_equal_one(tmp_path):
    write_lc(tmp_path / "lc.csv", ["0,1.0,0.1", "1,1.1,0.1", "2,0.9,0.1"])

    params = {
        "autocf": True,
        "prefix": "bad_minpts",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 1,
        "num_MC": 0,
        "lc1_name": "lc.csv",
    }

    with pytest.raises(ValueError, match=">= 2"):
        pyzdcf(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
            intr=False,
            parameters=params,
            sparse=False,
            verbose=False,
        )


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

    with pytest.warns(RuntimeWarning, match="below the ZDCF recommended minimum"):
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
    with pytest.warns(RuntimeWarning, match="below the ZDCF recommended minimum"):
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


def test_pyzdcf_rejects_invalid_sparse_mode(tmp_path):
    write_lc(tmp_path / "lc.csv", ["0,1.0,0.1", "1,1.1,0.1", "2,0.9,0.1"])

    params = {
        "autocf": True,
        "prefix": "bad_sparse",
        "uniform_sampling": True,
        "omit_zero_lags": False,
        "minpts": 2,
        "num_MC": 0,
        "lc1_name": "lc.csv",
    }

    with pytest.raises(ValueError, match="Invalid sparse mode"):
        pyzdcf(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
            intr=False,
            parameters=params,
            sparse="yes",
            verbose=False,
        )


def test_alcbin_keeps_last_pair_when_minpts_one():
    lc = pd.DataFrame({"t": [0, 1], "flux": [1.0, 2.0], "err": [0.1, 0.1]})
    config = ZDCFConfig(
        autocf=True, uniform_sampling=True, omit_zero_lags=False, minpts=1
    )

    bins = alcbin(lc, lc, config, sparse=False, verbose=False)

    assert [(zbin.tau, len(zbin.a_idx)) for zbin in bins] == [(0.0, 2), (1.0, 1)]


def test_alcbin_does_not_split_zero_lag_plateau():
    lc = pd.DataFrame({"t": [0, 1, 2], "flux": [1.0, 2.0, 3.0], "err": [0.1] * 3})
    config = ZDCFConfig(
        autocf=False, uniform_sampling=False, omit_zero_lags=False, minpts=1
    )

    bins = alcbin(lc, lc, config, sparse=False, verbose=False)

    assert sum(np.isclose(zbin.tau, 0.0) for zbin in bins) == 1


def test_clcdcf_is_numerically_stable_for_large_flux_offsets():
    flux = np.array([1e12, 1e12 + 1, 1e12 + 2], dtype=float)
    lc = pd.DataFrame({"t": [0, 1, 2], "flux": flux, "err": [0.1, 0.1, 0.1]})
    bins = [
        ZDCFBin(
            tau=0.0,
            sigtm=0.0,
            sigtp=0.0,
            a_idx=np.array([0, 1, 2], dtype=np.int32),
            b_idx=np.array([0, 1, 2], dtype=np.int32),
        )
    ]

    _, dcf_rs, _ = clcdcf(lc, lc, bins, MC=False)

    assert np.isclose(dcf_rs[0][0], 1.0, rtol=1e-6)


def test_clcdcf_marks_zero_variance_correlation_as_nan():
    lc = pd.DataFrame(
        {"t": [0, 1, 2], "flux": [5.0, 5.0, 5.0], "err": [0.1, 0.1, 0.1]}
    )
    bins = [
        ZDCFBin(
            tau=0.0,
            sigtm=0.0,
            sigtp=0.0,
            a_idx=np.array([0, 1, 2], dtype=np.int32),
            b_idx=np.array([0, 1, 2], dtype=np.int32),
        )
    ]

    _, dcf_rs, _ = clcdcf(lc, lc, bins, MC=False)

    assert np.isnan(dcf_rs[0][0])
    assert np.isnan(dcf_rs[1][0])
    assert np.isnan(dcf_rs[2][0])


def test_autocf_monte_carlo_uses_one_perturbed_curve_at_zero_lag():
    t = np.arange(20, dtype=float)
    lc = pd.DataFrame(
        {
            "t": t,
            "flux": np.sin(t / 3.0),
            "err": np.full_like(t, 2.0),
        }
    )

    df = pyzdcf_from_dataframes(
        lc,
        autocf=True,
        uniform_sampling=True,
        omit_zero_lags=False,
        minpts=11,
        num_MC=100,
        verbose=False,
        sparse=False,
    )

    zero_lag = df.loc[np.isclose(df["tau"], 0.0)].iloc[0]
    assert np.isclose(zero_lag["dcf"], 1.0, rtol=1e-6)


def test_minpts_below_zdcf_recommendation_warns():
    t = np.arange(6, dtype=float)
    lc = pd.DataFrame(
        {"t": t, "flux": np.sin(t), "err": np.full_like(t, 0.1)}
    )

    with pytest.warns(RuntimeWarning, match="below the ZDCF recommended minimum"):
        pyzdcf_from_dataframes(
            lc,
            autocf=True,
            uniform_sampling=True,
            omit_zero_lags=False,
            minpts=2,
            num_MC=0,
            verbose=False,
            sparse=False,
        )


def test_default_minpts_warns_when_extreme_bin_has_too_few_pairs():
    t = np.arange(12, dtype=float)
    lc = pd.DataFrame(
        {"t": t, "flux": np.sin(t), "err": np.full_like(t, 0.1)}
    )

    with pytest.warns(RuntimeWarning, match="fewer than 11 independent pairs"):
        df = pyzdcf_from_dataframes(
            lc,
            autocf=True,
            uniform_sampling=False,
            omit_zero_lags=False,
            minpts=0,
            num_MC=0,
            verbose=False,
            sparse=False,
        )

    assert df["#bin"].min() < 11


def test_dense_and_streaming_paths_match():
    lc = pd.DataFrame(
        {
            "t": np.arange(8, dtype=float),
            "flux": np.sin(np.arange(8, dtype=float)),
            "err": np.full(8, 0.1),
        }
    )

    params = dict(
        autocf=False,
        uniform_sampling=False,
        omit_zero_lags=False,
        minpts=2,
        num_MC=0,
        verbose=False,
        output_path=None,
    )
    with pytest.warns(RuntimeWarning, match="below the ZDCF recommended minimum"):
        dense = pyzdcf_from_dataframes(
            lc, lc.copy(), prefix="dense", sparse=False, **params
        )
    with pytest.warns(RuntimeWarning, match="below the ZDCF recommended minimum"):
        streaming = pyzdcf_from_dataframes(
            lc, lc.copy(), prefix="stream", sparse=True, **params
        )

    assert np.allclose(dense.to_numpy(), streaming.to_numpy())
