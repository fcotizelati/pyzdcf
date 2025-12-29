import pytest

from pyzdcf.pyzdcf import check_user_input, pyzdcf, read_obs, user_input


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
