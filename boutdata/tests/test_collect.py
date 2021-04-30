from glob import glob
import numpy as np
import numpy.testing as npt
from pathlib import Path
import pytest

from boutdata.collect import collect
from boutdata.squashoutput import squashoutput

from boutdata.tests.make_test_data import (
    create_dump_file,
    concatenate_data,
    expected_attributes,
)

# Note - using tmp_path fixture requires pytest>=3.9.0

squash_kwargs = [
    {},
    {"compress": True, "complevel": 1},
    {"compress": True, "complevel": 9},
]


def check_collected_data(
    expected, *, fieldperp_global_yind, path, squash, collect_kwargs, squash_kwargs
):
    if squash:
        squashoutput(path, outputname="boutdata.nc", **collect_kwargs)
        collect_kwargs["prefix"] = "boutdata"
        # Delete dump files to be sure we do not read from them
        dump_names = glob(str(path.joinpath("BOUT.dmp.*.nc")))
        for x in dump_names:
            Path(x).unlink()
        # Reset arguments that are taken care of by squashoutput
        for x in ("tind", "xind", "yind", "zind"):
            if x in collect_kwargs:
                collect_kwargs.pop(x)

    for varname in expected:
        actual = collect(varname, path=path, **collect_kwargs)
        npt.assert_array_equal(expected[varname], actual)
        actual_keys = list(actual.attributes.keys())
        if varname in expected_attributes:
            for a in expected_attributes[varname]:
                assert actual.attributes[a] == expected_attributes[varname][a]
                actual_keys.remove(a)

        if "fieldperp" in varname:
            assert actual.attributes["yindex_global"] == fieldperp_global_yind
            actual_keys.remove("yindex_global")

        assert actual_keys == ["bout_type"]

        if "field3d_t" in varname:
            assert actual.attributes["bout_type"] == "Field3D_t"
        elif "field3d" in varname:
            assert actual.attributes["bout_type"] == "Field3D"
        elif "field2d_t" in varname:
            assert actual.attributes["bout_type"] == "Field2D_t"
        elif "field2d" in varname:
            assert actual.attributes["bout_type"] == "Field2D"
        elif "fieldperp_t" in varname:
            assert actual.attributes["bout_type"] == "FieldPerp_t"
        elif "fieldperp" in varname:
            assert actual.attributes["bout_type"] == "FieldPerp"
        elif "_t" in varname or varname == "t_array":
            assert actual.attributes["bout_type"] == "scalar_t"
        else:
            assert actual.attributes["bout_type"] == "scalar"


class TestCollect:
    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_core_min_files(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 1
        grid_info["NYPE"] = 1
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 7
        grid_info["ixseps2"] = 7
        grid_info["jyseps1_1"] = -1
        grid_info["jyseps2_1"] = grid_info["ny"] // 2 - 1
        grid_info["ny_inner"] = grid_info["ny"] // 2
        grid_info["jyseps1_2"] = grid_info["ny"] // 2 - 1
        grid_info["jyseps2_2"] = grid_info["ny"] // 2

        fieldperp_global_yind = 3
        fieldperp_yproc_ind = 0

        rng = np.random.default_rng(100)

        # core
        # core includes "ylower" and "yupper" even though there is no actual y-boundary
        # because collect/squashoutput collect these points
        dump_params = [
            {
                "i": 0,
                "boundaries": ["xinner", "xouter", "ylower", "yupper"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_core(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 3
        grid_info["NYPE"] = 3
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 7
        grid_info["ixseps2"] = 7
        grid_info["jyseps1_1"] = -1
        grid_info["jyseps2_1"] = grid_info["ny"] // 2 - 1
        grid_info["ny_inner"] = grid_info["ny"] // 2
        grid_info["jyseps1_2"] = grid_info["ny"] // 2 - 1
        grid_info["jyseps2_2"] = grid_info["ny"] // 2

        fieldperp_global_yind = 3
        fieldperp_yproc_ind = 0

        rng = np.random.default_rng(100)

        # core
        # core includes "ylower" and "yupper" even though there is no actual y-boundary
        # because collect/squashoutput collect these points
        dump_params = [
            {
                "i": 0,
                "boundaries": ["xinner", "ylower"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 1,
                "boundaries": ["ylower"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 2,
                "boundaries": ["xouter", "ylower"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 3,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 4,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 5,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 6,
                "boundaries": ["xinner", "yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 7,
                "boundaries": ["yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 8,
                "boundaries": ["xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_sol_min_files(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 1
        grid_info["NYPE"] = 1
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 0
        grid_info["ixseps2"] = 0
        grid_info["jyseps1_1"] = -1
        grid_info["jyseps2_1"] = grid_info["ny"] // 2 - 1
        grid_info["ny_inner"] = grid_info["ny"] // 2
        grid_info["jyseps1_2"] = grid_info["ny"] // 2 - 1
        grid_info["jyseps2_2"] = grid_info["ny"] // 2

        fieldperp_global_yind = 3
        fieldperp_yproc_ind = 0

        rng = np.random.default_rng(100)

        # SOL
        dump_params = [
            {
                "i": 0,
                "boundaries": ["xinner", "xouter", "ylower", "yupper"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_sol(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 3
        grid_info["NYPE"] = 3
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 0
        grid_info["ixseps2"] = 0
        grid_info["jyseps1_1"] = -1
        grid_info["jyseps2_1"] = grid_info["ny"] // 2 - 1
        grid_info["ny_inner"] = grid_info["ny"] // 2
        grid_info["jyseps1_2"] = grid_info["ny"] // 2 - 1
        grid_info["jyseps2_2"] = grid_info["ny"] // 2

        fieldperp_global_yind = 3
        fieldperp_yproc_ind = 0

        rng = np.random.default_rng(100)

        # SOL
        dump_params = [
            {
                "i": 0,
                "boundaries": ["xinner", "ylower"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 1,
                "boundaries": ["ylower"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 2,
                "boundaries": ["xouter", "ylower"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 3,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 4,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 5,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 6,
                "boundaries": ["xinner", "yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 7,
                "boundaries": ["yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 8,
                "boundaries": ["xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_singlenull_min_files(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 1
        grid_info["NYPE"] = 3
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 4
        grid_info["ixseps2"] = 7
        grid_info["jyseps1_1"] = grid_info["MYSUB"] - 1
        grid_info["jyseps2_1"] = grid_info["ny"] // 2 - 1
        grid_info["ny_inner"] = grid_info["ny"] // 2
        grid_info["jyseps1_2"] = grid_info["ny"] // 2 - 1
        grid_info["jyseps2_2"] = 2 * grid_info["MYSUB"] - 1

        fieldperp_global_yind = 7
        fieldperp_yproc_ind = 1

        rng = np.random.default_rng(100)

        dump_params = [
            # inner divertor leg
            {
                "i": 0,
                "boundaries": ["xinner", "xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            # core
            {
                "i": 1,
                "boundaries": ["xinner", "xouter"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            # outer divertor leg
            {
                "i": 2,
                "boundaries": ["xinner", "xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_singlenull(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 3
        grid_info["NYPE"] = 9
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 7
        grid_info["ixseps2"] = 13
        grid_info["jyseps1_1"] = grid_info["MYSUB"] - 1
        grid_info["jyseps2_1"] = grid_info["ny"] // 2 - 1
        grid_info["ny_inner"] = grid_info["ny"] // 2
        grid_info["jyseps1_2"] = grid_info["ny"] // 2 - 1
        grid_info["jyseps2_2"] = 6 * grid_info["MYSUB"] - 1

        fieldperp_global_yind = 19
        fieldperp_yproc_ind = 4

        rng = np.random.default_rng(100)

        dump_params = [
            # inner divertor leg
            {
                "i": 0,
                "boundaries": ["xinner", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 1,
                "boundaries": ["ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 2,
                "boundaries": ["xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 3,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 4,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 5,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 6,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 7,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 8,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # core
            {
                "i": 9,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 10,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 11,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 12,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 13,
                "boundaries": [],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 14,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 15,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 16,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 17,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # outer divertor leg
            {
                "i": 18,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 19,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 20,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 21,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 22,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 23,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 24,
                "boundaries": ["xinner", "yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 25,
                "boundaries": ["yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 26,
                "boundaries": ["xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_connected_doublenull_min_files(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 1
        grid_info["NYPE"] = 6
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 4
        grid_info["ixseps2"] = 4
        grid_info["jyseps1_1"] = grid_info["MYSUB"] - 1
        grid_info["jyseps2_1"] = 2 * grid_info["MYSUB"] - 1
        grid_info["ny_inner"] = 3 * grid_info["MYSUB"]
        grid_info["jyseps1_2"] = 4 * grid_info["MYSUB"] - 1
        grid_info["jyseps2_2"] = 5 * grid_info["MYSUB"] - 1

        fieldperp_global_yind = 7
        fieldperp_yproc_ind = 1

        rng = np.random.default_rng(100)

        dump_params = [
            # inner, lower divertor leg
            {
                "i": 0,
                "boundaries": ["xinner", "xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            # inner core
            {
                "i": 1,
                "boundaries": ["xinner", "xouter"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            # inner, upper divertor leg
            {
                "i": 2,
                "boundaries": ["xinner", "xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
            # outer, upper divertor leg
            {
                "i": 3,
                "boundaries": ["xinner", "xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            # outer core
            {
                "i": 4,
                "boundaries": ["xinner", "xouter"],
                "fieldperp_global_yind": -1,
            },
            # outer, lower divertor leg
            {
                "i": 5,
                "boundaries": ["xinner", "xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_connected_doublenull(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 3
        grid_info["NYPE"] = 18
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 7
        grid_info["ixseps2"] = 7
        grid_info["jyseps1_1"] = 3 * grid_info["MYSUB"] - 1
        grid_info["jyseps2_1"] = 6 * grid_info["MYSUB"] - 1
        grid_info["ny_inner"] = 9 * grid_info["MYSUB"]
        grid_info["jyseps1_2"] = 12 * grid_info["MYSUB"] - 1
        grid_info["jyseps2_2"] = 15 * grid_info["MYSUB"] - 1

        fieldperp_global_yind = 19
        fieldperp_yproc_ind = 4

        rng = np.random.default_rng(100)

        dump_params = [
            # inner, lower divertor leg
            {
                "i": 0,
                "boundaries": ["xinner", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 1,
                "boundaries": ["ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 2,
                "boundaries": ["xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 3,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 4,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 5,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 6,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 7,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 8,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # inner core
            {
                "i": 9,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 10,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 11,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 12,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 13,
                "boundaries": [],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 14,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 15,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 16,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 17,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # inner, upper divertor leg
            {
                "i": 18,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 19,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 20,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 21,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 22,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 23,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 24,
                "boundaries": ["xinner", "yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 25,
                "boundaries": ["yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 26,
                "boundaries": ["xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
            # outer, upper divertor leg
            {
                "i": 27,
                "boundaries": ["xinner", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 28,
                "boundaries": ["ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 29,
                "boundaries": ["xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 30,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 31,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 32,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 33,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 34,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 35,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # outer core
            {
                "i": 36,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 37,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 38,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 39,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 40,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 41,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 42,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 43,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 44,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # outer, lower divertor leg
            {
                "i": 45,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 46,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 47,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 48,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 49,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 50,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 51,
                "boundaries": ["xinner", "yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 52,
                "boundaries": ["yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 53,
                "boundaries": ["xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_disconnected_doublenull_min_files(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 1
        grid_info["NYPE"] = 6
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 3
        grid_info["ixseps2"] = 5
        grid_info["jyseps1_1"] = grid_info["MYSUB"] - 1
        grid_info["jyseps2_1"] = 2 * grid_info["MYSUB"] - 1
        grid_info["ny_inner"] = 3 * grid_info["MYSUB"]
        grid_info["jyseps1_2"] = 4 * grid_info["MYSUB"] - 1
        grid_info["jyseps2_2"] = 5 * grid_info["MYSUB"] - 1

        fieldperp_global_yind = 7
        fieldperp_yproc_ind = 1

        rng = np.random.default_rng(100)

        dump_params = [
            # inner, lower divertor leg
            {
                "i": 0,
                "boundaries": ["xinner", "xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            # inner core
            {
                "i": 1,
                "boundaries": ["xinner", "xouter"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            # inner, upper divertor leg
            {
                "i": 2,
                "boundaries": ["xinner", "xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
            # outer, upper divertor leg
            {
                "i": 3,
                "boundaries": ["xinner", "xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            # outer core
            {
                "i": 4,
                "boundaries": ["xinner", "xouter"],
                "fieldperp_global_yind": -1,
            },
            # outer, lower divertor leg
            {
                "i": 5,
                "boundaries": ["xinner", "xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )

    @pytest.mark.parametrize("squash", [False, True])
    @pytest.mark.parametrize("squash_kwargs", squash_kwargs)
    def test_disconnected_doublenull(self, tmp_path, squash, squash_kwargs):
        grid_info = {}
        grid_info["iteration"] = 6
        grid_info["MXSUB"] = 3
        grid_info["MYSUB"] = 4
        grid_info["MZSUB"] = 5
        grid_info["MXG"] = 2
        grid_info["MYG"] = 2
        grid_info["MZG"] = 0
        grid_info["NXPE"] = 3
        grid_info["NYPE"] = 18
        grid_info["NZPE"] = 1
        grid_info["nx"] = grid_info["NXPE"] * grid_info["MXSUB"] + 2 * grid_info["MXG"]
        grid_info["ny"] = grid_info["NYPE"] * grid_info["MYSUB"]
        grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
        grid_info["MZ"] = grid_info["nz"]
        grid_info["ixseps1"] = 6
        grid_info["ixseps2"] = 11
        grid_info["jyseps1_1"] = 3 * grid_info["MYSUB"] - 1
        grid_info["jyseps2_1"] = 6 * grid_info["MYSUB"] - 1
        grid_info["ny_inner"] = 9 * grid_info["MYSUB"]
        grid_info["jyseps1_2"] = 12 * grid_info["MYSUB"] - 1
        grid_info["jyseps2_2"] = 15 * grid_info["MYSUB"] - 1

        fieldperp_global_yind = 19
        fieldperp_yproc_ind = 4

        rng = np.random.default_rng(100)

        dump_params = [
            # inner, lower divertor leg
            {
                "i": 0,
                "boundaries": ["xinner", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 1,
                "boundaries": ["ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 2,
                "boundaries": ["xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 3,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 4,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 5,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 6,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 7,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 8,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # inner core
            {
                "i": 9,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 10,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 11,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 12,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 13,
                "boundaries": [],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 14,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": fieldperp_global_yind,
            },
            {
                "i": 15,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 16,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 17,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # inner, upper divertor leg
            {
                "i": 18,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 19,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 20,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 21,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 22,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 23,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 24,
                "boundaries": ["xinner", "yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 25,
                "boundaries": ["yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 26,
                "boundaries": ["xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
            # outer, upper divertor leg
            {
                "i": 27,
                "boundaries": ["xinner", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 28,
                "boundaries": ["ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 29,
                "boundaries": ["xouter", "ylower"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 30,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 31,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 32,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 33,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 34,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 35,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # outer core
            {
                "i": 36,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 37,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 38,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 39,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 40,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 41,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 42,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 43,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 44,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            # outer, lower divertor leg
            {
                "i": 45,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 46,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 47,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 48,
                "boundaries": ["xinner"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 49,
                "boundaries": [],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 50,
                "boundaries": ["xouter"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 51,
                "boundaries": ["xinner", "yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 52,
                "boundaries": ["yupper"],
                "fieldperp_global_yind": -1,
            },
            {
                "i": 53,
                "boundaries": ["xouter", "yupper"],
                "fieldperp_global_yind": -1,
            },
        ]
        dumps = []
        for p in dump_params:
            dumps.append(
                create_dump_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    **p,
                )
            )

        expected = concatenate_data(
            dumps, nxpe=grid_info["NXPE"], fieldperp_yproc_ind=fieldperp_yproc_ind
        )

        collect_kwargs = {"xguards": True, "yguards": "include_upper"}

        check_collected_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=tmp_path,
            squash=squash,
            collect_kwargs=collect_kwargs,
            squash_kwargs=squash_kwargs,
        )
