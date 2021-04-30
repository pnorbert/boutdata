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
    {"compress": True},
    {"compress": True, "complevel": 0},
    {"compress": True, "complevel": 1},
    {"compress": True, "complevel": 2},
    {"compress": True, "complevel": 3},
    {"compress": True, "complevel": 4},
    {"compress": True, "complevel": 5},
    {"compress": True, "complevel": 6},
    {"compress": True, "complevel": 7},
    {"compress": True, "complevel": 8},
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

        dumps = []

        # core
        # core includes "ylower" and "yupper" even though there is no actual y-boundary
        # because collect/squashoutput collect these points
        dumps.append(
            create_dump_file(
                i=0,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "xouter", "ylower", "yupper"],
                fieldperp_global_yind=fieldperp_global_yind,
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

        dumps = []

        # core
        # core includes "ylower" and "yupper" even though there is no actual y-boundary
        # because collect/squashoutput collect these points
        dumps.append(
            create_dump_file(
                i=0,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "ylower"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=1,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["ylower"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=2,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter", "ylower"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=3,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=4,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=5,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=6,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "yupper"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=7,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["yupper"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=8,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter", "yupper"],
                fieldperp_global_yind=-1,
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

        dumps = []

        # SOL
        dumps.append(
            create_dump_file(
                i=0,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "xouter", "ylower", "yupper"],
                fieldperp_global_yind=fieldperp_global_yind,
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

        dumps = []

        # SOL
        dumps.append(
            create_dump_file(
                i=0,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "ylower"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=1,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["ylower"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=2,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter", "ylower"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=3,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=4,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=5,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=6,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "yupper"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=7,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["yupper"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=8,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter", "yupper"],
                fieldperp_global_yind=-1,
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

        dumps = []

        # inner divertor leg
        dumps.append(
            create_dump_file(
                i=0,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "xouter", "ylower"],
                fieldperp_global_yind=-1,
            )
        )

        # core
        dumps.append(
            create_dump_file(
                i=1,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "xouter"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )

        # outer divertor leg
        dumps.append(
            create_dump_file(
                i=2,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "xouter", "yupper"],
                fieldperp_global_yind=-1,
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

        dumps = []

        # inner divertor leg
        dumps.append(
            create_dump_file(
                i=0,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "ylower"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=1,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["ylower"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=2,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter", "ylower"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=3,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=4,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=5,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=6,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=7,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=8,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=-1,
            )
        )

        # core
        dumps.append(
            create_dump_file(
                i=9,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=10,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=11,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=12,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=13,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=14,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=fieldperp_global_yind,
            )
        )
        dumps.append(
            create_dump_file(
                i=15,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=16,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=17,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=-1,
            )
        )

        # outer divertor leg
        dumps.append(
            create_dump_file(
                i=18,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=19,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=20,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=21,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=22,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=[],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=23,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=24,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xinner", "yupper"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=25,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["yupper"],
                fieldperp_global_yind=-1,
            )
        )
        dumps.append(
            create_dump_file(
                i=26,
                tmpdir=tmp_path,
                rng=rng,
                grid_info=grid_info,
                boundaries=["xouter", "yupper"],
                fieldperp_global_yind=-1,
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
