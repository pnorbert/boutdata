# PYTHON_ARGCOMPLETE_OK

"""
Collect all data from BOUT.dmp.* files and create a single output file.

Output file named BOUT.dmp.nc by default

Useful because this discards ghost cell data (that is only useful for debugging)
and because single files are quicker to download.

"""

def squashoutput(datadir=".", outputname="BOUT.dmp.nc", format="NETCDF4", tind=None,
                 xind=None, yind=None, zind=None, xguards=True, yguards="include_upper",
                 singleprecision=False, compress=False, least_significant_digit=None,
                 quiet=False, complevel=None, append=False, delete=False,
                 tind_auto=False):
    """
    Collect all data from BOUT.dmp.* files and create a single output file.

    Parameters
    ----------
    datadir : str
        Directory where dump files are and where output file will be created.
        default "."
    outputname : str
        Name of the output file. File suffix specifies whether to use NetCDF or
        HDF5 (see boututils.datafile.DataFile for suffixes).
        default "BOUT.dmp.nc"
    format : str
        format argument passed to DataFile
        default "NETCDF4"
    tind : slice, int, or [int, int, int]
        tind argument passed to collect
        default None
    xind : slice, int, or [int, int, int]
        xind argument passed to collect
        default None
    yind : slice, int, or [int, int, int]
        yind argument passed to collect
        default None
    zind : slice, int, or [int, int, int]
        zind argument passed to collect
        default None
    xguards : bool
        xguards argument passed to collect
        default True
    yguards : bool or "include_upper"
        yguards argument passed to collect (note different default to collect's)
        default "include_upper"
    singleprecision : bool
        If true convert data to single-precision floats
        default False
    compress : bool
        If true enable compression in the output file
    least_significant_digit : int or None
        How many digits should be retained? Enables lossy
        compression. Default is lossless compression. Needs
        compression to be enabled.
    complevel : int or None
        Compression level, 1 should be fastest, and 9 should yield
        highest compression.
    quiet : bool
        Be less verbose. default False
    append : bool
        Append to existing squashed file
    delete : bool
        Delete the original files after squashing.
    tind_auto : bool, optional
        Read all files, to get the shortest length of time_indices. All data truncated
        to the shortest length.  Useful if writing got interrupted (default: False)
    """
    from boutdata.data import BoutOutputs
    from boututils.datafile import DataFile
    from boututils.boutarray import BoutArray
    import numpy
    import os
    import gc
    import tempfile
    import shutil
    import glob

    try:
        # If we are using the netCDF4 module (the usual case) set caching to zero, since
        # each variable is read and written exactly once so caching does not help, only
        # uses memory - for large data sets, the memory usage may become excessive.
        from netCDF4 import get_chunk_cache, set_chunk_cache
    except ImportError:
        netcdf4_chunk_cache = None
    else:
        netcdf4_chunk_cache = get_chunk_cache()
        set_chunk_cache(0)

    fullpath = os.path.join(datadir, outputname)

    if append:
        datadirnew = tempfile.mkdtemp(dir=datadir)
        for f in glob.glob(datadir + "/BOUT.dmp.*.??"):
            if not quiet:
                print("moving", f, flush=True)
            shutil.move(f, datadirnew)
        oldfile = datadirnew + "/" + outputname
        datadir = datadirnew

    if os.path.isfile(fullpath) and not append:
        raise ValueError(
            fullpath + " already exists. Collect may try to read from this file, which is presumably not desired behaviour.")

    # useful object from BOUT pylib to access output data
    outputs = BoutOutputs(datadir, info=False, xguards=xguards,
                          yguards=yguards, tind=tind, xind=xind, yind=yind, zind=zind,
                          tind_auto=tind_auto)
    outputvars = outputs.keys()
    # Read a value to cache the files
    outputs[outputvars[0]]

    if append:
        # move only after the file list is cached
        shutil.move(fullpath, oldfile)

    t_array_index = outputvars.index("t_array")
    outputvars.append(outputvars.pop(t_array_index))

    kwargs = {}
    if compress:
        kwargs['zlib'] = True
        if least_significant_digit is not None:
            kwargs['least_significant_digit'] = least_significant_digit
        if complevel is not None:
            kwargs['complevel'] = complevel
    if append:
        old = DataFile(oldfile)
        # Check if dump on restart was enabled
        # If so, we want to drop the duplicated entry
        cropnew = 0
        if old['t_array'][-1] == outputs['t_array'][0]:
            cropnew = 1
        # Make sure we don't end up with duplicated data:
        for ot in old['t_array']:
            if ot in outputs['t_array'][cropnew:]:
                raise RuntimeError(
                    "For some reason t_array has some duplicated entries in the new and old file.")
    # Create single file for output and write data
    with DataFile(fullpath, create=True, write=True, format=format, **kwargs) as f:
        for varname in outputvars:
            if not quiet:
                print(varname, flush=True)

            var = outputs[varname]
            if append:
                dims = outputs.dimensions[varname]
                if 't' in dims:
                    var = var[cropnew:, ...]
                    varold = old[varname]
                    var = BoutArray(numpy.append(
                        varold, var, axis=0), var.attributes)

            if singleprecision:
                if not isinstance(var, int):
                    var = BoutArray(numpy.float32(var), var.attributes)

            f.write(varname, var)
            # Write changes, free memory
            f.sync()
            var = None
            gc.collect()

    if delete:
        if append:
            os.remove(oldfile)
        for f in glob.glob(datadir + "/BOUT.dmp.*.??"):
            if not quiet:
                print("Deleting", f, flush=True)
            os.remove(f)
        if append:
            os.rmdir(datadir)

    if netcdf4_chunk_cache is not None:
        # Reset the default chunk_cache size that was changed for squashoutput
        # Note that get_chunk_cache() returns a tuple, so we have to unpack it when
        # passing to set_chunk_cache.
        set_chunk_cache(*netcdf4_chunk_cache)


def main():
    """
    Call the squashoutput function using arguments from command line - used to provide a
    command-line executable using setuptools entry_points in setup.py
    """

    import argparse
    from sys import exit

    try:
        import argcomplete
    except ImportError:
        argcomplete = None

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=(
            __doc__
            + "\n\n"
            + squashoutput.__doc__
            + "\n\nNote: the --tind, --xind, --yind and --zind command line arguments "
            "are converted\ndirectly to Python slice() objects and so use exclusive "
            "'stop' values. They can be\npassed up to 3 values: [stop], [start, stop], "
            "or [start, stop, step]."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    def str_to_bool(string):
        return string.lower() == "true" or string.lower() == "t"

    def int_or_none(string):
        try:
            return int(string)
        except ValueError:
            if string.lower() == "none" or string.lower() == "n":
                return None
            else:
                raise

    parser.add_argument("datadir", nargs="?", default=".")
    parser.add_argument("--outputname", default="BOUT.dmp.nc")
    parser.add_argument("--tind", type=int_or_none, nargs="*", default=[None])
    parser.add_argument("--xind", type=int_or_none, nargs="*", default=[None])
    parser.add_argument("--yind", type=int_or_none, nargs="*", default=[None])
    parser.add_argument("--zind", type=int_or_none, nargs="*", default=[None])
    parser.add_argument("-s", "--singleprecision", action="store_true", default=False)
    parser.add_argument("-c", "--compress", action="store_true", default=False)
    parser.add_argument("-l", "--complevel", type=int_or_none, default=None)
    parser.add_argument(
        "-i", "--least-significant-digit", type=int_or_none, default=None
    )
    parser.add_argument("-q", "--quiet", action="store_true", default=False)
    parser.add_argument("-a", "--append", action="store_true", default=False)
    parser.add_argument("-d", "--delete", action="store_true", default=False)

    if argcomplete:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    for ind in "txyz":
        args.__dict__[ind + "ind"] = slice(*args.__dict__[ind + "ind"])
    # Call the function, using command line arguments
    squashoutput(**args.__dict__)
