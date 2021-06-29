# PYTHON_ARGCOMPLETE_OK

"""
Collect all data from BOUT.dmp.* files and create a single output file.

Output file named BOUT.dmp.nc by default

Useful because this discards ghost cell data (that is only useful for debugging)
and because single files are quicker to download.

"""


def squashoutput(
    datadir=".",
    outputname="BOUT.dmp.nc",
    format="NETCDF4",
    tind=None,
    xind=None,
    yind=None,
    zind=None,
    xguards=True,
    yguards="include_upper",
    singleprecision=False,
    compress=False,
    least_significant_digit=None,
    quiet=False,
    complevel=None,
    append=False,
    delete=False,
    tind_auto=False,
    parallel=False,
    time_split_size=None,
    time_split_first_label=0,
    disable_parallel_write=False,
):
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
    parallel : bool or int, default False
        If set to True or 0, use the multiprocessing library to read data in parallel
        with the maximum number of available processors. If set to an int, use that many
        processes.
    time_split_size : int, optional
        By default no splitting is done. If an integer value is passed, the output is
        split into files with length in the t-dimension equal to that value. The outputs
        are labelled by prefacing a counter (starting by default at 0, but see
        time_split_first_label) to the file name before the .nc suffix.
    time_split_first_label : int, default 0
        Value at which to start the counter labelling output files when time_split_size
        is used.
    disable_parallel_write : bool, default False
        Parallel writing may increase memory usage, so it can be disabled even when
        reading in parallel by setting this argument to True.
    """
    # use local imports to allow fast import for tab-completion
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
            "{} already exists. Collect may try to read from this file, which is "
            "presumably not desired behaviour.".format(fullpath)
        )

    # useful object from BOUT pylib to access output data
    outputs = BoutOutputs(
        datadir,
        info=False,
        xguards=xguards,
        yguards=yguards,
        tind=tind,
        xind=xind,
        yind=yind,
        zind=zind,
        tind_auto=tind_auto,
        parallel=parallel,
    )
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
        kwargs["zlib"] = True
        if least_significant_digit is not None:
            kwargs["least_significant_digit"] = least_significant_digit
        if complevel is not None:
            kwargs["complevel"] = complevel
    if append:
        if time_split_size is not None:
            raise ValueError("'time_split_size' is not compatible with append=True")
        old = DataFile(oldfile)
        # Check if dump on restart was enabled
        # If so, we want to drop the duplicated entry
        cropnew = 0
        if old["t_array"][-1] == outputs["t_array"][0]:
            cropnew = 1
        # Make sure we don't end up with duplicated data:
        for ot in old["t_array"]:
            if ot in outputs["t_array"][cropnew:]:
                raise RuntimeError(
                    "For some reason t_array has some duplicated entries in the new "
                    "and old file."
                )
    kwargs["format"] = format

    # Create file(s) for output and write data
    if time_split_size is None:
        filenames = [fullpath]
        t_slices = [slice(None)]
    else:
        tind = outputs.tind
        # tind.stop + 1 - tind.start is the total number of t-indices ignoring the step.
        # Adding tind.step - 1 and integer-dividing by tind.step converts to the total
        # number accounting for the step.
        nt = (tind.stop + 1 - tind.start + tind.step - 1) // tind.step
        n_outputs = (nt + time_split_size - 1) // time_split_size
        filenames = []
        t_slices = []
        for i in range(n_outputs):
            parts = fullpath.split(".")
            parts[-2] += str(time_split_first_label + i)
            filename = ".".join(parts)
            filenames.append(filename)
            t_slices.append(slice(i * time_split_size, (i + 1) * time_split_size))

    workers = SquashWorkers(
        False if disable_parallel_write else parallel, filenames, kwargs
    )

    for varname in outputvars:
        if not quiet:
            print(varname, flush=True)

        var = outputs[varname]
        dims = outputs.dimensions[varname]
        if append:
            if "t" in dims:
                var = var[cropnew:, ...]
                varold = old[varname]
                var = BoutArray(numpy.append(varold, var, axis=0), var.attributes)

        if singleprecision:
            if not isinstance(var, int):
                var = BoutArray(numpy.float32(var), var.attributes)

        if "t" in dims:
            workers.write_data(varname, [var[t_slice] for t_slice in t_slices])
        else:
            workers.write_data(varname, [var])

        var = None
        gc.collect()

    del workers
    del outputs
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


class SquashWorkers:
    """
    Class for packaging up worker processes for parallel writes, or passing the write
    through in serial if parallel functionality not requested

    Parameters
    ----------
    parallel : bool or int, default False
        If False, write in serial. If True or 0 use all available processes. If positive
        integer, use that many processes.
    filenames : list of str
        Names of the files to write to.
    kwargs : dict
        Keyword arguments to pass to DataFile constructors
    """

    def __init__(self, parallel, filenames, kwargs):
        self.parallel = parallel
        self.kwargs = kwargs
        if self.parallel is False or len(filenames) == 1:
            # No point doing parallel writing if there is only one output file
            self.parallel = False
            self.open_files(filenames)
            return

        if self.parallel is True or self.parallel == 0:
            from boututils.run_wrapper import determineNumberOfCPUs

            self.parallel = determineNumberOfCPUs()

        self.n_outputs = len(filenames)

        self.create_workers(filenames)

    def __del__(self):
        if self.parallel is False:
            for f in self.files:
                f.close()
                del f
        else:
            for worker, connection, _ in self.workers:
                # Send None to terminate worker process cleanly
                connection.send(None)
                worker.join()
                connection.close()

    def open_files(self, filenames):
        """
        Open files for serial writing

        Parameters
        ----------
        filenames : list of str
            Names of the files to write to.
        """
        # use local imports to allow fast import for tab-completion
        from boututils.datafile import DataFile

        self.files = [
            DataFile(name, create=True, write=True, **self.kwargs) for name in filenames
        ]

    def create_workers(self, filenames):
        """
        Create workers for parallel writing

        Parameters
        ----------
        filenames : list of str
            Names of the files to write to.
        """
        from multiprocessing import Process, Pipe

        n = min(self.parallel, self.n_outputs)
        files_per_proc = [self.n_outputs // n for _ in range(n)]
        for i in range(self.n_outputs % n):
            files_per_proc[i] += 1
        assert sum(files_per_proc) == self.n_outputs
        counter = 0
        self.workers = []
        for i in range(n):
            files_dict = {}
            worker_indexes = range(counter, counter + files_per_proc[i])
            for index in worker_indexes:
                files_dict[index] = filenames[counter]
                counter = counter + 1
            parent_connection, child_connection = Pipe()
            worker = Process(
                target=self.worker_function,
                args=(child_connection, files_dict),
            )
            worker.start()
            self.workers.append((worker, parent_connection, worker_indexes))

    def worker_function(self, connection, files_dict):
        """
        Function controlling execution on worker processes
        """
        from boututils.datafile import DataFile
        from boututils.boutarray import BoutArray

        output_files = {}
        for i, name in files_dict.items():
            output_files[i] = DataFile(name, create=True, write=True, **self.kwargs)

        while True:
            args = connection.recv()
            if args is None:
                # Terminate process cleanly
                for f in files_dict.values():
                    f.close()
                connection.close()
                return

            varname, data, attributes = args

            if not isinstance(data, dict):
                # Time-independent variable or only one output file - write to all files
                data = BoutArray(data, attributes=attributes)
                for f in output_files.values():
                    f.write(varname, data)
                    # Write changes, free memory
                    f.sync()
            else:
                # Time-dependent variable, write each slice to a separate file
                for i, array in data.items():
                    array = BoutArray(array, attributes=attributes)
                    output_files[i].write(varname, array)
                    # Write changes, free memory
                    output_files[i].sync()

    def write_data(self, varname, data_list):
        """
        Write data to the output files

        Parameters
        ----------
        varname : str
            Name of the variable being written.
        data_list : list of BoutArray
            Data to be written, either a single entry for a time-independent variable,
            or self.n_outputs arrays giving all the time slices of a time-dependent
            variable.
        """
        if self.parallel is False:
            if len(data_list) == 1:
                # Time-independent variable or only one output file - write to all files
                for f in self.files:
                    f.write(varname, data_list[0])
                    # Write changes, free memory
                    f.sync()
            else:
                # Time-dependent variable, write each slice to a separate file
                for i, f in enumerate(self.files):
                    f.write(varname, data_list[i])
                    # Write changes, free memory
                    f.sync()
            return

        # Note, need to pass attributes separately because apparently pickling (which
        # happens to the data arrays when they are sent) converts BoutArrays to numpy
        # arrays, losing the attributes.
        if len(data_list) == 1:
            # Time-independent variable or only one output file - write to all files
            for _, connection, _ in self.workers:
                connection.send((varname, data_list[0], data_list[0].attributes))
        else:
            # Time-dependent variable, write each slice to a separate file
            for _, connection, worker_indexes in self.workers:
                connection.send(
                    (
                        varname,
                        {i: data_list[i] for i in worker_indexes},
                        data_list[0].attributes,
                    )
                )
