import contextlib
import csv
import datetime
import glob
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, Iterator, List, Optional, TypedDict

import numpy as np

import background as bg
import backgroundevaluation


@contextlib.contextmanager
def run_in_cwd(path) -> Iterator[str]:
    """
    Use in with-statement to temporarily change current working directory.

    Example::

        with run_in_cwd("/etc"):
            print(os.stat("passwd"))  # do something with /etc/passwd
        # after with-statement, we are back in the old directory
    """
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield cwd
    finally:
        os.chdir(cwd)


class TemporaryBackgroundCwd:
    tmpdir: str
    olddir: str

    def symlink_star(self, starname: str, datafile: str, resultdir: str) -> None:
        self.symlink_star_data(starname, datafile)
        self.symlink_star_results(starname, resultdir)

    def symlink_star_data(self, starname: str, datafile: str) -> None:
        assert os.path.basename(datafile) == f"{starname}.txt", (datafile, starname)
        datafile = os.path.join(self.olddir, datafile)
        assert os.path.exists(datafile), datafile
        os.symlink(datafile, f"data/{starname}.txt_")
        os.rename(f"data/{starname}.txt_", f"data/{starname}.txt")

    def symlink_star_results(self, starname: str, resultdir: str) -> None:
        assert not resultdir.endswith("/"), resultdir
        os.symlink(os.path.join(self.olddir, resultdir), f"results/{starname}_")
        os.rename(f"results/{starname}_", f"results/{starname}")


@contextlib.contextmanager
def run_in_fake_background_environment() -> Iterator[TemporaryBackgroundCwd]:
    """
    Use in with-statement to create a temp dir containing "data", "results"
    and "localPath.txt" and temporarily switch into it.

    Example::

        with run_in_fake_background_environment() as env:
            env.symlink_star(
                "CRT012345", "testfiles/CRT012345.txt", "CRT012345results"
            )

    The "env" object in the example is a TemporaryBackgroundCwd helper object.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with run_in_cwd(tmpdir) as olddir:
            os.mkdir("data")
            os.mkdir("results")
            os.mkdir("build")
            with open("build/localPath.txt", "w") as fp:
                print(tmpdir + "/\n", file=fp)
            os.symlink("build/localPath.txt", "localPath.txt")
            t = TemporaryBackgroundCwd()
            t.tmpdir = tmpdir
            t.olddir = olddir
            yield t


class BackgroundJob(TypedDict):
    "Object representing the inputs and outputs to a single run of ./background"
    starname: str  # e.g. 'CRT0102689620'
    data: str  # e.g. 'data/CRT0102689620.txt'
    results: str  # e.g. 'results/CRT0102689620'
    run: str  # two-digit string, e.g. '01'
    model: str  # model name, e.g. 'OneHarvey'


def read_job_list(filename: str) -> List[BackgroundJob]:
    """
    Read a newline-delimited JSON file containing BackgroundJob objects.
    """
    with open(filename) as fp:
        job_list: List[BackgroundJob] = [
            json.loads(line) for line in fp if line.startswith("{")
        ]
    return job_list


def write_job_list(filename: str, job_list: List[BackgroundJob]) -> None:
    """
    Write a newline-delimited JSON file of BackgroundJob objects.
    """
    with open(filename, "w") as fp:
        for job in job_list:
            fp.write(json.dumps(job) + "\n")


def make_background_job_list(
    datadir: str, datafiles: List[str], resultdir: str, run: int, model_name: str
) -> List[BackgroundJob]:
    """
    Create BackgroundJob objects from a list of paths to data files.

    Example::

        job_list = make_background_job_list(
            "data", glob.glob("data/CRT*.txt"), "results", 30, "OneHarvey"
        )
    """
    assert 0 <= run < 100
    run_str = "%02d" % run
    assert not datadir.endswith("/"), datadir
    assert not resultdir.endswith("/"), resultdir
    jobs: List[BackgroundJob] = []
    for datafile in datafiles:
        assert datafile.startswith(datadir + "/"), (datafile, datadir)
        assert datafile.endswith(".txt")
        datafilebase = datafile.rpartition(".txt")[0]
        starname = os.path.basename(datafilebase)
        # Replace datadir with resultdir to obtain resultfile
        resultfile = resultdir + datafilebase.partition(datadir)[2]
        jobs.append(
            {
                "starname": starname,
                "data": datafile,
                "results": resultfile,
                "run": run_str,
                "model": model_name,
            }
        )
    return jobs


def make_corot_job_list(run: int) -> List[BackgroundJob]:
    return make_background_job_list(
        datadir="data",
        datafiles=glob.glob("data/*/CRT*.txt"),
        resultdir="results",
        run=run,
        model_name="OneHarvey",
    )


def main_make_corot_job_list(run: int) -> None:
    job_list = make_corot_job_list(run=run)
    write_job_list("corotstars%02d.jsonl" % run, job_list)


@contextlib.contextmanager
def suppress_stdout():
    """
    Use in with-statement to suppress prints to stdout inside the block.

    Example::

        print("Something is printed!")
        with suppress_stdout():
            print("This line is not printed")
        print("Now this is printed again")
    """
    old_stdout = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            yield old_stdout
        finally:
            sys.stdout = old_stdout


class NothingToDo(Exception):
    pass


class WriteHyperparametersResult(TypedDict):
    skipped: List[BackgroundJob]
    written: List[BackgroundJob]
    badnumax: List[BackgroundJob]


def write_hyperparameters_from_numax_and_data(
    numax: Dict[str, float], jobs: List[BackgroundJob]
) -> WriteHyperparametersResult:
    """
    Invoke bg.set_background_priors() for each BackgroundJob in "jobs"
    using the numax guesses in the dictionary "numax".
    """
    if not jobs:
        raise NothingToDo("write_hyperparameters_from_numax_and_data: jobs is empty")
    skipped: List[BackgroundJob] = []
    written: List[BackgroundJob] = []
    badnumax: List[BackgroundJob] = []

    jobs2: List[BackgroundJob] = []
    for job in jobs:
        paramfile = os.path.join(
            job["results"], "background_hyperParameters_%s.txt" % job["run"]
        )
        if os.path.exists(paramfile):
            skipped.append(job)
        else:
            jobs2.append(job)
            os.makedirs(os.path.join(job["results"], job["run"]), exist_ok=True)
    if not jobs2:
        print(
            "write_hyperparameters_from_numax_and_data: All hyperparameters have already been written"
        )
        return {
            "skipped": skipped,
            "written": written,
            "badnumax": badnumax,
        }
    print("Run set_background_priors() on %s stars" % len(jobs2))
    # set_background_priors() is quite noisy, so we suppress all prints...
    with run_in_fake_background_environment() as env, suppress_stdout():
        for job in jobs2:
            env.symlink_star(job["starname"], job["data"], job["results"])
            star_numax = numax[job["starname"]]
            try:
                bg.set_background_priors(
                    catalog_id="",
                    star_id=job["starname"],
                    numax=star_numax,
                    model_name=job["model"],
                    dir_flag=int(job["run"]),
                )
            except ValueError:
                # Numerical issues - try increasing numax to 2
                badnumax.append(job)
                if star_numax >= 2:
                    # We should maybe 'raise' instead of 'continue' here
                    continue
                star_numax = 2
                try:
                    bg.set_background_priors(
                        catalog_id="",
                        star_id=job["starname"],
                        numax=star_numax,
                        model_name=job["model"],
                        dir_flag=int(job["run"]),
                    )
                except ValueError:
                    # We should maybe 'raise' instead of 'continue' here
                    continue
            written.append(job)
    return {
        "skipped": skipped,
        "written": written,
        "badnumax": badnumax,
    }


def read_corot_reftable(tablepath: str) -> Dict[str, float]:
    """
    Read tab-separated CSV file containing corot_id and numax values.
    """
    numax: Dict[str, float] = {}
    with open(tablepath) as fp:
        for row in csv.DictReader(fp, dialect="excel-tab"):
            corot_id = int(row["corot_id"])
            starname = "CRT0" + str(corot_id)
            star_numax = float(row["numax"])
            numax[starname] = star_numax
    return numax


def main_make_corot_hyperparameters(run: int) -> None:
    job_list = read_job_list("corotstars%02d.jsonl" % run)
    assert job_list
    numax = read_corot_reftable("projects/corot/estimatenumax/reftable30.txt")
    job_list = [j for j in job_list if j["starname"] in numax]
    assert job_list
    write_hyperparameters_from_numax_and_data(numax, job_list)
    write_job_list("corotstars%02d.jsonl" % run, job_list)


def run_background(
    job_list: List[BackgroundJob],
    background_executable: str,
    parallelism: Optional[int] = None,
) -> None:
    """
    Run ./background in parallel on each job in the job_list.
    """
    if parallelism is None:
        parallelism = len(os.sched_getaffinity(0)) or 1
    hostname = socket.gethostname()
    already_run = 0
    ok = 0
    errors = 0
    with run_in_fake_background_environment() as env:
        next_job = 0

        def runner() -> None:
            nonlocal already_run, ok, errors, next_job

            while next_job < len(job_list):
                job = job_list[next_job]
                next_job += 1
                print(
                    "[%s/%s] %s errors, %s ok - now running %s"
                    % (
                        next_job - already_run,
                        len(job_list) - already_run,
                        errors,
                        ok,
                        job["data"],
                    ),
                    flush=True,
                )
                env.symlink_star(job["starname"], job["data"], job["results"])
                runresults = os.path.join("results", job["starname"], job["run"])
                runningpath = os.path.join(runresults, "output_running.txt")
                errorpath = os.path.join(runresults, "output_error.txt")
                successpath = os.path.join(runresults, "output_success.txt")
                if os.path.exists(errorpath) or os.path.exists(successpath):
                    already_run += 1
                    continue
                prefix = ""
                cmdline = [
                    os.path.join(env.olddir, background_executable),
                    prefix,
                    job["starname"],
                    job["run"],
                    job["model"],
                    "background_hyperParameters",
                    "0.0",
                    "0",
                ]
                # Run ./background and redirect stdout+stderr to files
                with open(runningpath, "w") as fp:
                    fp.write(
                        "[%s] Running on %s: %s\n"
                        % (
                            datetime.datetime.now(),
                            hostname,
                            " ".join(map(shlex.quote, cmdline)),
                        )
                    )
                    fp.flush()
                    t_start = time.time()
                    exitcode = subprocess.call(
                        cmdline, stdin=subprocess.DEVNULL, stdout=fp, stderr=fp
                    )
                    fp.write(
                        "[%s] Elapsed time: %s - Exit code: %s\n"
                        % (datetime.datetime.now(), time.time() - t_start, exitcode)
                    )
                if exitcode == 0:
                    os.rename(runningpath, successpath)
                    ok += 1
                else:
                    os.rename(runningpath, errorpath)
                    errors += 1

        if parallelism <= 1:
            runner()
        else:
            threads = [threading.Thread(target=runner) for _ in range(parallelism)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        print(
            "All done! %s had already run, %s errors, %s ok"
            % (already_run, errors, ok),
            flush=True,
        )


def main_run_background(
    run: int, background_executable: str, parallelism: Optional[int] = None
) -> None:
    with open("corotstars%02d.jsonl" % run) as fp:
        job_list: List[BackgroundJob] = [
            json.loads(line) for line in fp if line.startswith("{")
        ]
    run_background(job_list, background_executable, parallelism)


def main_auto_adjust_hyperparameters(run1: int, run2: int) -> None:
    with open("corotstars%02d.jsonl" % run1) as fp:
        job_list: List[BackgroundJob] = [
            json.loads(line) for line in fp if line.startswith("{")
        ]

    jobs2: List[BackgroundJob] = []
    run1str = "%02d" % run1
    run2str = "%02d" % run2
    new_already_exists = 0
    success_count = 0
    running_count = 0
    for job in job_list:
        runresults1 = os.path.join(job["results"], job["run"])
        runresults2 = os.path.join(job["results"], run2str)
        if os.path.exists(runresults2):
            new_already_exists += 1
            jobs2.append(
                {
                    "starname": job["starname"],
                    "data": job["data"],
                    "results": job["results"],
                    "run": run2str,
                    "model": job["model"],
                }
            )
            continue
        runningpath = os.path.join(runresults1, "output_running.txt")
        errorpath = os.path.join(runresults1, "output_error.txt")
        successpath = os.path.join(runresults1, "output_success.txt")
        if os.path.exists(successpath):
            success_count += 1
            continue
        if os.path.exists(runningpath):
            running_count += 1
            continue
        if not os.path.exists(errorpath):
            continue
        new_params = np.asarray(
            backgroundevaluation.auto_adjust_run_hyperparameters(
                run1str, job["results"]
            )
        )

        header = """
        Hyper parameters used for setting up uniform priors.
        Each line corresponds to a different free parameter (coordinate).
        Column #1: Minima (lower boundaries)
        Column #2: Maxima (upper boundaries)
        """
        newrundir = os.path.join(job["results"], run2str)
        os.makedirs(newrundir, exist_ok=True)
        newhyperparamsfile = os.path.join(
            job["results"], "background_hyperParameters_" + run2str + ".txt"
        )
        np.savetxt(newhyperparamsfile, new_params, fmt="%.3f", header=header)
        jobs2.append(
            {
                "starname": job["starname"],
                "data": job["data"],
                "results": job["results"],
                "run": run2str,
                "model": job["model"],
            }
        )
    print(
        "Rerun %s stars of which %s already had new parameters, skip %s that are running, skip %s that were successful"
        % (len(jobs2), new_already_exists, running_count, success_count)
    )
    write_job_list("corotstars%02d.jsonl" % run2, jobs2)


def main() -> None:
    main_make_corot_job_list(run=42)
    main_make_corot_hyperparameters(run=42)
    background_executable = "/home/mathias/work/DiamondsBackground/build/background"
    main_run_background(42, background_executable)
    main_auto_adjust_hyperparameters(42, 43)
    main_run_background(43, background_executable)
    main_auto_adjust_hyperparameters(43, 44)
    main_run_background(44, background_executable)


if __name__ == "__main__":
    main()
