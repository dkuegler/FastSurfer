# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import argparse
from pathlib import Path
from typing import Any, Sequence


def setup_options():
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options : argparse.Namespace
        Namespace object holding options.
    """
    # Validation settings
    parser = argparse.ArgumentParser(description="Wrapper for spherical projection")

    parser.add_argument("--hemi", type=str, help="Hemisphere to analyze.")
    parser.add_argument("--sdir", type=Path, help="Surface directory of subject.")
    parser.add_argument("--subject", type=str, help="Name (ID) of subject.")
    parser.add_argument("--threads", type=int, help="Number of threads to use.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import sys
    opts = setup_options()

    try:
        from spherically_project import spherically_project_surface

        # # make sure the process has a username, so nibabel does not crash in
        # write_geometry
        # from os import environ
        # env = dict(environ)
        # env.setdefault("USERNAME", "UNKNOWN")
        # spherical_wrapper(cmd1, cmd2, env=env)
        spherically_project_surface(
            opts.sdir / f"{opts.hemi}.smoothwm.nofix",
            opts.sdir / f"{opts.hemi}.qsphere.nofix",
            use_cholmod=False,
        )
    except Exception as e:
        from traceback import print_exception
        import shutil

        print_exception(e)
        print("python spherical_project failed.\nRunning FreeSurfer fallback command")

        from FastSurferCNN.utils.run_tools import Popen

        # run the FreeSurfer fallback command
        recon_all = shutil.which("recon-all")
        static_args = ("-qsphere", "-no-isrunning")
        cmd = (recon_all, "-s", opts.subject, " -hemi ", opts.hemi) + static_args
        if opts.threads > 1:
            cmd += ("-threads", str(opts.threads), "-itkthreads", str(opts.threads))
        done = Popen(cmd).forward_output(encoding="utf-8", timeout=None)
        sys.exit(done.retcode)

    # if we get here, we are successful. Return exit code 0.
    sys.exit(0)
