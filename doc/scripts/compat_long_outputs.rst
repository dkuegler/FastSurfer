recon_surf: compat_long_outputs.py
==================================

`compat_long_outputs.py` is a compatibility script. It creates files that are not created by the FastSurfer longitudinal pipeline, but needed by other longitudinal tools such as FreeSurfer's longitudinal Hippocampus and Amygdala processing.


Full commandline interface of recon_surf/compat_long_outputs.py
---------------------------------------------------------------
.. argparse::
    :module: recon_surf.compat_long_outputs
    :func: make_parser
    :prog: recon_surf/compat_long_outputs.py
