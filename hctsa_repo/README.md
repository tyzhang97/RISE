# hctsa dependency for RISE

This directory is the **root directory of the hctsa project used by RISE**.

Place the complete contents of the official [benfulcher/hctsa](https://github.com/benfulcher/hctsa) repository in this directory. In particular, the following hctsa subdirectories must remain available here:

```text
Calculation/
Database/
Operations/
PeripheryFunctions/
TimeSeries/
Toolboxes/
...
```


The RISE hctsa workflow reads the standard operation list from
`Database/INP_mops.txt` and the RISE-provided `Database/Top49_INP_ops.txt`,
prepares MATLAB input files, and runs the MATLAB workers included or adapted in
the RISE package. `Top49_INP_ops.txt` contains the 49 temporal-dynamics
attributes selected for use by the RISE framework; it is the operation list used
to summarize the hctsa outputs for the RISE feature workflow. RISE provides this
49-operation list, while the remaining hctsa source files and standard operation
database come from the upstream hctsa project. A compatible MATLAB release,
MATLAB Engine for Python, hctsa, and the required MATLAB toolboxes must be
installed separately; they are not installed by the Python package or the Conda
environment.

Clone the upstream project directly into this directory, or copy an existing
checkout here:

```bash
git clone https://github.com/benfulcher/hctsa.git hctsa_repo
```

If this directory already exists, clone or copy the hctsa repository contents
inside it rather than creating an additional nested `hctsa_repo/hctsa/` level.

For the complete hctsa installation, setup, workflow, documentation, dependency,
and license information, refer to the official project README:

<https://github.com/benfulcher/hctsa/blob/main/README.md>
