#!/bin/bash
{sbatch_directives}

{env_exports}

# Nothing cluster-specific below this line. The container entry, its binds,
# every environment variable and the srun options are CONFIG -- see
# config/slurm/<cluster>.yaml and config/container/<cluster>_titan.yaml. A quirk
# buried in a template is a quirk nobody can override for one debugging run,
# and it does not appear in the archived config that says what a run was.
module purge

# core_pattern is relative (core.%h.%p) on both clusters, so a crashing rank
# writes a multi-GB dump into the job CWD -- on a quota-limited filesystem.
# Set at the batch shell AND inside the srun shell so every rank inherits it.
ulimit -c 0

echo "Launching {job_name}"
srun {srun_opts}bash -c 'ulimit -c 0; {launcher_cmd} {command}'
