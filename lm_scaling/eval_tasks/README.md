# Task overrides for the downstream evaluation

lm-eval-harness ships task configs that name the datasets' ORIGINAL locations,
and three of the nine tasks this study reports name repositories that
`datasets` 4.4.2 will not load: they are script-based (`social_i_qa`,
`super_glue`, bare `winogrande`), and loading-script support was removed. The
error is a `ConnectionError` about reaching the Hub, which on a compute node
with no route out looks like a network problem rather than a refusal.

Each of those files is the stock config with the DATASET ROUTE changed and
nothing else -- same prompt, same choices, same target, same metric -- so a row
stays comparable to a published one. `eval_downstream.py` passes this directory
to `TaskManager(include_path=...)`, where a config with the same `task:` name
replaces the stock one.

Keep them here rather than beside the cluster's cache: these decide what the
numbers in the table mean, and a file that lives only on a scratch filesystem
is not part of the experiment's record.
