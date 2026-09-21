"""The ladder's Megatron-LM side: submission, harvesting, and the checks.

NO MODEL CODE LIVES HERE ANY MORE. Complex KDA is a first-class attention
variant in the fork -- `megatron/core/ssm/complex_kda.py`, selected with
`experimental_attention_variant=complex_kda` and placed with
`linear_attention_freq`, exactly as GatedDeltaNet and mLSTM are. An arm is a
combination of those settings and `submit_ladder.arm_overrides` is the mapping.

It used to be a `--spec` module here, pointed at from the command line and
handed the arm through an argument of its own. That worked, and it made the
model something Megatron could not describe: `--spec` takes a module path and a
name, resolves it with `vars(module)[name]`, and assigns whatever it finds
directly as the layer spec. Nothing is validated, nothing reaches the
checkpoint's argument record, and the spec has to be built at import time
because a module-level `__getattr__` is never consulted. As a variant instead,
the model is described by typed config fields that Megatron checks, saves and
prints.

WHY MEGATRON AT ALL. Our torchtitan ladder reproduces the reference's
configuration but not its loss: at 47M/6BT the best matched torchtitan run is
2.9714 where Megatron on the same machine, same corpus and same
hyperparameters gives 2.9224 against their published 2.9272. Every
configuration difference we could find has been measured and set -- data
shuffling, qk_norm, beta2, projection biases, norm epsilon, initialization --
and 0.049 of framework remains. Running the ladder here removes it.
"""
