# Environment record (as-run, verbatim)

These three files are the **only trustworthy record** of the environment that produced
the released checkpoint and therefore every number in the paper. They were captured
automatically by Weights & Biases at the start of the training run
(`wandb/run-20251227_115819-hhmg10um/files/`) and are copied here **byte-verbatim**.

| file | what it is |
|---|---|
| `requirements.txt` | `pip freeze` of the training environment, 258 packages |
| `conda-environment.yaml` | the conda environment export |
| `wandb-metadata.json` | host, GPU, python version, and the **full training argv** |

Headline facts: python 3.10.16, 4 × NVIDIA H100 NVL, `torch==2.2.0+cu121`,
**stock** `transformers==4.50.3`, `tensorflow==2.18.0`, `robosuite==1.4.1`,
`mujoco==3.3.0`, `mplib==0.2.1`, `flash-attn==2.5.5`.

## Do not trust the old repo's dependency files instead

The research repo's root `pyproject.toml` and `requirements-min.txt` describe an
environment that has **never existed on the machine** (they pin a `transformers` fork,
TF 2.15 and tokenizers 0.19.1). `lilo-vla/pyproject.toml` is derived from
`requirements.txt` in this directory, not from those.

## These files contain absolute machine-local paths — on purpose

`conda-environment.yaml:283` and `wandb-metadata.json:58,65,68` record the original
`/mnt/arc/...` prefix, program path and interpreter. They are **provenance, not code**,
and are left verbatim so the record stays authentic.

A release check of the form

```bash
grep -rn "/mnt/arc" lilo-vla/ --include='*.py' --include='*.sh' --include='*.json' --include='*.yaml'
```

must therefore exclude this directory:

```bash
grep -rn "/mnt/arc" lilo-vla/ --include='*.py' --include='*.sh' --include='*.json' --include='*.yaml' \
     --exclude-dir=repro
```

No code anywhere in the release reads these files. If the absolute paths are considered
too revealing to publish, the four lines can be redacted — but that is a deliberate
edit to a record, and it should be noted in the file when it happens.
