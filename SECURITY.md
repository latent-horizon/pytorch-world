# Security policy

## Supported versions

Security fixes go into the latest release on [PyPI](https://pypi.org/project/torch-world/) and into `main`. Older releases are not patched; upgrade to the latest one.

| version | supported |
|---|---|
| latest release | yes |
| older releases | no |

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub's [private vulnerability reporting](https://github.com/latent-horizon/pytorch-world/security/advisories/new), or by email to kleyton.vsc@gmail.com.

Include what is affected (module, function, version), how to reproduce it, and what an attacker could do with it. You will get an acknowledgement within 3 working days and an assessment within 10. Once a fix is released, the advisory is published with credit to the reporter, unless you ask otherwise.

## Scope

pytorch-world is research code that loads model weights and datasets. Things worth reporting:

- code execution from loading a file the library downloads or reads, for example a checkpoint, dataset or hub asset;
- a release artifact on PyPI that does not match the tagged source;
- a GitHub workflow that exposes secrets or write access to untrusted code.

`torch.load` on a file you do not trust can run arbitrary code, as in PyTorch itself; the library loads its own checkpoints with `weights_only=True` or from safetensors where it can.
