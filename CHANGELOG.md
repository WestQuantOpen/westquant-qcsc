# Changelog

## 0.2.0

- **Breaking:** the console command is renamed from `westquant` to
  `westquant-qcsc`. Both this package and the `westquant` umbrella package
  installed a `westquant` script, so installing both gave non-deterministic
  command ownership. Subcommands and options are unchanged:
  `westquant audit .` becomes `westquant-qcsc audit .`.
- The Python API (`import qcsc`) is unchanged.

## 0.1.1

- Advisory capability records for CUDA-Q CPU/GPU targets.
