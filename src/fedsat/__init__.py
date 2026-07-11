"""fedsat — Cross-region federated learning for satellite image classification.

A small, tested package backing the rebuild described in PLAN.md. Notebooks in
``notebooks/`` are thin drivers that import from here so that logic is versioned,
reusable, and testable (unlike the old monolithic notebooks).

Modules
-------
config  : typed ExperimentConfig with YAML round-trip
utils   : seeding, hashing, device, small IO helpers
data    : EuroSAT loader + integrity gate + Dirichlet partitioning + splits + transforms
models  : backbone builder (ResNet-18/50, norm policy, multispectral stem)
engine  : centralized/local train + evaluation (all metrics from one model, on TEST)
"""

__version__ = "0.1.0"
