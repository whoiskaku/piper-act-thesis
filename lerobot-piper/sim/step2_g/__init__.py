"""Step 2 G0 scaffolding.

The current implementation is an M0 *synthetic contract test*.  It is not the
real-robot dynamics model G and does not constitute a model result.
"""

from .m0_synthetic import (
    CalibrationArtifact,
    CalibrationSchema,
    M0Config,
    ParameterBounds,
    PlantParameters,
    build_synthetic_study,
    calibrate,
    evaluate_role_separated_holdout,
    run_m0,
)

__all__ = [
    "CalibrationArtifact",
    "CalibrationSchema",
    "M0Config",
    "ParameterBounds",
    "PlantParameters",
    "build_synthetic_study",
    "calibrate",
    "evaluate_role_separated_holdout",
    "run_m0",
]
