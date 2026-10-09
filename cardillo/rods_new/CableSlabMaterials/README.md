# Material Models

Material models for finite element analysis.

## Available models

- **Elasticity** — 3D isotropic Hooke's law
- **Plasticity** — von Mises plasticity with linear kinematic hardening
- **ViscoElasticity** — generalized Maxwell model
- **ViscoPlasticity** — Perzyna rate-dependent plastic flow

## Material Interface

All material models implement:

'''python
getStress(kinematics, actions, element, intpointID)
'''

The material models require strain information in Voigt notation. Depending on the material model, additional history variables are stored internally.

'getStress()' evaluates the constitutive response at a specific material/integration point. The material point is currently identified using 'element.id' for the element number and 'intpointID' for the integration point number. This can be modified for compatibility with the broader FE framework.

After each converged load step, the 'commit()' method defined in 'BaseMaterial.py' should be called. This promotes the newly calculated history variables to the converged history state for the next load step.

## Kinematics

The current material models use a small-strain formulation. The element formulation is therefore responsible for supplying appropriate local strain measures to the material model. In applications involving large rigid-body rotations, these rotations should not result in artificial material strains.

Finite-strain material models can be added separately if required.
