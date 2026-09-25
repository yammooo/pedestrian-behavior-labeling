# Ideas backlog

Status: Uncommitted  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: optional **methods** to try if a question or observed failure calls for them.
- Links out: unresolved decisions to [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) and current design choices to [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md).

These are not architecture commitments. Promote an idea only after naming the result or failure that motivates it.

## Temporal modeling

- Several encoder branches, transformed into embedding, fed into ASFormer (or some similar architecture).
- Fixed chunks versus whole tracks if release lengths or memory demand it.

## Representations

- Consistent 2D pose extraction and a pose branch if motion leaves posture-related errors.
- Frozen pretrained visual embeddings; consider separate crop and scene context only if their contributions differ.
- Road-relative features, semantic segmentation, or crosswalk recognition if crossing/waiting errors need road context.

## Transfer and label quality

- Domain adaptation or modality dropout if `S→S` and `R→R` work but `S→R` fails for a diagnosed representation shift.
- Track-quality-aware labeling, confidence calibration, abstention, or active review if generated labels must meet a measured quality/coverage target.
