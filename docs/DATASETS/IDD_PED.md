# IDD-PeD

Status: Official documentation checked; optional later source; local release unverified

Last updated: 2026-10-01

## Candidate role

Later optional visual/contextual source for difficult unstructured pedestrian interactions. The [project](https://cvit.iiit.ac.in/research/projects/cvit-projects/iddped) describes Indian traffic scenes. Add it only after a named gap and controlled source ablations justify the effort.

## Documented annotation scope

The [official repository](https://github.com/Ruthvik9/IDD-PeD) describes tracked boxes and occlusion, frame-level behaviour attributes, and scene, interaction and location annotations. Behaviour is organized into separate crossing, traffic interaction, activity, attention, social and stationary categories. For example, stationary attributes include standing and sitting, while crossing categories distinguish designated and undesignated crossings.

These are multiple native concepts, not an accepted single LOKI-style state taxonomy. Applicability values such as `N/A` require native interpretation; they must not become generic negative behaviour labels.

## Modalities and selection

The current proposed use is visual behaviour/context supervision. No usable local 3D pedestrian correspondence has been established. Do not infer it from other IDD datasets.

The repository describes pedestrians requiring ego attention; verify the annotation population rather than assuming population coverage matches LOKI. Its value is a possible behavioural/domain stress test, not guaranteed transfer improvement.

## Remaining checks

Acquisition/version, exact attribute IDs/definitions, temporal cadence, missing labels, track continuity, context/pose files, access terms, and feasible sampling remain unverified. Complete the [native ontology audit](../LABEL_ONTOLOGY.md) before loss design. Record whether source additions help or hurt through the [evaluation protocol](../EVALUATION_PLAN.md).

## Pedestrian population and sequence scale (2026-10-01)

The [official release README](https://github.com/Ruthvik9/IDD-PeD) reports **3,284 training pedestrians + 1,632 test pedestrians = 4,916**. These are reported pedestrian split counts, not sliding-window samples; independent cross-clip identity continuity has not been audited.

The [paper, §III.C](https://cvit.iiit.ac.in/images/ConferencePapers/2025/IDD-PeD.pdf) instead describes **over 5,000 annotated pedestrians**, with a median track length of approximately **60 frames**, across videos lasting **45 s to 10 minutes**. It reports 205,145 annotated frames and 494,854 pedestrian boxes. The discrepancy with the release's 4,916 needs a release/filter audit; do not silently round the release count into the paper claim.

Nine `gp_set_*` archives are collection groups, not nine videos. Exact annotated video count remains unknown: browsing exposed per-video XML files, but did not retrieve all group inventories. Count XML video identities and pedestrian tracks from the acquired annotations, retaining behaviour missingness, selection criteria and split provenance.
