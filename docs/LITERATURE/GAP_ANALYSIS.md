# Gap analysis

Status: Draft; not a novelty claim.  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: evidence-backed comparisons with prior work and the current limits of any novelty claim.
- Links out: source details to paper notes and the working contribution hypothesis to [RESEARCH_DIRECTION.md](../RESEARCH_DIRECTION.md).

| Topic | Current evidence | Project implication | Verification needed |
|---|---|---|---|
| Synth→real binary C/NC self-labeling and pseudo-labeling | [Riaz et al. 2025](https://ddd.uab.cat/record/313066) trains on PedSynth, self-labels real PIE, retains synthetic training data, and uses real validation for stopping. | Established predecessor; synthetic→real labeling alone is not a novelty claim. | Preserve distinction between their binary prediction target and our offline multi-state target. |
| Automatic crossing labels and temporal smoothing | Existing C/NC work includes generated labels and temporal post-processing. | Neither automatic crossing nor smoothing alone is a contribution. | Compare closest geometry and self-labeling protocols in full review. |
| Synthetic rich behavior states | [ARCANE-PedSynth](https://arxiv.org/pdf/2605.24950) reports a 12-state FSM and PedSynth++; paper/code state lists differ per post-checkpoint review. | Candidate source for richer supervision. | Inspect actual released CSV and source version. |
| LOKI behavior labels | [LOKI](https://openaccess.thecvf.com/content/ICCV2021/papers/Girase_LOKI_Long_Term_and_Key_Intentions_for_Trajectory_Prediction_ICCV_2021_paper.pdf) reports frame-wise actions and derives future-action intention for an experiment. | Use original actions for offline target GT. | Inspect release and annotation rules. |
| Synth→real multi-state offline annotation and target-label efficiency | We have not yet identified a direct counterpart in the reviewed core papers. | Candidate gap: compare native rich, binary, and defensible collapsed source supervision at equal real-track budgets. | Complete focused primary-source review before claiming novelty. |
| Transfer of `WAITING_TO_CROSS` and generic `STOPPED` | PedSynth++ is crossing-centered; direct generic stop supervision is unclear. | Specific hypothesis and possible failure mode. | Released class counts, timelines, `R→R` and few-shot results. |
| Human cost and enrichment | ECP2.0 has dense tracks but no comparable behavior ontology reported. | Later test should count adaptation plus independent audit/review and quality versus coverage. | Define validation protocol before deployment. |

The working claim is limited to: **multi-state label efficiency from rich synthetic behavior may be underexplored**. The experiment must show a meaningful reduction in real labeled tracks; a small full-supervision F1 increase would support a narrower conclusion.
