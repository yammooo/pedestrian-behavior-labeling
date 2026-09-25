# Evaluation plan

Status: Draft  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: split and sampling rules, real-label budgets, baselines, metrics, diagnostics, and audit protocol.
- Links out: class meanings to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md); observed results will belong in future experiment records.

The primary evaluation unit is per-frame canonical behavior state. Macro-F1, per-class precision/recall/F1, and confusion matrices are the likely primary metrics because class imbalance is expected. Final metric choices still depend on the verified ontology.

## Diagnostic regimes

Let `S` be PedSynth++ and `R` be LOKI.

| Regime | Meaning | Main question |
|---|---|---|
| `S→S` | Train/evaluate on disjoint PedSynth++ data | **Learnability:** can the data, preprocessing, representation, and model solve the source task? |
| `R→R` | Same representation, trained/evaluated with proper LOKI splits | **Observability:** does this representation contain enough target-domain information? This is a diagnostic oracle, not the desired labeler. |
| `S→R` | Train on PedSynth++; zero-shot evaluate on LOKI | **Transferability:** does synthetic supervision generalize to real data? |
| `S→R_small` | PedSynth++ pretraining followed by a small labeled LOKI adaptation set | How much real annotation closes the transfer gap relative to training from scratch? |

Interpret failures in this order: `S→S` failure cannot be blamed on domain shift; good `S→S` plus poor `R→R` suggests insufficient target information or semantic ambiguity; good `S→S` and `R→R` with poor `S→R` points to domain or label mismatch. Very small real adaptation gains suggest calibration/decision-boundary mismatch; little improvement suggests a deeper representation or ontology problem. These are diagnostic interpretations, not proofs.

## Primary label-efficiency comparison

At each positive fixed LOKI adaptation budget, compare the **same input representation and architecture** initialized from scratch versus initialized with PedSynth++ supervision. Candidate budgets are `25`, `50`, and `100` tracks per class, plus a fully supervised reference, subject to actual class/track counts. The `0`-track case is a synthetic-only zero-shot diagnostic with no meaningful scratch counterpart. A track may contain multiple states; define selection and count unique tracks so budgets do not double-count episodes. These values are planning probes, not sample-complexity guarantees.

The main outcome is held-out real macro-F1 and per-class results as a function of **unique labeled tracks**, together with the target-label budget needed to reach a predeclared performance level. Candidate source objectives are binary crossing, a defensible collapsed four-state mapping, and native rich-state pretraining. Test only objectives supported by inspected labels; do not interpret a small full-supervision F1 gain alone as evidence of meaningful label efficiency.

The practical human cost is `B_human = B_adaptation + B_audit/review`. If labels are later exported to an unlabeled dataset, report accepted-label quality against automatic coverage and include the independent audit budget. Proposed 300–500 total reviewed/labeled tracks are only a planning heuristic, not a threshold or decision.

| Evaluation level | Candidate measures | Open condition |
|---|---|---|
| Frame/state labels | Per-class precision, recall, F1; macro-F1; balanced accuracy; confusion matrix | Depends on ontology and class balance. |
| Segments/transitions | Segment IoU; onset/offset error; temporal stability | Requires precise segment definitions. |
| Crossing events | Crossing F1; absolute onset error | Requires compatible event semantics. |
| Synthetic-to-real transfer | `S→R` zero-shot and `S→R_small` few-shot curves | Requires accepted PedSynth++→LOKI mapping and leakage-safe shared inputs. |
| Real cross-dataset transfer | LOKI↔EMT only if later justified | Requires ontology mapping and access. |
| Confidence/abstention | Coverage at target reliability; error-vs-confidence calibration | Requires a confidence definition. |
| Modality value | Ablations from kinematics through added evidence | Must use consistent splits and protocol. |

## Split and analysis safeguards

- Never split adjacent frames of the same track across train/test. Use an appropriate track, scene, sequence, or location-level split after release inspection.
- Report source and target results by class, especially `WAITING_TO_CROSS` versus `STOPPED`.
- Treat modality ablations as transfer diagnostics: motion; motion+pose; motion+pose+visual; and only additional combinations that answer a concrete question.
- For few-shot adaptation, compare fixed numbers of tracks with percentages only after target-size and class-balance inspection.
- Choose adaptation tracks from training scenes only, with a predeclared sampling method and repeated draws if small samples are unstable; never select them using held-out performance.

For a behavior-unlabeled deployment dataset such as ECP2.0, automatic labels alone are not sufficient evaluation. Candidate evidence includes a predeclared manual audit, agreement with derived events where defensible, and calibration transfer. The acceptable manual-validation amount is open.
