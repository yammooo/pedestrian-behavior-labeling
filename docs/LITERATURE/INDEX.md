# Literature index

Status: Living index  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: navigation and brief comparison of papers.
- Links out: methods and evidence to individual paper notes, and qualified gap claims to [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

| Paper | Main relevance | Online/offline distinction | What it already covers |
|---|---|---|---|
| [ARCANE-PedSynth / PedSynth++](paper_notes/arcane_pedsynth.md) | Candidate synthetic behavior-supervision source | Generated labels may supervise offline annotation; paper's application framing includes prediction | Synthetic multi-pedestrian data with 12-state behavior FSM |
| [LOKI](paper_notes/loki.md) | Rich tracks and action/intention labels | Source actions are frame-wise; paper derives future-action intention | Multimodal trajectory and intention benchmark |
| [Pedestrian Stop and Go Forecasting](paper_notes/stop_go.md) | Transitions and temporal dynamics | Forecasting | Stop/go prediction benchmark and hybrid fusion |
| [EMT](paper_notes/emt.md) | Cross-domain candidate with overlapping labels | Intention/trajectory benchmark; exact semantics to inspect | Gulf-region driving benchmark |
| [EuroCity Persons 2.0](paper_notes/ecp2.md) | Large tracked enrichment target | Track pseudo-GT generation, not behavior labeling | Dense person trajectory generation |
| [Minimizing Human Labeling](paper_notes/minimizing_human_labeling.md) | Existing automatic C/NC pseudo-labeling | Prediction/self-labeling | Synthetic-to-real pseudo-labeling and temporal smoothing |
| [PIE](paper_notes/pie.md) | Crossing-label semantics | Often prediction-focused | Pedestrian intention/crossing benchmark |
| [JAAD](paper_notes/jaad.md) | Crossing-label semantics | Often prediction-focused | Joint attention / action benchmark |
| [Rasouli & Kotseruba taxonomy](paper_notes/rasouli_kotseruba.md) | Task-definition vocabulary | Explicitly distinguishes tasks | Conceptual taxonomy, not an annotation pipeline |

See [GAP_ANALYSIS.md](GAP_ANALYSIS.md) for deliberately qualified evidence around possible contribution space. Only papers that change a project decision, evaluation design, or baseline choice should be added.
