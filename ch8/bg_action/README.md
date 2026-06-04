# BG Action Selection Model

This simulation implements a biologically realistic basal ganglia (BG) model for the **Probabilistic Selection (PS) task** (Frank et al., 2004). It is translated from the emergent 7.0+ project file `BG_4s_inhib_PS_e7a.proj`.

## Key Features

Unlike the simplified BG model (Chapter 8, `bg` sim) where actions are directly input to the network, this model:

1. **Selects its own actions** via disinhibitory gating of thalamus → premotor cortex
2. **Explores** via noise in premotor cortex (VM_NOISE with a declining schedule)
3. **Learns** reward associations through Go/NoGo pathways modulated by dopamine (D1/D2 rules)
4. **Simulates Parkinson's Disease** by reducing intact SNc dopamine units
5. **Simulates D2 agonist effects** by modulating D2 projection strength

## Network Architecture

Translated from `BG_4s_inhib_PS_e7a.proj`:

| Layer | Shape | Description |
|---|---|---|
| Input | 6×3 (18) | Stimulus representation (S1, S2) |
| Context | 9×2 (18) | Contextual state information |
| Go | 6×6 (36) | D1R medium spiny neurons (direct pathway) |
| NoGo | 6×6 (36) | D2R medium spiny neurons (indirect pathway) |
| StriatumInhib | 4×4 (16) | Striatal inhibitory interneurons |
| GPe | 2×2 (4) | External globus pallidus |
| GPi | 4×2 (8) | Internal globus pallidus / SNr |
| Thalamus | 2×2 (4) | Thalamus (gating output) |
| PMC | 4×2 (8) | Premotor cortex (action representation) |
| Output | 4×1 (4) | Response output |
| SNc | 2×2 (4) | Dopamine neurons (externally clamped) |

### BG Circuit

```
Input/Context → Go (D1) → GPi ⊣ Thalamus → PMC → Output
                 ↑ SNc ↑         ↑
Input/Context → NoGo (D2) → GPe ⊣ GPi
                 ↑ SNc (inhib) ↑
PMC ←→ Thalamus (recurrent)
```

## Task

The **Probabilistic Selection Task** (Frank et al., 2004):

### Training Phase
- **S1 trials (8020_R1R2)**: R1 rewarded 80%, R2 rewarded 20%
- **S2 trials (6040_R3R4)**: R3 rewarded 60%, R4 rewarded 40%

Candidate responses are biased in PMC so that only relevant actions compete (R1/R2 for S1; R3/R4 for S2).

### Test Phase
Measures **Choose-A** (selecting the most rewarded response) and **Avoid-B** (avoiding the most punished response), demonstrating Go vs. NoGo learning dissociation.

## Parameters

### DA Manipulation

| Parameter | Default | Description |
|---|---|---|
| `NumIntactSNc` | 4 | Intact DA neurons (4=healthy, 2=PD) |
| `D1Acq` / `D1Perf` | 0.6 | D1 projection strength (acq/perf) |
| `D2Acq` / `D2Perf` | 0.075 | D2 projection strength (acq/perf) |
| `DABurstVal` | 1.0 | SNc value for DA burst (reward) |
| `DADipVal` | 0.0 | SNc value for DA dip (punishment) |
| `TonicDA` | 0.026 | Tonic DA level |

### Noise Schedule (PMC exploration)
Follows the original emergent 7 schedule:
- Epochs 0–25: noise × 1.0 (full exploration)
- Epochs 25–80: noise ramps 1.0 → 0.5
- Epochs 80–100: noise ramps 0.5 → 0.2

## Mapping from BG_4s_inhib_PS_e7a.proj

### What was matched
- All 11 layers with their unit counts and shapes
- Full projection connectivity (Input/Context → Go/NoGo, Go → GPi, NoGo → GPe → GPi, GPi ⊣ Thalamus, Thalamus ↔ PMC)
- D1/D2 dopamine modulation of Go/NoGo learning
- SNc DA burst/dip reward delivery logic
- PMC noise-based exploration
- Probabilistic reward schedules (80/20, 60/40)
- PMC bias for candidate response selection
- PD simulation via SNc unit reduction
- D2 agonist simulation via projection strength
- Program flow: minus phase → determine action → deliver reward → plus phase → DA-modulated learning

### Unavoidable Differences from Emergent 7

1. **Inhibition**: The old model used `UNIT_INHIB` (unit-level inhibition) for most BG layers. The Go leabra implementation uses `FFFB` inhibition. Parameters (Gi, FB) are set to approximate the original behavior.

2. **Noise**: The original used `VM_NOISE` with a `noise_sched` controlling noise level across epochs. This implementation uses `GeNoise` (conductance noise) with programmatic schedule control.

3. **Tessellation projections**: The old model used `TesselPrjnSpec` for topographic SNc→Go/NoGo connectivity (mapping specific SNc units to specific striatal pools). This implementation uses `Full` connectivity with DA modulation applied uniformly.

4. **Bias weight manipulation**: The old model set bias weights on PMC units via `SetCnValName()` to bias candidate responses. This implementation uses external input (`Ext`) on PMC neurons.

5. **Connection specs**: The old model had detailed `wt_scale` parameters for each connection spec. Key values (abs, rel) are preserved but some nuanced parameters (e.g., `savg_cor`, specialized `hebb` ratios for each spec) may differ.

6. **STN**: The old model had an STN (subthalamic nucleus) component with specialized connections. The STN is not explicitly modeled as a separate layer in this translation but its effect (diffuse excitation of GPi) is approximated through GPi excitability settings.

7. **No separate `Striatum_Inhib` feedback/feedforward interneuron spec**: The old model distinguished `FBtoInhib` and `FFtoInhib` connection specs. This implementation groups them under generic inhibitory connectivity.

## Running

```bash
# GUI mode
go run bg_action.go

# Batch mode (50 networks, 30 epochs)
go run bg_action.go -runs 50 -epochs 30

# Batch mode with Parkinson's Disease
go run bg_action.go -runs 50 -epochs 30 -pd
```

## References

- Frank, M. J. (2005). Dynamic dopamine modulation in the basal ganglia: A neurocomputational account of cognitive deficits in medicated and non-medicated Parkinsonism. *Journal of Cognitive Neuroscience, 17*(1), 51-72.
- Frank, M. J., Seeberger, L. C., & O'Reilly, R. C. (2004). By carrot or by stick: Cognitive reinforcement learning in Parkinsonism. *Science, 306*(5703), 1940-1943.
