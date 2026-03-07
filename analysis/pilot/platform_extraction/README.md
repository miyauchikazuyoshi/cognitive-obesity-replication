# Pilot: Platform Investment-to-Extraction Phase Transition

## Research Question

Do major social media platforms exhibit a measurable transition from
"investment phase" (user-growth focused) to "extraction phase"
(monetization-focused), and does this transition correlate with declining
user welfare?

## Scripts

| Script | Description | Data Required |
|--------|-------------|---------------|
| `00_meta_arpu_attitudes.py` | Study 1: Meta ARPU vs Pew/Reuters user attitudes divergence | Compiled from SEC filings + Pew (built-in) |
| `01_msi_natural_experiment.py` | Study 2: Facebook MSI algorithm change (Jan 2018) natural experiment | Published aggregate data (built-in) |

## Running

```bash
# Study 1: ARPU × Attitudes (data is built-in)
python3 analysis/pilot/platform_extraction/00_meta_arpu_attitudes.py

# Study 2: MSI Natural Experiment (data is built-in)
python3 analysis/pilot/platform_extraction/01_msi_natural_experiment.py
```

## Connection to Cognitive Obesity

The extraction phase corresponds to platforms inflating information exposure I
while suppressing cognitive control C in the utility function L = α₁·I − α₂·C.
Measuring the extraction transition provides macro-level context for the
individual-level effects documented elsewhere in this project.

## Experiment Design Document

See: `docs/experiment_design/platform_extraction_phase_design.md`
