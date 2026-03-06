# Pilot Study: SNS Engagement & Mental Health Decomposition

## Background

This pilot extends the "Cognitive Obesity" framework by decomposing social media's
mental health impact into separable components:

1. **Writer vs ROM (Read-Only Member)**: Does *posting* cause harm, or does *passive consumption*?
2. **Evaluated vs Non-Evaluated Platforms**: Are likes/reactions the toxic ingredient?
3. **Push Notifications x Advertising**: Do interruptions and commercial content interact?

## Data Sources

| Script | Data Source | Access | Status |
|--------|-----------|--------|--------|
| `01_writer_vs_rom.py` | Understanding Society Wave 11 (UK) | UK Data Service registration (free) | Pipeline ready |
| `02_reddit_engagement_spectrum.py` | Reddit Mental Health Dataset | Zenodo open download | Pipeline ready |
| `00_synthetic_data.py` | Generated | Automatic | Available |

### Understanding Society Wave 11
- **URL**: https://www.understandingsociety.ac.uk/
- **Key variables**: viewing frequency, posting frequency, GHQ-12
- **Sample**: ~15,800 UK adults, longitudinal
- **Place data in**: `data/pilot/understanding_society/`

### Reddit Mental Health Dataset
- **URL**: https://zenodo.org/records/3941387
- **Key variables**: Posts from 28 subreddits (15 mental health support groups)
- **Sample**: ~3.4M posts, ~2.4M users
- **Place data in**: `data/pilot/reddit_mental_health/`

## Running

```bash
# Generate synthetic data for pipeline testing
python analysis/pilot/sns_engagement/00_synthetic_data.py

# Run with synthetic data (fallback) or real data
python analysis/pilot/sns_engagement/01_writer_vs_rom.py
python analysis/pilot/sns_engagement/02_reddit_engagement_spectrum.py
```

## Outputs

- `results/pilot_writer_vs_rom.json`
- `results/pilot_reddit_engagement.json`
- `results/figures/pilot_writer_vs_rom_interaction.png`
- `results/figures/pilot_reddit_engagement_spectrum.png`

## Experiment Designs (data not yet available)

See `docs/experiment_design/sns_decomposition_design.md` for:
- Evaluated vs Non-Evaluated platform comparison design
- Push Notification x Advertising 2x2 RCT design
