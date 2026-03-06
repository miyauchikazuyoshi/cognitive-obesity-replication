# Experiment Design: SNS Toxicity Decomposition

## Overview

This document outlines experiment designs for two research axes where public data
is not currently available. These designs extend the "Cognitive Obesity" framework
by isolating specific mechanisms through which social media may affect mental health.

**Parent project**: [Cognitive Obesity Replication Archive](../../README.md)

**Related pilot analyses** (data available):
- `analysis/pilot/sns_engagement/01_writer_vs_rom.py` — Writer vs ROM using Understanding Society
- `analysis/pilot/sns_engagement/02_reddit_engagement_spectrum.py` — Reddit engagement spectrum

---

## Study 1: Evaluated vs Non-Evaluated Platform Comparison

### Research Question
Does the presence of social evaluation mechanisms (likes, reactions, follower counts)
drive the mental health impact of social media, independent of content consumption?

### Hypotheses
- **H1**: Users of evaluated platforms (Instagram, Twitter/X) report higher depression
  and anxiety symptoms than users of non-evaluated platforms (Signal, Discord DMs,
  anonymous forums without voting).
- **H2**: The effect of platform evaluation on mental health is mediated by
  social comparison frequency.
- **H3**: The evaluation effect is moderated by trait self-esteem: low self-esteem
  individuals are more vulnerable.

### Design Options

#### Option A: Within-Subject Crossover (Recommended)
- **Participants**: N = 180 (active users of both evaluated and non-evaluated platforms)
- **Duration**: 4 weeks (2 weeks per condition, counterbalanced)
- **Conditions**:
  - Period 1: Use only evaluated platforms (Instagram/Twitter) for social interaction
  - Period 2: Use only non-evaluated platforms (Signal/Discord) for social interaction
  - (Order counterbalanced)
- **Washout**: 3-day washout between periods
- **Measurements** (ESM, 3x daily):
  - Momentary affect (PANAS-Short)
  - Social comparison frequency (single item)
  - Screen Time API data (passive logging)
- **Pre/Post per period**:
  - PHQ-9 (depression)
  - GAD-7 (anxiety)
  - Rosenberg Self-Esteem Scale
  - Social Media Social Comparison Scale

**Power analysis**: Within-subject crossover, d = 0.25, alpha = 0.05, power = 0.80
→ n ≈ 130. Recruit n = 180 for ~30% attrition.

#### Option B: Between-Subject with Natural Groups
- **Participants**: N = 400 (200 primarily-evaluated, 200 primarily-non-evaluated users)
- **Recruitment**: Screen for dominant platform type (>70% of social time on one type)
- **Design**: Cross-sectional survey + 2-week ESM diary
- **Advantage**: No behavior change required
- **Disadvantage**: Self-selection confounds

#### Option C: Natural Experiment — Instagram Hidden Likes
- Exploit Instagram's 2019 hidden likes experiment in select countries
- **Data**: Scrape public sentiment indicators pre/post policy change
- **Advantage**: Quasi-experimental
- **Disadvantage**: Ecological data only, no individual-level mental health

### Key Measures

| Measure | Instrument | Timing |
|---------|-----------|--------|
| Depression | PHQ-9 | Pre/post each period |
| Anxiety | GAD-7 | Pre/post each period |
| Momentary affect | PANAS-Short (10 items) | ESM 3x/day |
| Social comparison | Single item (1-5) | ESM 3x/day |
| Self-esteem | Rosenberg (10 items) | Baseline |
| Screen time | iOS/Android API | Continuous passive |
| Platform usage | Self-report + API | Daily |

### Analysis Plan
1. Mixed-effects model: PHQ-9 ~ condition × period + (1|participant)
2. Mediation analysis: condition → social comparison → PHQ-9
3. Moderation: condition × baseline self-esteem → PHQ-9
4. ESM trajectories: multilevel growth curve for momentary affect

---

## Study 2: Push Notification × Advertising 2×2 RCT

### Research Question
Do push notifications and advertising on social media independently and
interactively affect mental health and usage patterns?

### Hypotheses
- **H1 (Notification main effect)**: Disabling push notifications reduces
  depression symptoms and compulsive checking behavior.
- **H2 (Advertising main effect)**: Blocking social media ads reduces
  materialistic values and appearance anxiety.
- **H3 (Interaction)**: The combination of notifications + ads produces
  a synergistic (super-additive) effect on mental health deterioration.
  Rationale: notifications pull users back into ad-laden feeds, creating
  a reinforcement loop.

### Design
**2×2 Between-Subject RCT**

|                  | Ads Present     | Ads Blocked        |
|------------------|-----------------|--------------------|
| **Push ON**      | Cell 1 (control) | Cell 2 (ads-only)  |
| **Push OFF**     | Cell 3 (notif-only) | Cell 4 (both off) |

- **Participants**: N = 280 (70 per cell)
- **Duration**: 2 weeks intervention + 1 week follow-up
- **Platform**: Instagram (most studied, largest ad load)

### Intervention Protocol

#### Push Notification Manipulation
- **ON condition**: Default settings (all notifications enabled)
- **OFF condition**: Participant disables ALL Instagram notifications
  - Verified via screenshot of notification settings
  - Daily compliance check via brief survey

#### Advertising Manipulation
- **Present condition**: Normal Instagram use
- **Blocked condition**: Install DNS-level ad blocker (e.g., NextDNS, AdGuard)
  - Blocks Instagram in-feed ads and story ads
  - Does not block organic sponsored content
  - Verified via periodic screenshot requests

### Measurements

**Baseline (Day 0)**:
- PHQ-9, GAD-7
- Materialism Values Scale (MVS-Short)
- Social Media Addiction Scale (BSMAS)
- Instagram usage: Screen Time screenshot

**Daily diary (Days 1-14)**:
- Single-item mood (1-10)
- Number of Instagram opens (self-report + Screen Time)
- Perceived ad exposure (1-5)
- Notification-triggered opens (self-report)

**Post-intervention (Day 15)**:
- PHQ-9, GAD-7
- MVS-Short
- BSMAS
- Screen Time screenshot
- Qualitative: "How did the intervention affect your experience?"

**Follow-up (Day 22)**:
- PHQ-9, GAD-7
- Did you maintain the changes? (Y/N + reason)

### Power Analysis
- Primary outcome: PHQ-9 change score
- Expected interaction effect: d = 0.25 (small-medium)
- 2×2 ANOVA interaction: alpha = 0.05, power = 0.80
- Required: n ≈ 64 per cell → n = 256 total
- With 10% attrition: recruit n = 280

### Analysis Plan
1. **Primary**: 2×2 ANOVA on ΔPHQ-9 (pre-post change)
2. **Secondary**: Separate ANOVAs on ΔGAD-7, ΔBSMAS, ΔScreen Time
3. **Interaction decomposition**: Simple effects if interaction is significant
4. **Daily trajectory**: Multilevel growth curve for daily mood
5. **Mediation**: Notification condition → checking frequency → ΔPHQ-9
6. **Sensitivity**: Per-protocol analysis excluding non-compliant participants

### Ethical Considerations
- No deception: participants know they are in an experiment
- No harm expected: interventions reduce exposure, not increase it
- Participants can withdraw at any time
- IRB approval required before recruitment
- Data stored on encrypted servers, anonymized for analysis

---

## Feasibility & Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Preparation | 4 weeks | IRB submission, instrument finalization, pilot (n=10) |
| Recruitment | 2 weeks | Online recruitment via Prolific or university panels |
| Study 1 | 4 weeks | Crossover intervention + ESM |
| Study 2 | 3 weeks | 2×2 RCT + follow-up |
| Analysis | 4 weeks | Data cleaning, modeling, manuscript drafting |
| **Total** | **~17 weeks** | |

### Budget Estimate (Study 1 + Study 2 combined)

| Item | Cost |
|------|------|
| Participant compensation (480 × $30) | $14,400 |
| Prolific platform fees (~30%) | $4,320 |
| ESM platform (e.g., ExperienceSampler) | $500 |
| Ad blocker licenses (if needed) | $200 |
| Qualtrics/REDCap survey hosting | $0 (institutional) |
| **Total** | **~$19,420** |

---

## Connection to Cognitive Obesity Framework

These experiments directly test the "processing capacity" mechanism proposed
in the parent paper:

1. **Evaluation mechanism** → maps to the "social comparison overload" component
   of cognitive load. If likes/reactions are the primary driver, the cognitive
   obesity metaphor gains specificity: it's not information *volume* but
   information *valence variability* (intermittent social reinforcement).

2. **Push notifications** → map to the "interruption cost" component.
   Each notification forces a context switch, consuming processing capacity
   regardless of content quality.

3. **Advertising** → maps directly to the "ad_proxy" variable in the macro
   analysis. If ad blocking improves mental health, this validates the proxy
   construction at the individual level.

The 2×2 design specifically tests whether the **interaction** of these
components (as the macro model's multiplicative proxy implies) exceeds
the sum of their individual effects.
