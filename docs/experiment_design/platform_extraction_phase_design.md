# Experiment Design: Platform Investment-to-Extraction Phase Transition

## Overview

This document designs studies to empirically measure and date the transition of
major social media platforms (Meta/Facebook, YouTube) from **investment phase**
(user acquisition, creator subsidies) to **extraction phase** (attention monetization,
ad density maximization). The core hypothesis is that this transition degrades
user welfare while increasing engagement metrics — a divergence central to the
Cognitive Obesity framework.

**Parent project**: [Cognitive Obesity Replication Archive](../../README.md)

**Connection to Cognitive Obesity**: In the framework L = α₁·I − α₂·C, the
extraction phase corresponds to platforms artificially inflating I (information
exposure via algorithmic engagement optimization) while suppressing users' ability
to exercise C (cognitive control), resulting in net utility loss L despite
increased time-on-platform.

---

## Theoretical Framework

### Platform Lifecycle Model

```
Phase 1: Investment      → Subsidize creators, organic reach high, low ad load
Phase 2: Maturity        → Monetization begins, balanced value exchange
Phase 3: Extraction      → Ad density increases, algorithmic engagement maximization,
                           creator revenue squeezed, user satisfaction declines
Phase 4: Re-investment?  → Competitor threat (TikTok) forces temporary re-investment
```

### Key Prediction

> **The extraction phase creates a measurable divergence between platform engagement
> metrics (DAU, time-on-platform) and user welfare metrics (life satisfaction,
> self-reported social media attitudes, GHQ-12).** This divergence should be
> detectable as a structural break in the time-series relationship between the two.

### Operationalization

| Construct | Platform-Side Metric | User-Side Metric |
|-----------|---------------------|------------------|
| Extraction intensity | ARPU, ad impressions/user, ad revenue growth | Ad annoyance, perceived ad density |
| Engagement optimization | DAU/MAU ratio, time-on-platform | Self-reported compulsive use, regret |
| User welfare | (not directly measured) | Life satisfaction, social media attitudes, GHQ |
| Creator value | Creator fund payouts, CPM rates | Creator satisfaction surveys |

---

## Study 1: ARPU × Attitudes Divergence (Time-Series Analysis)

### Research Question
Does increasing ad monetization intensity (ARPU) predict declining user attitudes
toward social media, after controlling for overall platform adoption trends?

### Design
**Interrupted time-series with structural break detection**

- **Unit of analysis**: Year × platform (2012–2025)
- **Treatment**: Platform phase transition (endogenously estimated via breakpoint detection)

### Data Sources (All Free, Publicly Available)

| Source | Variables | Years | Access |
|--------|----------|-------|--------|
| Meta 10-K SEC filings | ARPU (by region), ad impressions growth %, DAU/MAU | 2012–2025 | Free (SEC EDGAR) |
| Alphabet 10-K SEC filings | YouTube ad revenue (from 2019) | 2019–2025 | Free (SEC EDGAR) |
| Pew Research ATP | "Social media mostly good/bad for society" | 2018–2025 | Free (registration) |
| Pew Social Media Fact Sheet | Platform adoption %, usage frequency | 2005–2025 | Free |
| Reuters Digital News Report | Trust in news via social media, by platform | 2012–2025 | Chart data free |

### Variables

**Dependent Variables (User Welfare)**:
- `pew_bad_pct`: % saying social media is "mostly bad for society" (Pew ATP)
- `reuters_trust`: Trust in news found via [platform] (Reuters DNR, 0-10 scale)
- `news_avoidance`: % actively avoiding news (Reuters DNR)

**Independent Variables (Extraction Intensity)**:
- `meta_arpu_us`: Meta Average Revenue Per User, US & Canada (quarterly, from 10-K)
- `meta_ad_impressions_growth`: YoY % change in ad impressions delivered
- `meta_ad_price_growth`: YoY % change in average price per ad
- `yt_ad_revenue`: YouTube advertising revenue (quarterly, from 10-K)

**Control Variables**:
- `platform_dau`: Daily active users (controls for scale effects)
- `smartphone_penetration`: % of adults with smartphones (Pew)
- `tiktok_adoption`: TikTok adoption rate (controls for competitive re-investment)

### Analytical Approach

1. **Divergence Index Construction**:
   ```
   DI_t = z(engagement_t) - z(satisfaction_t)
   ```
   Where z() is within-variable standardization. DI > 0 indicates extraction
   (engagement up, satisfaction down).

2. **Structural Break Detection**:
   - Bai-Perron test for multiple structural breaks in DI_t
   - Chow test at hypothesized breakpoints (2017 Adpocalypse, 2018 MSI change)
   - Quandt Likelihood Ratio for unknown breakpoint

3. **Granger Causality Test**:
   - Does ARPU growth Granger-cause attitude decline?
   - Does attitude decline Granger-cause DAU decline (with lag)?

4. **Event Study / Interrupted Time Series**:
   - Events: Jan 2018 (Facebook MSI algorithm), 2017 Q1 (YouTube Adpocalypse),
     2021 Q2 (Apple ATT), 2022 Q4 (Meta cost cuts)
   - ARIMA with intervention dummies

### Expected Results

| Hypothesis | Test | Expected Direction |
|-----------|------|-------------------|
| H1: ARPU↑ → Attitude↓ | Granger causality | ARPU leads attitude decline by 1-2 quarters |
| H2: Structural break exists | Bai-Perron | Break at 2017–2018 for both platforms |
| H3: Divergence accelerates | DI trend test | DI slope increases post-break |
| H4: TikTok competition → temporary re-convergence | DI dip test | DI decreases 2022–2023 |

### Power and Feasibility

- **Observations**: ~52 quarters (2012Q1–2025Q4) for Meta; ~24 quarters for YouTube
- **Challenge**: Small T for time-series analysis
- **Mitigation**: Supplement with annual cross-national data from Reuters DNR
  (48 countries × 14 years = 672 country-year observations)
- **Implementation time**: ~2 weeks for data compilation + analysis

---

## Study 2: Natural Experiment — Facebook MSI Algorithm Change (Jan 2018)

### Research Question
Did Facebook's January 2018 algorithm change (prioritizing "Meaningful Social
Interactions" over publisher content) function as a measurable extraction-phase
event that increased engagement while degrading user experience?

### Design
**Regression Discontinuity in Time (RDiT)**

The MSI change was a sharp, exogenous policy shift announced on January 11, 2018.

### Data Sources

| Source | Pre-period | Post-period | Access |
|--------|-----------|-------------|--------|
| Meta 10-K | 2015–2017 | 2018–2020 | Free |
| Pew ATP Wave 25 (Mar 2017) | Pre | | Free |
| Pew ATP Wave 51 (Jul 2019) | | Post | Free |
| Reuters DNR (2016, 2017, 2018, 2019) | Both | Both | Free (chart data) |
| Frances Haugen leaked documents (2021) | Internal metrics | | Public (SEC testimony) |

### Variables

**Outcome measures (user welfare)**:
- Change in "Facebook is good for society" (Pew, pre vs post)
- Change in trust in news via Facebook (Reuters DNR)
- Change in Facebook-specific news avoidance

**Mechanism measures (engagement)**:
- Meta DAU/MAU ratio (proxy for engagement stickiness)
- Meta ARPU (monetization intensity)
- Time spent on Facebook (internal metric, cited in press releases)

**Platform response measures (extraction indicators)**:
- Organic reach for Pages (documented ~50% decline)
- Publisher traffic from Facebook (Chartbeat/SimilarWeb data, cited in industry reports)

### Analytical Approach

1. **RDiT around Jan 2018**:
   - Running variable: months from Jan 2018
   - Bandwidth selection: Imbens-Kalyanaraman optimal bandwidth
   - Local linear regression with triangular kernel

2. **Difference-in-Differences**:
   - Treatment: Facebook users
   - Control: YouTube-only users (not affected by MSI change)
   - Pre: 2016–2017; Post: 2018–2019
   - Outcome: Platform satisfaction / trust

3. **Synthetic Control**:
   - Construct synthetic Facebook from weighted combination of other platforms
   - Compare actual vs synthetic Facebook satisfaction post-MSI

### Internal Validity Threats

| Threat | Mitigation |
|--------|-----------|
| Concurrent events (Cambridge Analytica, Mar 2018) | Include interaction term; separate event windows |
| Selection (who leaves Facebook) | Use population-level surveys (Pew), not user-level data |
| Measurement (attitude ≠ welfare) | Triangulate with GHQ/life satisfaction from GSS |

---

## Study 3: Ad Density × Time Spent × Satisfaction (Cross-Sectional Decomposition)

### Research Question
Among current social media users, does the ratio of perceived ad density to
content satisfaction mediate the relationship between time spent and subjective
wellbeing?

### Design
**Cross-sectional mediation analysis** (requires new survey data)

### Proposed Survey Instrument

**Sample**: n = 1,000 adults (Prolific/MTurk), stratified by platform use

**Measures**:

1. **Platform use** (per platform: Facebook, YouTube, Instagram, TikTok, X):
   - Daily time spent (self-report + screen time screenshot validation)
   - Passive vs active use ratio (scrolling vs posting/commenting)

2. **Perceived extraction indicators** (novel scale, 5 items per platform):
   - "I see more ads on [platform] than I used to" (1-7)
   - "The content [platform] shows me is what I want to see" (1-7, reverse)
   - "[Platform] makes it hard to stop using" (1-7)
   - "I feel [platform] values my time" (1-7, reverse)
   - "I feel [platform] is designed for advertisers, not for me" (1-7)

3. **Wellbeing outcomes**:
   - GHQ-12 (General Health Questionnaire)
   - SWLS (Satisfaction with Life Scale)
   - PHQ-4 (ultra-brief depression/anxiety)
   - Single-item life satisfaction (0-10)

4. **Controls**:
   - Age, sex, income, education
   - Trait self-control (Brief Self-Control Scale)
   - Social comparison orientation (INCOM-6)

### Analytical Approach

```
Path Model:
  Time Spent → Perceived Extraction → Wellbeing (-)
  Time Spent → Content Satisfaction → Wellbeing (+)

  Mediation: Does perceived extraction suppress the positive path?
```

1. **Structural Equation Modeling (SEM)**:
   - Direct effect: Time → Wellbeing
   - Indirect via extraction: Time → Extraction → Wellbeing (−)
   - Indirect via satisfaction: Time → Satisfaction → Wellbeing (+)
   - Platform as moderator (Facebook vs YouTube vs TikTok)

2. **Platform comparison**:
   - Older platforms (FB, YT) expected to show stronger extraction path
   - Newer/competing platforms (TikTok) expected to show weaker extraction path

### Power Analysis
- Medium effect (f² = 0.05) for mediation paths
- 5 predictors + 3 mediators
- Required: n ≈ 400 (80% power, α = .05)
- Planned: n = 1,000 (allows platform-level subgroup analysis)

### Budget Estimate
- Prolific sample (n=1,000, 15-min survey, £1.50/participant): ~$2,000
- Total with platform fees and pilot: ~$2,800

---

## Study 4: Longitudinal Panel — GSS + Digital Society Module (Secondary Analysis)

### Research Question
Does the 2024 GSS Digital Society module, combined with long-running wellbeing
items, reveal a cohort-level divergence between internet adoption and happiness?

### Design
**Repeated cross-sectional analysis** of GSS 1972–2024

### Data Source
- General Social Survey (GSS): Free, no registration required
- URL: https://gssdataexplorer.norc.org/
- Format: Stata, SPSS, R, CSV

### Variables

**Wellbeing (available 1972–2024)**:
- `happy`: "Taken all together, how happy would you say you are?" (3-point)
- `trust`: "Can most people be trusted?" (binary)
- `satfin`: Financial satisfaction
- `health`: Self-rated health

**Digital variables (available ~2000s onward)**:
- `usewww` / `wwwhr`: Internet use and hours
- `socrel`: Social evening with relatives frequency
- `socfrnd`: Social evening with friends frequency
- New 2024 Digital Society items (social media use, platform trust, digital wellbeing)

### Analytical Approach

1. **Divergence Detection**:
   - Plot z(happiness) vs z(internet_use) from 2000–2024
   - Identify divergence onset year
   - Compare with platform lifecycle milestones

2. **Age-Period-Cohort Decomposition**:
   - Is the happiness decline driven by period effects (platform extraction)
     or cohort effects (digital natives)?
   - Use intrinsic estimator for APC model

3. **2024 Deep Dive**:
   - Cross-tabulate new digital society items with happiness/trust
   - Test: heavy social media users with high perceived extraction report
     lower happiness than heavy users with low perceived extraction

### Feasibility
- Data already exists and is free
- No new data collection needed
- Implementation time: ~1 week

---

## Data Collection Priority

| Study | Data Status | New Collection? | Timeline | Cost |
|-------|------------|----------------|----------|------|
| Study 1 (ARPU × Attitudes) | Public data exists | No (compilation) | 2 weeks | $0 |
| Study 2 (MSI Natural Experiment) | Public data exists | No (compilation) | 2 weeks | $0 |
| Study 4 (GSS Analysis) | Public data exists | No (download) | 1 week | $0 |
| Study 3 (Survey) | Requires new survey | Yes | 4-6 weeks | ~$2,800 |

**Recommended order**: Study 4 → Study 1 → Study 2 → Study 3

Study 4 is the quickest win (GSS data is free and immediately available).
Studies 1 & 2 use the same compiled dataset (SEC filings + survey data).
Study 3 requires ethics approval and budget.

---

## Connection to Existing Pilot Studies

This document extends the pilot infrastructure at `analysis/pilot/`:

| Existing Pilot | Relationship |
|---------------|-------------|
| Writer vs ROM (Understanding Society) | **Complementary**: ROM behavior = passive consumption in extraction phase |
| Reddit Engagement Spectrum | **Complementary**: Engagement tiers map to investment/extraction user archetypes |
| SNS Decomposition (eval/non-eval) | **Upstream**: Evaluation mechanisms are tools of extraction |
| **This study** | **Macro-level**: Tests whether platform-level extraction predicts population-level welfare decline |

---

## References

- Allcott, H., et al. (2020). The welfare effects of social media. *AER*, 110(3), 629-676.
- Haugen, F. (2021). SEC testimony and leaked Facebook documents.
- Shakya, H. B., & Christakis, N. A. (2017). Association of Facebook use with compromised well-being. *Am J Epidemiology*, 185(3), 203-211.
- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. *J Applied Econometrics*, 18(1), 1-22.
- Meta Platforms, Inc. (2024). Annual Report (Form 10-K). SEC EDGAR.
- Alphabet Inc. (2024). Annual Report (Form 10-K). SEC EDGAR.
- Newman, N., et al. (2025). Reuters Institute Digital News Report 2025. University of Oxford.
- Smith, A., et al. (2018-2025). Social Media Use surveys. Pew Research Center.
