# Pilot Study Data Sources

## 1. Reddit Mental Health Dataset (Automated)

**Script**: `data/download_reddit_mh.py`

```bash
python3 data/download_reddit_mh.py
```

- **Source**: Zenodo (https://zenodo.org/records/3941387)
- **Citation**: Low et al. (2020). "Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19"
- **License**: CC-BY-4.0
- **Size**: ~700 MB (2019 period only)
- **Downloads to**: `data/pilot/reddit_mental_health/raw_2019/`
- **Aggregated output**: `data/pilot/reddit_mental_health/user_engagement_summary.csv`

### What it contains
- 27 subreddits (15 mental health + 12 control), Jan-Apr 2019
- Per-post: author, subreddit, date, text features (TF-IDF, LIWC, sentiment)
- Aggregated: per-user post counts, MH subreddit participation rate, engagement tier

### Analysis
```bash
python3 analysis/pilot/sns_engagement/02_reddit_engagement_spectrum.py
```

---

## 2. Understanding Society Wave 11 (Manual Registration Required)

**No automated download** — requires free UK Data Service registration.

### Step-by-step

1. **Register** at https://ukdataservice.ac.uk/
   - Click "Register" → create account (any academic email)
   - Free for academic research

2. **Find the dataset**:
   - Go to https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=6614
   - Or search "Understanding Society" → Study Number 6614

3. **Download Wave 11 (k_indresp)**:
   - Click "Access Data" → "Download"
   - Select format: **CSV** (or Stata/SPSS)
   - Download the individual response file: `k_indresp.csv`
   - This is the Wave 11 (2019-2021) individual questionnaire

4. **Place the file**:
   ```
   data/pilot/understanding_society/k_indresp.csv
   ```

### Key variables needed

| Variable | USOC Name | Description |
|----------|-----------|-------------|
| Viewing frequency | `k_socmedia1` | How often view social media (0-7 scale) |
| Posting frequency | `k_socmedia2` | How often post on social media (0-7 scale) |
| Mental health | `k_scghq2_dv` | GHQ-12 Caseness score (0-36) |
| Age | `k_age_dv` | Age at interview |
| Sex | `k_sex` | Sex (1=male, 2=female) |

### Analysis
```bash
python3 analysis/pilot/sns_engagement/01_writer_vs_rom.py
```

---

## 3. Synthetic Data (For Pipeline Testing)

**Script**: `analysis/pilot/sns_engagement/00_synthetic_data.py`

```bash
python3 analysis/pilot/sns_engagement/00_synthetic_data.py
```

- Generates `data/pilot/sns_engagement_synthetic.csv` (survey-style, n=2,000)
- Generates `data/pilot/reddit_synthetic.csv` (Reddit-style, n=5,000)
- Used as fallback when real data is not available
- **WARNING**: Contains injected correlation structure — not for hypothesis testing!

---

## Directory Structure

```
data/pilot/
├── README_pilot_data.md              ← This file
├── sns_engagement_synthetic.csv      ← Synthetic survey data
├── reddit_synthetic.csv              ← Synthetic Reddit data
├── understanding_society/            ← Manual download
│   └── k_indresp.csv                ← USOC Wave 11
└── reddit_mental_health/             ← Automated download
    ├── user_engagement_summary.csv   ← Aggregated per-user
    └── raw_2019/                     ← Raw subreddit CSVs
        ├── depression_2019_features_tfidf_256.csv
        ├── anxiety_2019_features_tfidf_256.csv
        └── ... (27 files)
```
