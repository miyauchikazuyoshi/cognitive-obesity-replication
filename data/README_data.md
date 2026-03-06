# データソース一覧

本リポジトリで使用する全データはパブリックドメインまたはオープンアクセス。
再配布は行わず、取得スクリプトを提供する。

まず `python data/check_data.py` を実行すると、手元で不足しているファイルと次の手順が表示される。

## 自動取得可能

| データ | スクリプト | ライセンス |
|---|---|---|
| NHANES 2017-2018 | `download_nhanes.py` | Public domain (CDC/NCHS) |
| ATUS 2010-2013 | `download_atus.py` | Public domain (BLS) ※環境により手動が必要 |
| World Bank WDI | `download_macro.py` | CC-BY 4.0 |
| WHO GHO (身体不活発率・自殺率) | `download_who_gho.py` | WHO terms (API) |
| OWID (平均就学年数・UNODC殺人率) | `download_owid.py` | OWID terms (source-specific) |

## 手動ダウンロード必要

| データ | URL | 取得方法 |
|---|---|---|
| ATUS 2010-2013 (BLS) | https://www.bls.gov/tus/data/ | ZIPをブラウザで取得（bot対策で403になる場合あり） |
| IHME GBD Results | https://vizhub.healthdata.org/gbd-results/ | Web UIからCSVダウンロード |
| WHO NCD Profiles | https://www.who.int/teams/noncommunicable-diseases/surveillance/data/ | （任意）GHO APIが使えない場合の代替 |

### ATUS (BLS) 取得手順（403になる場合）

`python data/download_atus.py` が `HTTP Error 403: Forbidden` を返す場合、BLS側のbot対策が発動しています。
その場合は以下で対応してください。

1. ブラウザで以下の公式ページから ZIP をダウンロード（ファイル名はそのまま保持）:
   - ATUS multi-year (2003-2024): https://www.bls.gov/tus/data/datafiles-0324.htm
     - `atusact-0324.zip`（Activity）
     - `atusresp-0324.zip`（Respondent）
   - Well-Being module (2010/2012/2013 multi-year): https://www.bls.gov/tus/data/wbdatafiles_1013.htm
     - `wbresp_1013.zip`（WB Respondent; Cantril ladder など）
     - （任意）`wbact_1013.zip`（WB Activity; 本リポジトリでは未使用だが拡張用）
2. `data/atus/` に ZIP を配置
3. 展開のみ実行:

```bash
python data/download_atus.py
```

このスクリプトは、ZIPが既に存在する場合はダウンロードをスキップし、展開だけを行います。
（BLS公式の推奨どおり、リンクを右クリックして「名前を付けてリンク先を保存」すると通る場合があります）

### IHME GBD 取得手順

1. https://vizhub.healthdata.org/gbd-results/ にアクセス
2. **鬱病（必須）**:
   - Cause: Depressive disorders
   - Measure: Prevalence
   - Metric: Rate
   - Location: All countries and territories
   - Age: Age-standardized
   - Year: 1990-2023
3. （任意）自殺・殺人もIHMEで揃える場合:
   - Self-harm / Interpersonal violence
   - Measure: Deaths, Metric: Rate
   - その他の設定は上と同じ
4. CSVをダウンロードし `data/macro/` に配置し、ファイル名を以下にリネーム:
   - `ihme_depression.csv`（必須）
   - `ihme_suicide.csv`（任意）
   - `ihme_homicide.csv`（任意）

※ もし誤って複数の Age/Sex/Measure/Metric を含む CSV を落としても、
`data/build_macro_panel.py` は列が存在する限り **Age-standardized × Both × (Prevalence/Deaths) × Rate**
に自動で絞り込む（該当スライスが含まれている必要あり）。

### 配置後のディレクトリ構造

```
data/
├── nhanes/          ← download_nhanes.py が生成
│   ├── DEMO_J.XPT
│   ├── DPQ_J.XPT
│   ├── PAQ_J.XPT
│   ├── BMX_J.XPT
│   └── HIQ_J.XPT
├── atus/            ← download_atus.py が生成
│   ├── atusact_0324.dat
│   ├── wbresp_1013.dat
│   └── atusresp_0324.dat
└── macro/           ← download_macro.py + 手動
    ├── worldbank_wdi.csv
    ├── owid_mean_years_of_schooling.csv
    ├── owid_homicide_rate_unodc.csv
    ├── ihme_depression.csv
    ├── ihme_suicide.csv              （任意）
    ├── ihme_homicide.csv             （任意）
    └── who_ncd_pac.csv
    └── who_suicide_mh12.csv
```

## マクロパネルの組み立て（重要）

Block B / Block C-macro の分析スクリプトは、上記CSVを直接読むのではなく、
組み立て済みパネルを参照する。

以下を実行して `panel_merged.csv` を生成する:

```bash
python data/build_macro_panel.py
```

出力（`data/macro/`）:

- `panel_merged.csv`（World Bank + IHME を統合）
- `panel_with_inactivity.csv`（WHO が統合できた場合のみ）

国名の表記ゆれ等で結合に失敗する国が出た場合、スクリプトが警告を出す。
その場合は、手元でCSVの列名/国名を整形して再実行する。
