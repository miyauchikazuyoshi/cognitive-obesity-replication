# Preprint Completion Guide: Cognitive Obesity 英文プレプリント完成ガイド

**作成日**: 2026-03-06
**対象文書**: English Preprint (19ページ, 不完全) vs Japanese v9_4 (41ページ, 完全版)
**目的**: 英文プレプリントの欠落部分を特定し、完成に必要な作業を体系的に整理する

---

## 1. Gap Analysis Table（差分分析表）

日本語版と英語版の構成要素を節ごとに対照し、英語版のステータスを示す。

| # | 日本語版タイトル | English Title | JP Pages | EN Pages | Status |
|---|---|---|---|---|---|
| -- | 概要 (Abstract) 日本語 | -- | 2-3 | -- | N/A (日本語のみ) |
| -- | Abstract (英語) | Abstract | 2-3 | 1 | **Missing** (キーワードのみ、本文なし) |
| -- | 筆者の前提と本論文の位置づけ | Author's Background & Paper Positioning | 3 | -- | **Missing** |
| 1 | 導入 | Introduction | 3-4 | -- | **Missing** (英語版に Section 1 なし) |
| 2 | マクロデータ分析 (Study 1) 導入 | Macro Data Analysis (Study 1) Introduction | 4 | -- | **Missing** |
| 2.1.1 | データと方法 | Data and Methods | 4-5 | -- | **Missing** |
| 2.1.2 | 主要知見（知見1-4） | Key Findings (Findings 1-4) | 5-9 | -- | **Missing** |
| 2.1.3 | 限界 | Limitations | 9-10 | -- | **Missing** |
| 2.2.1 | 因果方向の検証: ラグ付きパネル分析 | Causal Direction: Lagged Panel Analysis | 10-11 | -- | **Missing** |
| 2.2.2 | 第一近似の崩壊: TWFE | First Approximation Collapse: TWFE | 11-12 | -- | **Missing** |
| 2.2.3 | ハードアウトカム検証: 自殺率 | Hard Outcome Validation: Suicide Rate | 12-13 | -- | **Missing** |
| 2.2.4 | 閾値掃引分析 | Threshold Sweep Analysis | 13 | -- | **Missing** |
| 2.2.5 | 広告曝露プロキシ | Ad Exposure Proxy | 13-15 | -- | **Missing** |
| 2.2.5.1 | プロキシの妥当性検証 | Proxy Validation | 14 | -- | **Missing** |
| 2.2.5.2 | 1階差分による識別 | First-Difference Identification | 14-15 | -- | **Missing** |
| 2.2.6 | 閾値構造の形式的検証 | Formal Threshold Testing (Hansen PTR) | 15-16 | -- | **Missing** |
| 2.2.7 | 認知的密度の修飾効果 | Cognitive Density Moderation (Service Sector) | 16-17 | -- | **Missing** |
| 2.2.8 | 追加的頑健性検証 | Additional Robustness Checks | 17 | -- | **Missing** |
| 2.3 | Block C: 体験的処理とメンタルヘルス | Block C: Experiential Processing & Mental Health | 17-19 | -- | **Missing** |
| 2.3.1 | マクロレベルでの操作化の構造的限界 | Structural Limitations of Macro Operationalization | 17 | -- | **Missing** |
| 2.3.2 | NHANES 2017-2018 | NHANES 2017-2018 (N=5,032) | 17-18 | -- | **Missing** |
| 2.3.3 | ATUS Wellbeing Module | ATUS Wellbeing Module (N=21,736) | 18-19 | -- | **Missing** |
| 2.4 | Study 1 の結論と展望 | Study 1 Conclusions & Outlook | 19-20 | -- | **Missing** |
| 3.1 | 栄養代謝アナロジー | Nutritional Metabolism Analogy | 20-21 | 7 | **Complete** |
| 3.2.1 | 感情粒度と処理コスト | Emotional Granularity & Processing Cost | 20-21 | 7 | **Complete** |
| 3.2.2 | 情報入力は一方的に処理コストを増大 | Information Input Unilaterally Increases Cost | 21 | 7-8 | **Complete** |
| 3.2.3 | 教育は圧縮として機能する | Education Functions as Compression | 21 | 7-8 | **Complete** |
| 3.2.4 | 閾値モデル | Threshold Model | 21 | 8 | **Complete** |
| 3.2.5 | 数理的定式化 | Mathematical Formulation | 21-23 | 8-9 | **Partial** (英語版は簡略、日本語版はより詳細な理論的根拠あり) |
| 3.2.6 | パラメータの経験的推定 | Empirical Parameter Estimation | 22-23 | 9 | **Complete** |
| 3.3 | 感覚運動ループの閉合度 | Sensorimotor Loop Closure | 23-26 | 9-10 | **Partial** (英語版は基本構造のみ。日本語版は処理モード理論との接続、操作的定義の詳細な議論、境界事例を含む大幅に詳細) |
| 3.4 | 自己参照ループの一般病理学 | Self-Referential Loop as General Pathology | 25-26 | 10 | **Partial** (英語版は概要のみ。日本語版は表形式の体系的整理あり) |
| 3.5 | 生成AIの台頭 | Generative AI: The Ultimate Test Case | 26-28 | 10-11 | **Partial** (英語版は基本予測のみ。日本語版は3.5.1-3.5.3のサブセクション構造、AI精神病の臨床報告、Anthropic/OpenAIのデータ、Perlis et al. 2026等の詳細あり) |
| 4 | 操作化: 体験的処理の定量化 | Operationalization | 28-29 | 11-12 | **Complete** |
| 5 | 個人レベル検証デザイン | Proposed Individual-Level Confirmatory Design | 29 | 12 | **Complete** |
| 6 | 個人レベルの収束的証拠 | Individual-Level Convergent Evidence | 29-33 | 13-15 | **Partial** (基本構造は存在するが、日本語版は6.6「差分的予測のまとめ」テーブル等が大幅に詳細) |
| 6.6 | 差分的予測のまとめ | Differential Predictions Summary (Table 2) | 32-33 | -- | **Missing** |
| 7 | 反証条件 | Falsification Conditions | 33-34 | 15-16 | **Complete** |
| 7.1 | 立証実験デザイン | Minimum-Cost Verification Designs | 33-34 | 15-16 | **Complete** |
| 8.1 | 既存文献との位置づけ | Positioning Relative to Existing Literature | 34-36 | 16-17 | **Partial** (英語版は簡略。日本語版は行動活性化療法との関係、現代概念群の再解釈等が大幅に詳細) |
| 8.2 | 政策的含意 | Policy Implications | 35-37 | 17 | **Partial** (英語版は基本のみ。日本語版は倫理的留保、広告形態の質的転換、労働政策等が詳細) |
| 8.2.1 | AI設計への含意: 認知運動化の原理 | AI Design: Cognitive Motorization | 36-37 | 17 | **Partial** (英語版は概要のみ。日本語版はループ類型別変換表、検証デザイン、倫理的限界を含む) |
| 8.2.2 | 検証可能なデザイン | Testable AI Design | 36-37 | -- | **Missing** |
| 8.2.3 | 認知運動化の倫理的限界 | Ethical Limitations of Cognitive Motorization | 37 | -- | **Missing** |
| 8.3 | 観察問題 | The Observation Problem | 37 | 17-18 | **Complete** |
| 8.4 | 現時点の証拠の率直な評価 | Honest Assessment of Current Evidence | 37-39 | 18 | **Partial** (英語版は基本のみ。日本語版はBlock Cの個人レベルデータ要約を含む) |
| App A | Appendix A: 個人レベル分析の詳細仕様 | Appendix A: Individual-Level Analysis Specifications | 39-41 | -- | **Missing** |
| -- | 主要参考文献 | References | 40-41 | 18-19 | **Partial** (英語版は20本程度、日本語版は40本以上) |
| -- | データ利用可能性と再現資料 | Data Availability & Replication | -- | 19 | **Complete** |

### ステータス集計

| Status | 件数 |
|---|---|
| **Complete** | 12 |
| **Partial** | 10 |
| **Missing** | 22 |

---

## 2. Priority Order for Completion（完成優先順位）

英文プレプリントを学術誌投稿可能な状態にするための優先順位を以下に示す。

### Priority 1: Critical Missing Sections（最優先: 論文の骨格に不可欠）

| 順位 | Section | 理由 |
|---|---|---|
| 1 | **Abstract** (本文) | 査読者が最初に読む。日本語版に英語Abstractあり、クリーンアップで対応可能 |
| 2 | **Section 1: Introduction** | 論文の問題設定・動機・構造の説明がなければ読者は続きを理解できない |
| 3 | **Section 2 全体** (2.1.1-2.4) | Study 1のデータ分析は論文の実証的基盤。Section 3以降が参照する全ての統計結果がここにある |

### Priority 2: Major Content Gaps（高優先: 実質的な内容欠落）

| 順位 | Section | 理由 |
|---|---|---|
| 4 | **Section 3.5 拡充** (AI関連) | 日本語版は3.5.1-3.5.3のサブセクション構造、AI精神病の臨床報告、企業データ等の重要内容を含む |
| 5 | **Section 3.3 拡充** (ループ閉合度) | 操作的定義の詳細、処理モード理論との接続、境界事例の議論が不足 |
| 6 | **Section 6.6** (差分的予測のまとめ Table 2) | バランスモデルvs既存理論の差分を体系化する重要な表 |

### Priority 3: Enhanced Content（中優先: 議論の深化）

| 順位 | Section | 理由 |
|---|---|---|
| 7 | **Section 8.1 拡充** | BA療法との関係、現代概念群の再解釈 |
| 8 | **Section 8.2 拡充** | 倫理的留保、広告形態の質的転換、労働政策 |
| 9 | **Section 8.2.2-8.2.3** | AI認知運動化の検証デザインと倫理的限界 |
| 10 | **Section 8.4 拡充** | Block Cの個人レベルデータ要約の追加 |

### Priority 4: Supporting Materials（低優先: 補助的資料）

| 順位 | Section | 理由 |
|---|---|---|
| 11 | **Appendix A** | 分析仕様の詳細。再現性には重要だが論文本体の理解には必須ではない |
| 12 | **References 拡充** | 日本語版で追加された文献の反映 |
| 13 | **筆者の前提と本論文の位置づけ** | 独立研究者としての透明性宣言。投稿先によっては不要 |

---

## 3. Section-by-Section Guide（セクション別完成ガイド）

---

### 3.1 Abstract（概要）

**対応する日本語版**: Pages 2-3（英語Abstract部分: "This paper proposes..."から始まる段落）
**現状**: 英語版にはキーワードのみ存在。Abstract本文が完全に欠落。
**日本語版の英語Abstract**: Pages 2-3に完全な英語Abstractが既に存在する。クリーンアップと若干の編集のみで対応可能。

**含むべき主要内容**:
- 中核的主張: 精神病理の規定因はスクリーンタイム絶対量ではなくバランス
- 177ヶ国パネルデータ分析の主要知見 (1)-(4)
- 個人レベル検証 (NHANES N=5,032; ATUS N=21,736)
- 理論的枠組み（感覚運動ループ閉合度、認知的飢餓）
- 重大な限界（生態学的誤謬）
- 反証条件の明示

**参照される図表**: なし
**推定語数**: ~400語（日本語版の英語Abstract自体が約380語）
**翻訳上の注意点**:
- 日本語版のAbstractの英語部分をそのまま使用可能だが、OCR由来の文字化け修正が必要
- "involuntary cognitive input" は一貫して使用すること
- "sensorimotor loop closure" の表現を統一

---

### 3.2 Section 1: Introduction（導入）

**対応する日本語版**: Pages 3-4（「1. 導入」セクション全体）
**現状**: 英語版に完全に欠落

**含むべき主要内容**:
1. 先進国メンタルヘルスの逆説（物質的繁栄 vs 鬱病有病率の史上最高値）
2. 規制派（Haidt, Twenge）vs デジタルリテラシー派の10年間の論争と収束の失敗
3. 本論文の中核的主張: 両陣営が見ている変数が間違っている（絶対量 vs バランス）
4. 栄養学アナロジーの導入（肥満の原因は「食べること」ではなくバランス不均衡）
5. 「認知入力」の本質の説明: 単なる情報量ではなくSNSが引き起こす感情弁別の負荷
6. デジタルプラットフォームのアルゴリズムが感情を離散化する構造的非対称性
7. 「認知の肥満（Cognitive Obesity）」の命名と枠組みの3つの貢献 (a)(b)(c)

**参照される図表**: なし
**推定語数**: ~800-1,000語
**翻訳上の注意点**:
- 「6択のリアクションボタンに圧縮される」「バズる怒り方」等の具体例は英語読者にも通じるように適応
- "cognitive input" と "experiential processing" の初出定義をここで明確にする
- "additive imbalance (EB = Intake - Expenditure)" の数式表現を導入部で提示
- 日本語版の文体はやや随筆的。英語版は学術論文の文体に適応させる

---

### 3.3 Section 2: Macro Data Analysis (Study 1)（マクロデータ分析）

これは英語版で最も大きな欠落であり、ページ数にして2-19ページ分（日本語版の4-20ページに相当）が全て Missing である。

#### 3.3.1 Section 2 Introduction & Block Structure

**対応する日本語版**: Page 4（「2. マクロデータ分析（Study 1）」の導入段落）
**含むべき主要内容**:
- Block A/B/Cの3ブロック構造の説明
- 粗い第一近似から段階的に精緻化する探索的プロセスとしての構成説明
- R = I/E を出発点とし、TWFEでの崩壊、広告曝露プロキシへの更新という流れの予告

**推定語数**: ~300語

#### 3.3.2 Section 2.1.1 Data and Methods

**対応する日本語版**: Pages 4-5
**含むべき主要内容**:
- データセット: 177ヶ国 x 1990-2023年パネルデータ (3,899 country-year observations)
- データソース: IHME GBD (鬱病, 自殺), UNODC (殺人), World Bank WDI (インターネット, GDP), UNDP HDR (教育)
- 3段階の分析: Pooled OLS (Model 1-3)
- 相関分析の3レベル: (a) 横断的Pearson, (b) 国内時系列, (c) グローバル時系列
- 所得水準による層別化: 高所得51, 上位中所得44, 下位中所得46, 低所得31

**推定語数**: ~400語

#### 3.3.3 Section 2.1.2 Key Findings (Findings 1-4)

**対応する日本語版**: Pages 5-9

**知見1: 相関構造が発展段階で相転移する**
- 鬱病-殺人の国内時系列相関の所得別パターン（高所得 r=-0.323, 低所得 r=+0.007）
- 日本のケース（r=-0.865, 全177ヶ国中最強の負の相関）
- 解釈: 低所得国は外的原因、高所得国は処理容量が律速
- **Figure 1, Figure 2** への参照

**知見2: Internet/Education比が鬱病の予測因子**
- Model 1結果: Internet β=+0.009, Education β=-0.103, GDP β=+0.074 (n.s.)
- Model 2: Intake_ratio の偏相関 r=+0.300 (p<10^-12)
- **Figure 3** への参照

**知見3: 自殺と殺人は正の相関**
- Achenbach内在化/外在化モデルの予測に反する結果
- 自殺-殺人: 中央値 r=+0.293, グローバル r=+0.638
- Model 3: 殺人率 β=-0.006 (p<0.001)
- **Figure 4** への参照

**知見4: 固定効果モデルおよび交絡変数統制**
- Country FE: Internet β=+0.003 (p<10^-9), within R²=0.158
- 交絡変数統制: 都市化, Gini, 医療支出
- 所得層別FE: 高所得 R²=0.194, Chow検定有意
- Intake_ratio閾値前後の鬱病変化率（20倍加速, t(57)=9.10）
- 日本の位置づけ（66ヶ国中1位）
- **Figure 5** への参照

**推定語数**: ~2,000語
**参照される図表**: Figure 1-5
**翻訳上の注意点**:
- 統計値（β, t, p, R²）は原文のまま正確に転記
- "phase transition" / "相転移" の訳語を一貫して使用
- "within R²" の説明を英語読者向けに明確にする

#### 3.3.4 Section 2.1.3 Limitations

**対応する日本語版**: Pages 9-10
**含むべき主要内容**:
- Pooled OLS の低い R² (0.11-0.13)
- 生態学的誤謬（Simpson's paradox の可能性）
- 横断的依存性と動的パネルバイアス（Driscoll-Kraay, Bai IFE, CCE, Nickell bias）
- 自殺率の過少報告問題
- 因果方向の未確定
- Internet普及率の質的異質性（SNS, 教育, ゲーム等の合算）
- 教育年数の代理変数としての根本的限界

**推定語数**: ~600語

#### 3.3.5 Section 2.2.1 Causal Direction: Lagged Panel Analysis

**対応する日本語版**: Pages 10-11
**含むべき主要内容**:
- 順方向モデル: D(t) = a*D(t-1) + b*R(t-1) → b=0.006, t=5.44
- 逆方向モデル: R(t) = a*R(t-1) + b*D(t-1) → b=-0.299, t=-4.01（符号が負 = 逆因果棄却）
- 所得層別ラグ: 高所得のみ有意（t=5.41）
- 2年ラグ: t=13.09（累積的影響）
- 1階差分での再検証（非定常性への対処）
- 構成要素モデル vs バランスモデルの比較（AIC）
- **Figure 6** への参照

**推定語数**: ~500語

#### 3.3.6 Section 2.2.2 First Approximation Collapse: TWFE

**対応する日本語版**: Pages 11-12
**含むべき主要内容**:
- Table 5の再現: FEなし/Country FE/Year FE/TWFEでの係数変化
- TWFE: β=-0.060, t=-8.25（符号反転）
- GDP β=0.252, t=18.27（診断バイアスの示唆）
- **Figure 7** への参照

**推定語数**: ~400語

#### 3.3.7 Section 2.2.3 Hard Outcome Validation: Suicide Rate

**対応する日本語版**: Pages 12-13
**含むべき主要内容**:
- 自殺率TWFE: GDP t=-0.07（ゼロ）→ 鬱病の診断バイアス確認
- 自殺率と第一近似: TWFE β=-0.722, t=-7.54（保護的）
- 認知肥満仮説の単純予測との逆方向 → 二面性（恩恵 vs 負荷）の発見

**推定語数**: ~300語

#### 3.3.8 Section 2.2.4 Threshold Sweep Analysis

**対応する日本語版**: Page 13
**含むべき主要内容**:
- Internet普及率 30%-85% の閾値掃引
- 50-70%で係数の符号反転（保護→有害）
- 鬱病・自殺の両方で同パターン → 診断バイアスでは説明不可
- **Figure 8** への参照
- 先進国サンプルの限界

**推定語数**: ~400語

#### 3.3.9 Section 2.2.5 Ad Exposure Proxy

**対応する日本語版**: Pages 13-15
**含むべき主要内容**:
- Internet普及率の限界（「蛇口の有無」vs「水の量」）
- DataReportalのスクリーンタイムデータ（量 vs 質の乖離）
- Ad Proxy = (Internet/100) x GDP/capita の構成根拠
- 閾値掃引結果: 相転移は$1,000前後
- 2.2.5.1: プロキシの外的妥当性（Spearman ρ=0.83）
- 2.2.5.2: 1階差分による識別（Δproxy t=3.19, ΔGDP t=0.16 n.s.）
- **Figure 9, 10** への参照

**推定語数**: ~800語
**翻訳上の注意点**:
- "non-selective cognitive input" の定義と具体例を明確に
- "advertising ecosystem expansion" の概念を丁寧に導入

#### 3.3.10 Section 2.2.6 Formal Threshold Testing (Hansen PTR)

**対応する日本語版**: Pages 15-16
**含むべき主要内容**:
- Hansen PTR結果: 鬱病は非有意（p=0.654）、自殺は有意（γ*=25.7, F=213）
- 鬱病=連続的dose-response反転（二次項 F=80.04）
- 反転点: 中央値9,169, 95%CI [7,686, 9,965] → 先進国上位
- 臨床的知見との整合（鬱病=グラデーション、自殺=閾値）
- **Figure 12** への参照

**推定語数**: ~500語

#### 3.3.11 Section 2.2.7 Cognitive Density Moderation

**対応する日本語版**: Pages 16-17
**含むべき主要内容**:
- サービス業雇用比率による層別化
- 交互作用 F=48.97
- サービス業比率が高いほど同一曝露での感受性が高い
- **Figure 13** への参照

**推定語数**: ~300語

#### 3.3.12 Section 2.2.8 Additional Robustness Checks

**対応する日本語版**: Page 17
**推定語数**: ~200語

#### 3.3.13 Section 2.3 Block C (Individual-Level)

**対応する日本語版**: Pages 17-19
**含むべき主要内容**:
- 2.3.1: マクロ体験的処理操作化の失敗（WHO身体不活発指標のTWFE崩壊）
- 2.3.2: NHANES (N=5,032): PHQ-9 vs 余暇運動（d=0.36, フルモデル β=-0.71）
- 2.3.3: ATUS (N=21,736): 2x2対偶検証、3x3用量反応、加法モデルの支持
- Block C小括: 2成分構造の確認、加法モデル、Fair/Poor健康率3.1倍

**推定語数**: ~1,200語

#### 3.3.14 Section 2.4 Study 1 Conclusions

**対応する日本語版**: Pages 19-20
**含むべき主要内容**:
- 7つの帰結の段階的要約
- 示されなかった事項の率直な列挙
- 個人レベルデータの必要性の確立

**推定語数**: ~600語

---

### 3.4 Section 3.3 Sensorimotor Loop Closure（拡充）

**対応する日本語版**: Pages 23-26
**現状**: 英語版は基本的な3分類（閉/開/自己参照）とゲームの例を含むが、以下が欠落
**追加すべき内容**:
- 処理モード理論（Watkins & Teasdale 2001, 2004）との明示的接続
- 体験的処理の操作的定義（第1項/第2項の分類原理 (a)-(e)）
- 境界事例の議論（ポッドキャスト、臨場体験、創造的アウトプット）
- 「認知的 vs 身体的」ではなく「ループが環境と閉じているか」が基準であることの強調
- Boers et al. (2019) の実証的アンカー

**推定語数**: 追加 ~800語

---

### 3.5 Section 3.5 Generative AI（拡充）

**対応する日本語版**: Pages 26-28
**現状**: 英語版は予測AとTherabotデータのみ
**追加すべき内容**:
- 3.5.1: 入力側の増大（SNS→AIの認知入力密度の質的飛躍）
- 3.5.2: AI追従性（sycophancy）の構造的分析
- 実証データ: Perlis et al. (2026, JAMA, N=20,847), McBain et al. (2025)
- AI精神病の臨床報告: Pierre et al. (2026), Chandra et al. (2026), Human Line Project
- AI企業自身のデータ: Anthropic (2026, arXiv, 150万会話), OpenAI+MIT (Phang et al. 2025, 400万会話+RCT)
- 予測B（ループ構造依存性）、予測C（身体活動による緩衝）の追加

**推定語数**: 追加 ~1,000語
**翻訳上の注意点**:
- "sycophancy" / "追従性" は AI alignment 分野の専門用語として定訳あり
- "disempowerment" は Anthropic の用語をそのまま使用
- Anthropic/OpenAI のデータは2025-2026年の最新知見であり、査読者への impact が大きい

---

### 3.6 Section 6.6 Differential Predictions Summary（新規）

**対応する日本語版**: Pages 32-33
**現状**: 英語版に完全に欠落
**含むべき内容**:
- Table 2: バランスモデル vs 既存理論（ART/SRT）の予測比較表
- 4つの差分的予測とその検証条件
- 予測1: 自然 x スマホ交互作用
- 予測2: 屋内運動 vs 自然散歩
- 予測3: 物質購買 x アレキシサイミア交互作用
- 予測4: Hedonic adaptation速度

**推定語数**: ~600語

---

### 3.7 Section 8.1 Discussion（拡充）

**対応する日本語版**: Pages 34-36
**追加すべき内容**:
- 行動活性化療法（BA）との詳細な関係（反芻→直接体験の転換 = 自己参照ループ→閉ループの転換）
- BAの活動選択へのループ閉合度の追加（mastery/pleasure + loop closure）
- 現代概念群のバランスモデルによる再解釈表（ワークライフバランス、マインドフルネス、デジタルデトックス、flow、attention economy、burnout）

**推定語数**: 追加 ~600語

---

### 3.8 Section 8.2 Policy Implications（拡充）

**対応する日本語版**: Pages 35-37
**追加すべき内容**:
- 倫理的留保（個人帰責モデルの否定、環境設計への焦点、権威主義的転用リスクへの警告）
- 広告形態の質的転換の詳細（スポンサーシップ→プログラマティック、Monsellのタスク切替コスト）
- サービス業構成の修飾効果が示す政策課題（認知的保護の不均等性、ギグワーカー問題）

**推定語数**: 追加 ~500語

---

### 3.9 Sections 8.2.2-8.2.3（新規）

**対応する日本語版**: Pages 36-37
**Section 8.2.2 含むべき内容**:
- AI認知運動化の効果測定デザイン（14日間ESM、前半通常/後半介入）
- ループ類型別変換表
- n=50の被験者内比較で中効果量検出可能

**Section 8.2.3 含むべき内容**:
- 重度うつ病患者へのリスク（アンヘドニアと行動促しの逆効果）
- AI によるループ検知精度の限界
- 商業的AIにおける sycophancy へのデフォルト回帰インセンティブ
- 安全策 (a)-(d)
- 「予防的環境設計」としての位置づけ

**推定語数**: ~700語

---

### 3.10 Appendix A: Individual-Level Analysis Specifications

**対応する日本語版**: Pages 39-41
**含むべき内容**:
- A.1 NHANES 2017-2018: データソース、包含基準、変数定義、モデル仕様、サーベイウェイトの注記
- A.2 ATUS Wellbeing Module: データソース、活動コード分類の詳細、分析手法、共変量の意図的未投入の根拠
- A.3 共通の限界と今後の改善

**推定語数**: ~1,500語
**翻訳上の注意点**:
- 活動コード（T120303等）は原文のまま使用
- 変数名（RIDAGEYR, DMDEDUC2等）は原文のまま使用
- サーベイウェイト未適用の限界を明示的に記述

---

### 3.11 References（参考文献の拡充）

**現状**: 英語版は約20本。日本語版は約40本以上。
**追加すべき主要文献**:
- Boers, E., et al. (2019). JAMA Pediatrics.
- Chandra, K., et al. (2026). arXiv:2602.19141.
- McBain, R. K., et al. (2025). JAMA Network Open.
- Perlis, R. H., et al. (2026). JAMA Network Open.
- Pierre, J. M., et al. (2026). PMC12863933.
- Phang, J., et al. (2025). arXiv:2504.03888.
- Anthropic (2026). arXiv:2601.19062.
- Hamilton, Farmer, et al. (2015). Biological Psychiatry.
- Bluhm, et al. (2009). PNAS.
- Bettmann, et al. (2025). Ecopsychology.
- Fernandez-Montero, et al. (2024).
- Noetel, et al. (2024). BMJ.
- Yeo, et al. (2023). Psychological Medicine.
- Tak, et al. (2017). Journal of Youth and Adolescence.
- Sheridan, et al. (2022). Annals of Work Exposures and Health.
- Foley, P. (2025). Human-Centered Change and Innovation.
- Arend & Schafer (2019). Psychological Methods.
- Gilovich, Kumar & Jampol (2015); Gilovich & Kumar (2020).
- Ohly, et al. (2016). [ART systematic review]

---

## 4. Abstract Draft（英文Abstract草稿）

以下は日本語版 Pages 2-3 の英語Abstract部分をクリーンアップしたものである。

> **Abstract**
>
> This paper proposes an integrative framework for the relationship between digital media and mental health, grounded in a nutritional metabolism analogy. The core claim is that the determinant of psychopathology is not the absolute amount of screen time but the balance between involuntary cognitive input and experiential processing capacity, classified by sensorimotor loop closure---the degree of bidirectional causal coupling with the environment. Systematic analysis of 177-country panel data (1990--2023) reveals: (1) the positive association between internet penetration and depression observed under country fixed effects collapses under Two-Way Fixed Effects (TWFE), indicating dependence on common time trends; (2) GDP is the dominant predictor of depression prevalence (t = 18.27), reflecting diagnostic infrastructure bias confirmed through hard-outcome validation (suicide rate: GDP effect t = -0.07); (3) a continuous dose-response reversal: Hansen Panel Threshold Regression identifies a significant threshold for suicide (gamma\* = 25.7, F = 213, protective saturation) but not for depression (p = 0.654), while quadratic modeling reveals a significant inverted-U relationship (F = 80.04) with reversal point at proxy approximately 10,000, indicating most countries remain in the "benefit > harm" zone; (4) an ad exposure proxy (Internet Penetration x GDP/capita) captures this benefit-to-harm reversal more cleanly (threshold approximately $1,000) and survives first-difference analysis controlling for GDP change (Delta-proxy: t = 3.19; Delta-GDP: t = 0.16, n.s.), demonstrating that the proxy captures information beyond GDP---namely, the expansion of advertising ecosystems accompanying internet diffusion. Individual-level validation (NHANES N = 5,032; ATUS N = 21,736) confirms that physical exercise and active cognitive leisure independently predict wellbeing as additive channels (interaction p = 0.95), with the absence of both producing the worst health outcomes (3.1x Fair/Poor health rate). These results suggest that the mental health impact of information technology depends not on quantity but on quality---specifically, the density of involuntary cognitive input (targeted advertising, algorithmic recommendations, push notifications). Existing findings from behavioral economics, environmental psychology, and neuroscience are reinterpreted through the balance model to provide individual-level convergent evidence. The framework derives a quality-based classification of screen time via sensorimotor loop closure, the prescriptive concept of "cognitive starvation," and policy principles including qualitative transformation of advertising formats. A critical limitation is that all macro-level findings are subject to ecological fallacy: country-level associations cannot be directly attributed to individual-level mechanisms. This study is exploratory and requires individual-level confirmatory validation through experience sampling methodology. Falsification criteria are specified.
>
> **Keywords:** cognitive obesity, digital media, mental health, balance model, sensorimotor loop closure, advertising exposure, experiential processing

---

## 5. Formatting Checklist（フォーマットチェックリスト）

### 5.1 Target Journal Formatting（投稿先候補と要件）

| Journal | Format | Word Limit | Abstract | Figures | Fit |
|---|---|---|---|---|---|
| **PNAS** | Research Article | 6 pages (~4,500 words) | 250 words | 6 max | Study 1 + Framework に絞る必要あり。現状は長すぎる |
| **Nature Human Behaviour** | Article | 3,000 words main + Methods | 150 words | 8 max | メインテキスト大幅圧縮が必要。Methods に分析詳細を移動 |
| **Psychological Bulletin** | Review/Theoretical | 制限なし | 250 words | 制限なし | 理論統合論文として最も適合。長さの制約なし |
| **Perspectives on Psychological Science** | Target Article | ~10,000 words | 150 words | 制限なし | 理論的枠組み提案に適合。Commentary を招くフォーマット |
| **JAMA Psychiatry** | Original Investigation | 3,500 words | 350 words (structured) | 5 max | Study 1 の疫学データに適合。理論部分は Supplement |
| **PLoS ONE** | Research Article | 制限なし | 300 words | 制限なし | 最も柔軟。探索的研究のプレプリントとして最適 |

**推奨**: 現在の論文の構成（理論枠組み + マクロデータ + 収束的証拠 + 反証条件）を考慮すると、**Perspectives on Psychological Science**（Target Article）または **Psychological Bulletin** が最も自然な投稿先である。PNAS/Nature HB への投稿には大幅な圧縮と再構成が必要。

### 5.2 Figure/Table Numbering Consistency（図表番号の整合性）

日本語版の図表と英語版での対応:

| JP Figure | 内容 | EN 対応 | Status |
|---|---|---|---|
| Figure 1 | 鬱病 vs 殺人: グローバル分析 | 未配置 | Section 2.1.2 に必要 |
| Figure 2 | 所得水準別内在化-外在化相関 | 未配置 | Section 2.1.2 に必要 |
| Figure 3 | 認知肥満モデル国際横断的エビデンス | 未配置 | Section 2.1.2 に必要 |
| Figure 4 | 内在化-外在化対称性分析 | 未配置 | Section 2.1.2 に必要 |
| Figure 5 | 頑健性分析 | 未配置 | Section 2.1.2 に必要 |
| Figure 6 | Granger型因果方向テスト | 未配置 | Section 2.2.1 に必要 |
| Figure 7 | 第一近似の崩壊 | 未配置 | Section 2.2.2 に必要 |
| Figure 8 | 閾値掃引分析 | 未配置 | Section 2.2.4 に必要 |
| Figure 9 | 広告曝露プロキシ閾値掃引 | 未配置 | Section 2.2.5 に必要 |
| Figure 10 | プロキシ妥当性検証 | 未配置 | Section 2.2.5.1 に必要 |
| Figure 12 | Dose-Response反転分析 | 未配置 | Section 2.2.6 に必要 |
| Figure 13 | サービス業構成によるdose-response修飾 | 未配置 | Section 2.2.7 に必要 |

**注意**: Figure 11 は日本語版に言及がない。番号の欠落を確認すること。

### 5.3 Cross-Reference Checks（相互参照チェック）

英語版で他のセクションへの参照が存在するが、参照先が欠落しているケース:

- Page 7: "Section 2.3" → Section 2.3 は英語版に存在しない
- Page 7: "empirically demonstrated in Section 2.3" → 同上
- Page 8: "Study 1" → Section 2 全体が欠落
- Page 9: "Study 1's benefit-to-harm reversal" → Section 2.2.4-2.2.5 が欠落
- Page 13: "Section 2" → 欠落
- Page 13: "Section 2.3.3" → 欠落
- Page 16: "Section 3.5" → 英語版では簡略すぎて参照内容が不十分

### 5.4 Terminology Consistency（用語の一貫性チェックリスト）

| 日本語 | English (使用すべき訳語) | 注意 |
|---|---|---|
| 認知の肥満 | Cognitive Obesity | タイトルと一致 |
| 認知入力 | Cognitive Input | "information input" と混用しない |
| 体験的処理 | Experiential Processing | "experiential" を一貫使用 |
| 非選択的認知入力 | Involuntary Cognitive Input | "non-selective" も可だが統一 |
| 感覚運動ループ閉合度 | Sensorimotor Loop Closure | "closure degree" も可だが統一 |
| 認知的飢餓 | Cognitive Starvation | 処方的概念として定義 |
| 認知運動化 | Cognitive Motorization | ドイツ語 kognitiv-motorisierung も併記 |
| 自己参照ループ | Self-Referential Loop | "self-referencing" と混用しない |
| 相転移 | Phase Transition | 物理学的比喩として使用 |
| 第一近似 | First Approximation | R = I/E を指す |
| 広告曝露プロキシ | Ad Exposure Proxy / AdProxy | 変数名として一貫使用 |
| 処理容量 | Processing Capacity | "capacity" を一貫使用 |
| 加法的不均衡 | Additive Imbalance | 比率ではなく差分であることを強調 |
| 閾値 | Threshold | 文脈に応じて R*, gamma* 等 |
| 反証条件 | Falsification Conditions | Popper的意味で使用 |

---

## 6. Total Estimated Work（総作業量の推定）

### 語数推定

| Category | Sections | Estimated Words |
|---|---|---|
| **新規翻訳 (Missing)** | Abstract, Sec 1, Sec 2全体 (2.1-2.4), Sec 6.6, Sec 8.2.2-8.2.3, Appendix A | ~10,400 |
| **拡充翻訳 (Partial)** | Sec 3.3, 3.5, 8.1, 8.2, 8.4 | ~3,500 |
| **参考文献追加** | ~20本 | ~600 |
| **図表キャプション** | Figure 1-13 | ~800 |
| **合計** | | **~15,300語** |

### 作業時間推定

| タスク | 推定時間 |
|---|---|
| Section 2 全体の翻訳・執筆 | 20-30時間 |
| Section 1 (Introduction) の翻訳・適応 | 3-5時間 |
| Abstract のクリーンアップ | 1-2時間 |
| Section 3.3, 3.5 の拡充 | 5-8時間 |
| Section 6.6, 8.1, 8.2 の拡充・翻訳 | 5-8時間 |
| Section 8.2.2-8.2.3 の翻訳 | 2-3時間 |
| Appendix A の翻訳 | 4-6時間 |
| 参考文献の整理・追加 | 2-3時間 |
| 図表キャプションの英訳 | 2-3時間 |
| 相互参照・用語統一チェック | 3-5時間 |
| **合計** | **47-73時間** |

### 現在の英語版の語数と完成後の推定

- 現在の英語版: ~5,800語（Pages 7-19）
- 追加予定: ~15,300語
- 完成後の推定総語数: **~21,100語**（Appendix含む）
- 本文のみ: **~18,000語**

この規模は Psychological Bulletin や Perspectives on Psychological Science に適合する。PNAS (4,500語) や Nature Human Behaviour (3,000語 + Methods) への投稿には、本文の大幅な圧縮（~4,000語）と Methods/Supplementary Information への分離が必要となる。

---

## 補足: 英語版ページ構造の再確認

英語版PDFのページ構成（現状）:
- **Page 1**: タイトル、著者情報、Abstract（キーワードのみ）
- **Pages 2-6**: **画像のみ**（テキストなし。おそらくFigure 1-5等の図表が配置されているが、テキスト抽出されていない）
- **Page 7**: Section 3.1 から開始（Section 1, 2 が完全に欠落）
- **Pages 7-18**: Section 3-8 および References
- **Page 19**: Data Availability

Pages 2-6 が画像のみであることは、元のPDFにSection 2の図表のみが配置され、本文テキストが挿入されていない状態を示す。これらのページにSection 2の本文テキストを挿入する必要がある。
