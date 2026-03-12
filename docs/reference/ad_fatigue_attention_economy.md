# Ad Fatigue / Attention Economy 文献検証メモ

作成日: 2025-03-12
背景: Grokとの議論で提案されたAd Fatigue関連文献の信頼性を検証

---

## 1. 検証結果サマリ

| 文献 | ステータス | 備考 |
|------|-----------|------|
| Firth et al. 2019 | **検証済** | World Psychiatry, IF~60+。ただしテーマは「インターネットと認知」であり、SNS×メンタルヘルスの疫学研究ではない |
| Niederdeppe et al. 2021 | **検証不可** | Grokの幻覚の可能性大。message fatigueの主要論文は So, Kim & Cohen (2017) |
| IJRPR Ad Fatigue論文 | **存在するが引用不可** | predatorylist.com 掲載の捕食的ジャーナル |
| Herbert Simon 1971 | **検証済** | 注意経済の原典。確立された引用 |
| Tim Wu 2016 | **検証済** | Knopf出版、正規の書籍 |
| Shoshana Zuboff 2019 | **検証済** | PublicAffairs出版、Harvard教授 |

---

## 2. 検証済み引用（推奨セット）

### 2.1 注意経済の原典

**Simon, H. A. (1971).** Designing organizations for an information-rich world.
In M. Greenberger (Ed.), *Computers, Communications, and the Public Interest* (pp. 37-72).
Johns Hopkins Press.
- 注意経済概念の起源。情報過多が注意の希少性を生むという議論
- PDF: https://gwern.net/doc/design/1971-simon.pdf

### 2.2 インターネット×認知変化

**Firth, J., Torous, J., Stubbs, B., et al. (2019).** The "online brain": how the Internet may be changing our cognition.
*World Psychiatry*, 18(2), 119-129.
DOI: [10.1002/wps.20617](https://doi.org/10.1002/wps.20617)
PMC: [PMC6502424](https://pmc.ncbi.nlm.nih.gov/articles/PMC6502424/)
- 被引用 ~433回
- 注意力、記憶、社会認知へのインターネットの影響をレビュー
- 認知肥満フレームワークとの接続: 情報過負荷 I の認知的影響を実証

### 2.3 メッセージ疲労

**So, J., Kim, S., & Cohen, H. (2017).** Message fatigue: Conceptual definition, operationalization, and correlates.
*Communication Monographs*, 84(1), 5-29.
DOI: [10.1080/03637751.2016.1250429](https://doi.org/10.1080/03637751.2016.1250429)
- メッセージ疲労の概念定義・操作化の基盤論文
- Grokが「Niederdeppe et al. 2021」として紹介したトピックの実際の主要論文

### 2.4 SNS疲労×メンタルヘルス

**Dhir, A., Yossatorn, Y., Kaur, P., & Chen, S. (2018).** Online social media fatigue and psychological wellbeing — A study of compulsive use, fear of missing out, fatigue, anxiety and depression.
*International Journal of Information Management*, 40, 141-152.
DOI: [10.1016/j.ijinfomgt.2018.01.012](https://doi.org/10.1016/j.ijinfomgt.2018.01.012)
- SNS疲労 → 不安・抑うつの直接的リンクを実証
- FoMO（取り残され不安）→ 強迫的使用 → 疲労のパスモデル

### 2.5 SNS疲労ドライバーのシステマティックレビュー

**Zheng, H., & Ling, M. (2021).** Drivers of social media fatigue: A systematic review.
*Telematics and Informatics*, 64, 101696.
DOI: [10.1016/j.tele.2021.101696](https://doi.org/10.1016/j.tele.2021.101696)
- 情報過負荷、プライバシー懸念、社会的比較がSNS疲労の主要ドライバー

### 2.6 注意経済の書籍

**Wu, T. (2016).** *The Attention Merchants: The Epic Scramble to Get Inside Our Heads.*
Alfred A. Knopf (Penguin Random House).
ISBN: 978-0-385-35201-7
- 19世紀新聞から現代SNSまでの注意収穫の歴史
- 著者はColumbia Law School教授、「ネット中立性」命名者

**Zuboff, S. (2019).** *The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power.*
PublicAffairs (Hachette).
ISBN: 978-1-61039-569-4
- 行動データ抽出→予測商品化のフレームワーク
- 著者はHarvard Business School名誉教授
- プラットフォーム抽出フェーズとの直接的関連性が高い

---

## 3. 引用不可・要注意文献

### 3.1 IJRPR Ad Fatigue論文（2025）

IJRPR = International Journal of Research Publication and Reviews
- **predatorylist.com に掲載** されている捕食的ジャーナル
- 偽のインパクトファクター（5.536を主張するがJCR非掲載）
- 最大75%の剽窃が検出された論文あり
- AI生成コンテンツの混入
- 主要学術データベースに非索引
- **結論: 絶対に引用しない**

### 3.2 Niederdeppe et al. 2021（Grok提案）

- Jeff NiederdeppeはCornell大の著名な健康コミュニケーション研究者
- しかし2021年の情報過負荷/メッセージ疲労に関する論文は **確認できず**
- Grokの幻覚（hallucination）の可能性が高い
- 同トピックの正確な引用は So, Kim & Cohen (2017) を使用すべき

---

## 4. 認知肥満フレームワークとの接続

これらの文献は論文の以下の議論を補強する：

1. **注意経済** (Simon 1971, Wu 2016): プラットフォームが注意を収穫する構造的インセンティブ → 情報露出 I の膨張メカニズム
2. **SNS疲労** (Dhir et al. 2018, Zheng & Ling 2021): 過剰な情報消費 → 認知疲労 → メンタルヘルス悪化のパスウェイ → L = α₁·I − α₂·C の実証的裏付け
3. **監視資本主義** (Zuboff 2019): extraction phase における行動データ商品化 → ユーザー福利の低下 → Study 1 (ARPU×態度乖離) の理論的基盤
4. **メッセージ疲労** (So et al. 2017): 広告・通知の過剰 → 認知的制御 C の低下メカニズム

---

## 5. Grok会話から得られた教訓

1. **引用の信頼性**: LLMが生成する引用は必ず検証が必要。今回もNiederdeppe et al. 2021は確認不可
2. **ジャーナル品質**: Grokが提案したIJRPRは捕食的ジャーナル。LLMはジャーナルの質を評価できない
3. **概念的示唆の価値**: Ad Fatigue × 認知肥満の接続という概念自体は有効。ただし信頼できるソースで裏付ける必要あり
4. **推奨ワークフロー**: Grok/ChatGPT等の提案 → PubMed/Semantic Scholar/Google Scholarで検証 → 引用決定
