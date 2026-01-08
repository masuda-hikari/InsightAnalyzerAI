# InsightAnalyzerAI

AI駆動のデータ分析アシスタント
- 自然言語でデータに質問し、即座にインサイトを得る

## 概要
InsightAnalyzerAIは、データセットをアップロードして自然言語で質問するだけで、AIが適切な分析を実行し結果を返すツールです。データアナリストがいなくても、誰でもデータから価値ある情報を引き出せます。

## 機能

- **自然言語クエリ**: 「先月の地域別売上合計は？」のような質問に回答
- **自動分析**: データのパターンや異常値を自動検出
- **可視化**: 適切なチャートを自動生成
- **レポート生成**: 分析結果をわかりやすいレポートに

## 使用例
```python
from insight_analyzer import InsightAnalyzer

# データ読み込み
analyzer = InsightAnalyzer("sales_data.csv")

# 自然言語で質問
result = analyzer.ask("先月期の地域別売上の内訳を教えて")
print(result.answer)
# => "先月期の地域別売上: 東京 ¥12.5M, 大阪 ¥8.2M, 名古屋 ¥5.1M..."

# チャート生成
result.show_chart()
```

## ユースケース

- **中小企業経営者**: 売上データから傾向を把握
- **マーケティング担当**: キャンペーン効果の分析
- **研究者**: 実験データの探索的分析
- **スタートアップ**: リソースなしでデータドリブン意思決定

## 価値提案
**月額¥2,980で専門アナリスト相当の分析力**

従来、データ分析には専門知識を持つアナリストが必要でした。InsightAnalyzerAIを使えば：
- SQL/Pythonの知識不要
- 数秒で回答を取得
- 24時間いつでも利用可能
- 人件費の大幅削減（アナリスト雇用 vs ツール利用）

## インストール

```bash
pip install -r requirements.txt
```

## クイックスタート
```bash
# CLIモード
python -m src.insight_analyzer data/sample_sales.csv

# LLM統合モード（推奨）
OPENAI_API_KEY=sk-... python -m src.insight_analyzer data/sample_sales.csv --explain

# 対話モード
> 総売上を計算して
結果: ¥45,800,000
[LLM使用 | クエリ: result = df['total_sales'].sum() | 実行時間: 150ms]

> 月別のトレンドをグラフで見せて
[チャートを生成しました: output/monthly_trend.png]
```

### CLIオプション
- `--chart`: チャートも生成する
- `--no-llm`: LLM統合を無効化（フォールバックモード）
- `--explain`: LLMで結果を日本語で説明

## 料金プラン（SaaS版）
| プラン | 月額 | 機能 |
|--------|------|------|
| Free | ¥0 | 1MB制限, 10クエリ/日 |
| Basic | ¥2,980 | 50MB, 100クエリ/日, チャート |
| Pro | ¥9,800 | 500MB, 無制限, API連携 |
| Enterprise | 要見積 | オンプレミス, カスタム |

## 現在のステータス

**開発中 - Phase 2 (LLM統合)**

### Phase 1 (MVP) ✅完了
- [x] プロジェクト構造
- [x] CSV読み込み・基本統計（DataLoader）
- [x] CLIインターフェース
- [x] 単純なクエリ処理（QueryParser + QueryExecutor）
- [x] モジュール分離
- [x] 実行時間計測・信頼度スコア
- [x] 基本チャート生成（matplotlib）

### Phase 2 (LLM統合) 🚧進行中
- [x] OpenAI API統合（LLMHandler）
- [x] 自然言語→Pandasコード変換
- [x] 結果の自然言語説明生成
- [x] フォールバック機構（LLM失敗時）
- [ ] プロンプト最適化

### Phase 3以降
- [ ] Web UI（Streamlit）
- [ ] 自動インサイト
- [ ] SaaS化

## ライセンス

商用ライセンス - 詳細はお問い合わせください

## 貢献

現在は非公開開発中です。
---

**InsightAnalyzerAI** - データの力を、すべての人に。
