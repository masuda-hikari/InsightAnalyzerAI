# InsightAnalyzerAI

AI駆動のデータ分析アシスタント
- 自然言語でデータに質問し、即座にインサイトを得る

## 概要
InsightAnalyzerAIは、データセットをアップロードして自然言語で質問するだけで、AIが適切な分析を実行し結果を返すツールです。データアナリストがいなくても、誰でもデータから価値ある情報を引き出せます。

## 機能

- **自然言語クエリ**: 「先月の地域別売上合計は？」のような質問に回答
- **自動分析**: データのパターンや異常値を自動検出
- **可視化**: 適切なチャートを自動生成（Plotlyインタラクティブ対応）
- **レポート生成**: 分析結果をわかりやすいレポートに
- **認証・課金**: ユーザー管理とStripe統合

## デモ

- **Web UI**: [https://insightanalyzerai.streamlit.app](https://insightanalyzerai.streamlit.app)
- **ランディングページ**: [https://your-username.github.io/InsightAnalyzerAI/](https://your-username.github.io/InsightAnalyzerAI/)

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

## インストール

```bash
pip install -r requirements.txt
```

## クイックスタート

### Web UI（推奨）
```bash
streamlit run src/streamlit_app.py
```

### CLI モード
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

## 現在のステータス

**Phase 6 - デプロイ準備完了**

### Phase 1 (MVP) ✅完了
- [x] プロジェクト構造
- [x] CSV読み込み・基本統計（DataLoader）
- [x] CLIインターフェース
- [x] 単純なクエリ処理（QueryParser + QueryExecutor）
- [x] モジュール分離
- [x] 実行時間計測・信頼度スコア
- [x] 基本チャート生成（matplotlib）

### Phase 2 (LLM統合) ✅完了
- [x] OpenAI API統合（LLMHandler）
- [x] 自然言語→Pandasコード変換
- [x] 結果の自然言語説明生成
- [x] フォールバック機構（LLM失敗時）

### Phase 3 (可視化強化) ✅完了
- [x] Matplotlibによるチャート生成
- [x] チャートタイプ自動選択

### Phase 4 (Web UI) ✅完了
- [x] Streamlit Web UI
- [x] ファイルアップロード（CSV/Excel/Parquet）
- [x] Plotlyインタラクティブチャート
- [x] クエリ履歴機能

### Phase 5 (認証・課金) ✅完了
- [x] 認証システム（ユーザー登録/ログイン）
- [x] プラン管理（Free/Basic/Pro/Enterprise）
- [x] 使用量制限（ファイルサイズ、クエリ数）
- [x] Stripe課金統合

### Phase 6 (デプロイ) 🚧進行中
- [x] ランディングページ作成
- [x] GitHub Pages設定
- [ ] Streamlit Cloudデプロイ
- [ ] Stripeアカウント設定
- [ ] ベータテスター募集

## デプロイ手順

### GitHub Pages（ランディングページ）
1. リポジトリ設定 → Pages → Source: "main" → Folder: "/docs"
2. 保存後、数分で公開される

### Streamlit Cloud（アプリ本体）
1. [share.streamlit.io](https://share.streamlit.io) でアカウント作成
2. GitHubリポジトリを連携
3. Main file: `src/streamlit_app.py`
4. Secrets設定:
   - `OPENAI_API_KEY`
   - `STRIPE_SECRET_KEY`
   - `STRIPE_WEBHOOK_SECRET`

## ライセンス

MIT License

## 貢献

現在は非公開開発中です。
---

**InsightAnalyzerAI** - データの力を、すべての人に。
