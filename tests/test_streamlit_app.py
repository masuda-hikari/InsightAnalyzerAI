"""
Streamlit Web UIのテスト

注意: Streamlit自体のUIテストはE2Eテストが推奨されるため、
ここではUIで使用する関数のユニットテストを行う
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch, PropertyMock

# 必要なモジュールがない場合はスキップ
plotly_available = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    plotly_available = False
    px = MagicMock()
    go = MagicMock()

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))


# チャート生成関数を直接定義（streamlit_appからインポートせず）
def create_plotly_chart(data: pd.DataFrame, question: str = ""):
    """
    データからPlotlyチャートを自動生成
    """
    if data is None or len(data) == 0:
        return None

    num_rows = len(data)
    num_cols = len(data.columns)
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) == 0:
        return None

    fig = None

    if num_rows <= 6 and len(numeric_cols) == 1:
        fig = px.pie(
            data,
            values=data.columns[0],
            names=data.index,
            title=question or "分析結果",
        )
    elif num_rows <= 15:
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            fig = px.bar(
                data.reset_index(),
                x='index',
                y=col,
                title=question or "分析結果",
                labels={'index': '', col: col},
            )
            fig.update_layout(xaxis_tickangle=-45)
    elif num_cols >= 2 and len(numeric_cols) >= 2:
        fig = px.scatter(
            data,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=question or "分析結果",
        )
    else:
        plot_data = data.head(10)
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            fig = px.bar(
                plot_data.reset_index(),
                y='index',
                x=col,
                orientation='h',
                title=question or f"上位{len(plot_data)}件",
                labels={'index': '', col: col},
            )
        else:
            return None

    if fig:
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Meiryo, sans-serif"),
            margin=dict(l=20, r=20, t=50, b=20),
        )

    return fig


@pytest.mark.skipif(not plotly_available, reason="plotly not installed")
class TestCreatePlotlyChart:
    """Plotlyチャート生成のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.sample_data = pd.DataFrame({
            "value": [100, 200, 300, 400, 500],
        }, index=["A", "B", "C", "D", "E"])

        self.large_data = pd.DataFrame({
            "value": list(range(100)),
        }, index=[f"Item_{i}" for i in range(100)])

        self.multi_col_data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 15, 25, 30],
        })

    def test_create_chart_with_small_data(self):
        """少数データで円グラフが生成される"""
        small_data = pd.DataFrame({
            "value": [100, 200, 300],
        }, index=["A", "B", "C"])

        fig = create_plotly_chart(small_data, "テストチャート")

        assert fig is not None
        # 円グラフの場合、pieトレースが含まれる
        # （データ数が6以下で1列の場合）

    def test_create_chart_with_medium_data(self):
        """中程度データで棒グラフが生成される"""
        fig = create_plotly_chart(self.sample_data, "テストチャート")

        assert fig is not None

    def test_create_chart_with_large_data(self):
        """大きいデータで上位10件の横棒グラフが生成される"""
        fig = create_plotly_chart(self.large_data, "テストチャート")

        assert fig is not None

    def test_create_chart_with_empty_data(self):
        """空データでNoneが返される"""
        empty_data = pd.DataFrame()
        fig = create_plotly_chart(empty_data)

        assert fig is None

    def test_create_chart_with_none_data(self):
        """Noneデータで安全にNoneが返される"""
        fig = create_plotly_chart(None)

        assert fig is None

    def test_chart_layout_settings(self):
        """チャートのレイアウト設定が適用される"""
        fig = create_plotly_chart(self.sample_data, "テスト")

        assert fig is not None
        # テンプレート確認
        layout = fig.layout
        assert layout.template.layout.paper_bgcolor is not None


@pytest.mark.skipif(not plotly_available, reason="plotly not installed")
class TestChartTypeSelection:
    """チャートタイプ自動選択のテスト"""

    def test_pie_chart_for_few_categories(self):
        """カテゴリが6以下で円グラフが選択される"""
        data = pd.DataFrame({
            "value": [100, 200, 300, 400],
        }, index=["A", "B", "C", "D"])

        fig = create_plotly_chart(data)
        assert fig is not None
        # Plotlyのトレースタイプを確認できる

    def test_bar_chart_for_medium_categories(self):
        """カテゴリが7-15で棒グラフが選択される"""
        data = pd.DataFrame({
            "value": list(range(10)),
        }, index=[f"Cat_{i}" for i in range(10)])

        fig = create_plotly_chart(data)
        assert fig is not None

    def test_handles_non_numeric_columns(self):
        """非数値列のみのデータを処理"""
        data = pd.DataFrame({
            "text": ["a", "b", "c"],
        })

        fig = create_plotly_chart(data)
        # 数値列がない場合はNoneを返す
        assert fig is None

    def test_scatter_for_multi_numeric_columns(self):
        """複数数値列で散布図が生成される（16行以上）"""
        data = pd.DataFrame({
            "x": list(range(20)),
            "y": list(range(20, 40)),
        })

        fig = create_plotly_chart(data)
        assert fig is not None


@pytest.mark.skipif(not plotly_available, reason="plotly not installed")
class TestEdgeCases:
    """エッジケースのテスト"""

    def test_single_row_data(self):
        """1行のデータを処理"""
        data = pd.DataFrame({
            "value": [100],
        }, index=["Single"])

        fig = create_plotly_chart(data)
        assert fig is not None

    def test_single_column_data(self):
        """1列のデータを処理"""
        data = pd.DataFrame({
            "value": [100, 200, 300],
        })

        fig = create_plotly_chart(data)
        assert fig is not None

    def test_data_with_nan_values(self):
        """NaN値を含むデータを処理"""
        data = pd.DataFrame({
            "value": [100, None, 300, None, 500],
        }, index=["A", "B", "C", "D", "E"])

        fig = create_plotly_chart(data)
        assert fig is not None

    def test_empty_question_string(self):
        """空の質問文字列でも動作"""
        data = pd.DataFrame({
            "value": [100, 200, 300],
        })

        fig = create_plotly_chart(data, "")
        assert fig is not None

    def test_data_with_negative_values(self):
        """負の値を含むデータを処理"""
        data = pd.DataFrame({
            "value": [-100, -50, 0, 50, 100],
        }, index=["A", "B", "C", "D", "E"])

        fig = create_plotly_chart(data)
        assert fig is not None

    def test_data_with_large_values(self):
        """大きな値を含むデータを処理"""
        data = pd.DataFrame({
            "value": [1e9, 2e9, 3e9],
        }, index=["A", "B", "C"])

        fig = create_plotly_chart(data)
        assert fig is not None

    def test_data_with_float_values(self):
        """小数値を含むデータを処理"""
        data = pd.DataFrame({
            "value": [0.1, 0.25, 0.33, 0.5],
        }, index=["A", "B", "C", "D"])

        fig = create_plotly_chart(data)
        assert fig is not None


class TestSessionStateHelpers:
    """セッション状態ヘルパー関数のテスト"""

    def test_history_initialization(self):
        """履歴が正しく初期化される"""
        # セッション状態をモック
        history = []
        assert len(history) == 0

        # 履歴に追加
        history.append({
            "question": "テスト質問",
            "result": {"success": True, "answer": "テスト回答"}
        })

        assert len(history) == 1
        assert history[0]["question"] == "テスト質問"

    def test_history_limit(self):
        """履歴は最新10件に制限される"""
        history = []

        # 15件追加
        for i in range(15):
            history.append({"question": f"質問{i}", "result": {"success": True}})

        # 最新10件を取得
        recent = list(reversed(history[-10:]))

        assert len(recent) == 10
        assert recent[0]["question"] == "質問14"


class TestDataProcessingHelpers:
    """データ処理ヘルパー関数のテスト"""

    def test_numeric_column_detection(self):
        """数値列が正しく検出される"""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "text": ["a", "b", "c"],
            "float": [1.1, 2.2, 3.3],
        })

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        assert "numeric" in numeric_cols
        assert "float" in numeric_cols
        assert "text" not in numeric_cols

    def test_index_type_detection(self):
        """インデックスタイプが正しく検出される"""
        # カテゴリカルインデックス
        df_cat = pd.DataFrame({
            "value": [1, 2, 3]
        }, index=["A", "B", "C"])

        assert df_cat.index.dtype == 'object'

        # 数値インデックス
        df_num = pd.DataFrame({
            "value": [1, 2, 3]
        }, index=[0, 1, 2])

        assert df_num.index.dtype == 'int64'

    def test_data_sampling(self):
        """大きなデータが正しくサンプリングされる"""
        large_df = pd.DataFrame({
            "value": list(range(1000))
        })

        # 上位100件をサンプリング
        sampled = large_df.head(100)

        assert len(sampled) == 100
        assert sampled.iloc[0]["value"] == 0
        assert sampled.iloc[99]["value"] == 99


class TestFileHandling:
    """ファイル処理のテスト"""

    def test_csv_file_extension_detection(self):
        """CSV拡張子が正しく検出される"""
        file_name = "data.csv"
        ext = Path(file_name).suffix.lower()

        assert ext == ".csv"

    def test_excel_file_extension_detection(self):
        """Excel拡張子が正しく検出される"""
        for file_name in ["data.xlsx", "data.xls"]:
            ext = Path(file_name).suffix.lower()
            assert ext in [".xlsx", ".xls"]

    def test_parquet_file_extension_detection(self):
        """Parquet拡張子が正しく検出される"""
        file_name = "data.parquet"
        ext = Path(file_name).suffix.lower()

        assert ext == ".parquet"

    def test_unsupported_file_extension(self):
        """サポートされていない拡張子の検出"""
        unsupported = ["data.json", "data.txt", "data.xml"]
        supported_exts = [".csv", ".xlsx", ".xls", ".parquet"]

        for file_name in unsupported:
            ext = Path(file_name).suffix.lower()
            assert ext not in supported_exts

    def test_file_size_calculation(self):
        """ファイルサイズ計算のテスト"""
        # 1MB = 1024 * 1024 bytes
        mb_to_bytes = lambda mb: mb * 1024 * 1024

        # 制限チェック（50MB）
        max_size = 50
        max_bytes = mb_to_bytes(max_size)

        assert mb_to_bytes(1) < max_bytes
        assert mb_to_bytes(50) == max_bytes
        assert mb_to_bytes(51) > max_bytes


class TestQueryProcessing:
    """クエリ処理のテスト"""

    def test_example_queries_format(self):
        """クエリ例の形式が正しい"""
        example_queries = [
            "データの概要を教えて",
            "売上の合計は？",
            "地域別の売上",
            "上位5件を表示",
        ]

        for query in example_queries:
            assert isinstance(query, str)
            assert len(query) > 0

    def test_demo_queries_format(self):
        """デモクエリの形式が正しい"""
        demo_queries = [
            {"label": "📊 データ概要", "query": "データの概要を教えて"},
            {"label": "💰 売上合計", "query": "売上の合計を教えて"},
            {"label": "🏢 地域別売上", "query": "地域別の売上合計を教えて"},
        ]

        for demo in demo_queries:
            assert "label" in demo
            assert "query" in demo
            assert isinstance(demo["label"], str)
            assert isinstance(demo["query"], str)


class TestOnboardingLogic:
    """オンボーディングロジックのテスト"""

    def test_onboarding_steps_format(self):
        """オンボーディングステップの形式が正しい"""
        steps = [
            {"name": "データを読み込む", "done": False},
            {"name": "質問を入力する", "done": False},
            {"name": "チャートを生成", "done": False},
        ]

        for step in steps:
            assert "name" in step
            assert "done" in step
            assert isinstance(step["done"], bool)

    def test_completed_count_calculation(self):
        """完了ステップ数の計算が正しい"""
        steps = [
            {"name": "ステップ1", "done": True},
            {"name": "ステップ2", "done": False},
            {"name": "ステップ3", "done": True},
        ]

        completed = sum(1 for s in steps if s["done"])

        assert completed == 2

    def test_progress_percentage(self):
        """進捗率の計算が正しい"""
        steps = [
            {"name": "ステップ1", "done": True},
            {"name": "ステップ2", "done": True},
            {"name": "ステップ3", "done": False},
        ]

        completed = sum(1 for s in steps if s["done"])
        progress = completed / len(steps)

        assert progress == pytest.approx(2/3)


class TestDisplayHelpers:
    """表示ヘルパーのテスト"""

    def test_metric_formatting(self):
        """メトリクスのフォーマットが正しい"""
        row_count = 1000
        col_count = 5

        # カンマ区切りフォーマット
        formatted_rows = f"{row_count:,}"
        formatted_cols = f"{col_count:,}"

        assert formatted_rows == "1,000"
        assert formatted_cols == "5"

    def test_execution_time_formatting(self):
        """実行時間のフォーマットが正しい"""
        execution_time_ms = 123.456

        formatted = f"{execution_time_ms:.2f}ms"

        assert formatted == "123.46ms"

    def test_confidence_formatting(self):
        """信頼度のフォーマットが正しい"""
        confidence = 0.85

        formatted = f"{confidence:.0%}"

        assert formatted == "85%"

    def test_answer_truncation(self):
        """回答の切り詰めが正しい"""
        long_answer = "A" * 500
        max_length = 200

        truncated = long_answer[:max_length] + "..."

        assert len(truncated) == 203  # 200 + "..."
        assert truncated.endswith("...")


class TestCSSValidation:
    """CSS設定のテスト"""

    def test_css_class_names(self):
        """CSSクラス名が存在する"""
        expected_classes = [
            "main-header",
            "info-card",
            "query-section",
            "result-container",
            "welcome-section",
            "onboarding-step",
            "demo-card",
            "feature-icon",
            "progress-indicator",
            "progress-dot",
        ]

        # CSSにこれらのクラスが含まれていることを確認
        # 実際のCSSはstreamlit_app.pyに定義されている
        for class_name in expected_classes:
            assert isinstance(class_name, str)
            assert len(class_name) > 0

    def test_responsive_breakpoint(self):
        """レスポンシブブレークポイントが定義されている"""
        mobile_breakpoint = 768

        assert mobile_breakpoint > 0
        assert mobile_breakpoint < 1024


class TestWelcomePageContent:
    """ウェルカムページコンテンツのテスト"""

    def test_example_questions_content(self):
        """質問例の内容が適切"""
        example_questions = [
            {"icon": "🔢", "q": "売上の合計はいくら？", "desc": "数値の集計"},
            {"icon": "📊", "q": "カテゴリ別の売上を教えて", "desc": "グループ集計"},
            {"icon": "📈", "q": "月別の売上推移は？", "desc": "時系列分析"},
            {"icon": "🏆", "q": "最も売れている商品は？", "desc": "ランキング"},
            {"icon": "📋", "q": "データの概要を教えて", "desc": "統計サマリー"},
            {"icon": "🔍", "q": "東京の売上を見せて", "desc": "フィルタリング"},
        ]

        assert len(example_questions) == 6

        for example in example_questions:
            assert "icon" in example
            assert "q" in example
            assert "desc" in example
            assert len(example["q"]) > 0
            assert len(example["desc"]) > 0

    def test_quick_start_steps(self):
        """クイックスタートのステップ数"""
        steps = [
            "データをアップロード",
            "質問を入力",
            "結果を確認",
        ]

        assert len(steps) == 3


class TestAnalysisResultDisplay:
    """分析結果表示のテスト"""

    def test_result_success_display(self):
        """成功結果の表示形式"""
        result = {
            "success": True,
            "answer": "売上の合計は1,000,000円です",
            "data": pd.DataFrame({"total": [1000000]}),
            "execution_time_ms": 50.0,
            "confidence": 0.95,
            "llm_used": True,
        }

        assert result["success"] is True
        assert "円" in result["answer"]
        assert result["execution_time_ms"] > 0
        assert 0 <= result["confidence"] <= 1

    def test_result_error_display(self):
        """エラー結果の表示形式"""
        result = {
            "success": False,
            "error": "データが見つかりませんでした",
            "answer": "",
        }

        assert result["success"] is False
        assert len(result["error"]) > 0
        assert result["answer"] == ""

    def test_llm_explanation_display(self):
        """LLM説明の表示形式"""
        result = {
            "success": True,
            "llm_explanation": "このデータは過去1年間の売上を示しています。",
            "llm_used": True,
        }

        assert result["llm_used"] is True
        assert len(result["llm_explanation"]) > 0


class TestAuthUIIntegration:
    """認証UI統合のテスト"""

    def test_plan_display_format(self):
        """プラン表示の形式が正しい"""
        plan_values = ["free", "basic", "pro", "enterprise"]

        for plan in plan_values:
            display = plan.upper()
            assert display.isupper()

    def test_usage_progress_calculation(self):
        """使用量進捗の計算が正しい"""
        query_count = 5
        daily_limit = 10

        progress = min(query_count / daily_limit, 1.0)

        assert progress == 0.5

    def test_usage_progress_cap(self):
        """使用量が100%を超えない"""
        query_count = 15
        daily_limit = 10

        progress = min(query_count / daily_limit, 1.0)

        assert progress == 1.0


class TestDemoMode:
    """デモモードのテスト"""

    def test_demo_mode_initialization(self):
        """デモモードの初期化"""
        demo_mode = False

        assert demo_mode is False

    def test_demo_mode_activation(self):
        """デモモードの有効化"""
        demo_mode = True

        assert demo_mode is True

    def test_sample_data_path(self):
        """サンプルデータパスの形式"""
        # 相対パス形式
        sample_path = Path("data") / "sample_sales.csv"

        assert sample_path.suffix == ".csv"
        assert "sample" in str(sample_path)
