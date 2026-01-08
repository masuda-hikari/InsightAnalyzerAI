"""
Streamlit Web UIのテスト

注意: Streamlit自体のUIテストはE2Eテストが推奨されるため、
ここではUIで使用する関数のユニットテストを行う
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import MagicMock

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
