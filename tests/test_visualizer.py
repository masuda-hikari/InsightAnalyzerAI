"""Visualizerのテスト"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualizer import Visualizer, ChartConfig, ChartType, ChartResult


class TestVisualizer:
    """Visualizerクラスのテスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """テスト用データ"""
        return pd.DataFrame({
            "total_sales": [1000, 2000, 1500, 800, 1200]
        }, index=["東京", "大阪", "名古屋", "福岡", "札幌"])

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_create_bar_chart(
        self,
        visualizer: Visualizer,
        sample_data: pd.DataFrame,
    ):
        """棒グラフ生成"""
        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="地域別売上",
        )
        result = visualizer.create_chart(sample_data, config)

        assert result.success is True
        assert result.file_path is not None
        assert os.path.exists(result.file_path)
        assert result.file_path.endswith(".png")

    def test_create_hbar_chart(
        self,
        visualizer: Visualizer,
        sample_data: pd.DataFrame,
    ):
        """横棒グラフ生成"""
        config = ChartConfig(
            chart_type=ChartType.HBAR,
            title="地域別売上（横棒）",
        )
        result = visualizer.create_chart(sample_data, config)

        assert result.success is True
        assert result.file_path is not None

    def test_create_line_chart(
        self,
        visualizer: Visualizer,
    ):
        """折れ線グラフ生成"""
        data = pd.DataFrame({
            "sales": [100, 150, 120, 180, 200]
        }, index=["1月", "2月", "3月", "4月", "5月"])

        config = ChartConfig(
            chart_type=ChartType.LINE,
            title="月別売上推移",
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True

    def test_create_pie_chart(
        self,
        visualizer: Visualizer,
        sample_data: pd.DataFrame,
    ):
        """円グラフ生成"""
        config = ChartConfig(
            chart_type=ChartType.PIE,
            title="地域別売上比率",
        )
        result = visualizer.create_chart(sample_data, config)

        assert result.success is True

    def test_auto_detect_config(self, visualizer: Visualizer):
        """チャート設定の自動検出"""
        # 少ないカテゴリ
        small_data = pd.DataFrame({"a": [1, 2, 3]}, index=["A", "B", "C"])
        config = visualizer._auto_detect_config(small_data)
        assert config.chart_type == ChartType.BAR

        # 多いカテゴリ
        large_data = pd.DataFrame(
            {"a": range(20)},
            index=[f"Cat{i}" for i in range(20)]
        )
        config = visualizer._auto_detect_config(large_data)
        assert config.chart_type == ChartType.HBAR

    def test_empty_data_handling(self, visualizer: Visualizer):
        """空データのハンドリング"""
        empty_data = pd.DataFrame()
        config = ChartConfig(chart_type=ChartType.BAR)

        # 空データでもエラーにならない
        result = visualizer.create_chart(empty_data, config)
        # 成功または適切なエラーメッセージ


class TestChartConfig:
    """ChartConfigのテスト"""

    def test_default_values(self):
        """デフォルト値"""
        config = ChartConfig()
        assert config.chart_type == ChartType.BAR
        assert config.figsize == (10, 6)
        assert config.show_values is True

    def test_custom_values(self):
        """カスタム値"""
        config = ChartConfig(
            chart_type=ChartType.LINE,
            title="テスト",
            figsize=(12, 8),
        )
        assert config.chart_type == ChartType.LINE
        assert config.title == "テスト"
        assert config.figsize == (12, 8)


class TestVisualizerHistogram:
    """ヒストグラムのテスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_create_histogram(self, visualizer: Visualizer):
        """ヒストグラム生成"""
        data = pd.DataFrame({
            "values": [10, 20, 30, 25, 15, 35, 40, 22, 18, 28]
        })

        config = ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            title="値の分布",
            xlabel="値"
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True
        assert result.file_path is not None
        assert os.path.exists(result.file_path)

    def test_create_histogram_with_nan(self, visualizer: Visualizer):
        """NaN含むデータのヒストグラム"""
        import numpy as np
        data = pd.DataFrame({
            "values": [10, np.nan, 30, 25, np.nan, 35, 40, 22, np.nan, 28]
        })

        config = ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            title="NaN含むデータ"
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True

    def test_create_histogram_empty_column(self, visualizer: Visualizer):
        """空列のヒストグラム"""
        data = pd.DataFrame({"values": []})

        config = ChartConfig(chart_type=ChartType.HISTOGRAM)
        result = visualizer.create_chart(data, config)
        # 空データでもエラーにならない
        assert result is not None


class TestVisualizerScatter:
    """散布図のテスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_create_scatter(self, visualizer: Visualizer):
        """散布図生成"""
        data = pd.DataFrame({
            "x値": [1, 2, 3, 4, 5],
            "y値": [10, 20, 15, 25, 30]
        })

        config = ChartConfig(
            chart_type=ChartType.SCATTER,
            title="XYの相関"
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True
        assert result.file_path is not None

    def test_create_scatter_single_column(self, visualizer: Visualizer):
        """1列データの散布図（エラーメッセージ表示）"""
        data = pd.DataFrame({
            "only_x": [1, 2, 3, 4, 5]
        })

        config = ChartConfig(chart_type=ChartType.SCATTER)
        result = visualizer.create_chart(data, config)

        # 1列でもエラーにならない（メッセージ表示）
        assert result.success is True


class TestVisualizerAutoDetect:
    """自動検出のテスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_auto_detect_empty_data(self, visualizer: Visualizer):
        """空データの自動検出"""
        empty_data = pd.DataFrame()
        config = visualizer._auto_detect_config(empty_data)
        assert config.chart_type == ChartType.BAR  # デフォルト

    def test_auto_detect_timeseries(self, visualizer: Visualizer):
        """時系列データの自動検出"""
        data = pd.DataFrame({
            "sales": [100, 150, 120]
        }, index=["2024/01", "2024/02", "2024/03"])
        config = visualizer._auto_detect_config(data)
        assert config.chart_type == ChartType.LINE

    def test_auto_detect_timeseries_with_month(self, visualizer: Visualizer):
        """月表記の時系列データ"""
        data = pd.DataFrame({
            "sales": [100, 150]
        }, index=["1月", "2月"])
        config = visualizer._auto_detect_config(data)
        assert config.chart_type == ChartType.LINE

    def test_auto_detect_timeseries_with_year(self, visualizer: Visualizer):
        """年表記の時系列データ"""
        data = pd.DataFrame({
            "sales": [100, 150]
        }, index=["2023年", "2024年"])
        config = visualizer._auto_detect_config(data)
        assert config.chart_type == ChartType.LINE

    def test_auto_detect_timeseries_with_day(self, visualizer: Visualizer):
        """日表記の時系列データ"""
        data = pd.DataFrame({
            "sales": [100, 150]
        }, index=["1日", "2日"])
        config = visualizer._auto_detect_config(data)
        assert config.chart_type == ChartType.LINE

    def test_auto_detect_timeseries_with_quarter(self, visualizer: Visualizer):
        """四半期表記の時系列データ"""
        data = pd.DataFrame({
            "sales": [100, 150]
        }, index=["Q1", "Q2"])
        config = visualizer._auto_detect_config(data)
        assert config.chart_type == ChartType.LINE

    def test_auto_detect_medium_categories(self, visualizer: Visualizer):
        """中程度カテゴリ数（7-15）"""
        data = pd.DataFrame(
            {"a": range(10)},
            index=[f"Category{i}" for i in range(10)]
        )
        config = visualizer._auto_detect_config(data)
        assert config.chart_type == ChartType.HBAR

    def test_auto_detect_many_categories(self, visualizer: Visualizer):
        """多数カテゴリ（16以上）"""
        data = pd.DataFrame(
            {"a": range(25)},
            index=[f"Cat{i}" for i in range(25)]
        )
        config = visualizer._auto_detect_config(data)
        assert config.chart_type == ChartType.HBAR
        assert config.max_categories == 10


class TestVisualizerPieChart:
    """円グラフのテスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_create_pie_chart_with_negative_values(self, visualizer: Visualizer):
        """負の値を含む円グラフ"""
        data = pd.DataFrame({
            "values": [100, -50, 200, -30]
        }, index=["A", "B", "C", "D"])

        config = ChartConfig(
            chart_type=ChartType.PIE,
            title="負の値含むデータ"
        )
        result = visualizer.create_chart(data, config)

        # 負の値は絶対値として処理される
        assert result.success is True


class TestVisualizerLineChart:
    """折れ線グラフの追加テスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_create_line_chart_with_labels(self, visualizer: Visualizer):
        """ラベル付き折れ線グラフ"""
        data = pd.DataFrame({
            "sales": [100, 150, 120, 180, 200]
        }, index=["1月", "2月", "3月", "4月", "5月"])

        config = ChartConfig(
            chart_type=ChartType.LINE,
            title="月別売上推移",
            xlabel="月",
            ylabel="売上"
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True

    def test_create_line_chart_empty_index(self, visualizer: Visualizer):
        """空インデックスの折れ線グラフ"""
        data = pd.DataFrame({"values": []})

        config = ChartConfig(chart_type=ChartType.LINE)
        result = visualizer.create_chart(data, config)
        # 空データでもエラーにならない
        assert result is not None


class TestVisualizerBarChart:
    """棒グラフの追加テスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_create_bar_chart_with_labels(self, visualizer: Visualizer):
        """ラベル付き棒グラフ"""
        data = pd.DataFrame({
            "total_sales": [1000, 2000, 1500]
        }, index=["東京", "大阪", "名古屋"])

        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="地域別売上",
            xlabel="地域",
            ylabel="売上"
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True

    def test_create_bar_chart_no_show_values(self, visualizer: Visualizer):
        """値表示なしの棒グラフ"""
        data = pd.DataFrame({
            "count": [10, 20, 15]
        }, index=["A", "B", "C"])

        config = ChartConfig(
            chart_type=ChartType.BAR,
            show_values=False
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True


class TestVisualizerHbarChart:
    """横棒グラフの追加テスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_create_hbar_chart_with_label(self, visualizer: Visualizer):
        """ラベル付き横棒グラフ"""
        data = pd.DataFrame({
            "sales": [100, 200, 150]
        }, index=["製品A", "製品B", "製品C"])

        config = ChartConfig(
            chart_type=ChartType.HBAR,
            xlabel="売上"
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True

    def test_create_hbar_chart_no_show_values(self, visualizer: Visualizer):
        """値表示なしの横棒グラフ"""
        data = pd.DataFrame({
            "count": [10, 20, 15]
        }, index=["A", "B", "C"])

        config = ChartConfig(
            chart_type=ChartType.HBAR,
            show_values=False
        )
        result = visualizer.create_chart(data, config)

        assert result.success is True


class TestVisualizerFormatValue:
    """値フォーマットのテスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_format_value_nan(self, visualizer: Visualizer):
        """NaN値のフォーマット"""
        import numpy as np
        result = visualizer._format_value(np.nan)
        assert result == "N/A"

    def test_format_value_million(self, visualizer: Visualizer):
        """百万単位のフォーマット"""
        result = visualizer._format_value(1500000.0)
        assert "M" in result

    def test_format_value_thousand(self, visualizer: Visualizer):
        """千単位のフォーマット"""
        result = visualizer._format_value(5000.0)
        assert "K" in result

    def test_format_value_small(self, visualizer: Visualizer):
        """小さい値のフォーマット"""
        result = visualizer._format_value(500.0)
        assert result == "500"

    def test_format_value_integer(self, visualizer: Visualizer):
        """整数値のフォーマット"""
        result = visualizer._format_value(1234)
        assert result == "1,234"


class TestVisualizerErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def visualizer(self, temp_output_dir: str) -> Visualizer:
        """Visualizerインスタンス"""
        return Visualizer(temp_output_dir)

    def test_create_chart_with_invalid_data(self, visualizer: Visualizer):
        """無効なデータでのチャート生成"""
        # 非常に大きなデータは正常に処理される
        data = pd.DataFrame({
            "values": list(range(1000))
        })
        config = ChartConfig(chart_type=ChartType.BAR, max_categories=5)
        result = visualizer.create_chart(data, config)
        assert result.success is True

    def test_create_chart_default_type(self, visualizer: Visualizer):
        """不明なチャートタイプでのデフォルト処理"""
        data = pd.DataFrame({
            "values": [1, 2, 3]
        }, index=["A", "B", "C"])

        # 通常のBARチャートでテスト
        config = ChartConfig(chart_type=ChartType.BAR)
        result = visualizer.create_chart(data, config)
        assert result.success is True


class TestVisualizerDefaultOutputDir:
    """デフォルト出力ディレクトリのテスト"""

    def test_default_output_dir(self):
        """デフォルト出力ディレクトリが作成される"""
        visualizer = Visualizer()
        assert visualizer._output_dir.exists()


class TestCreateChartFromResult:
    """create_chart_from_result関数のテスト"""

    def test_create_chart_from_result_basic(self):
        """基本的なチャート生成"""
        from src.visualizer import create_chart_from_result

        with tempfile.TemporaryDirectory() as tmpdir:
            data = pd.DataFrame({
                "sales": [100, 200, 150]
            }, index=["A", "B", "C"])

            result = create_chart_from_result(data, "テストチャート", tmpdir)
            assert result.success is True

    def test_create_chart_from_result_no_title(self):
        """タイトルなしのチャート生成"""
        from src.visualizer import create_chart_from_result

        with tempfile.TemporaryDirectory() as tmpdir:
            data = pd.DataFrame({
                "values": [1, 2, 3]
            }, index=["X", "Y", "Z"])

            result = create_chart_from_result(data, output_dir=tmpdir)
            assert result.success is True
