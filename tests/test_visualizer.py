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
