"""
可視化モジュール

データ分析結果をチャートとして可視化する
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union, List, Tuple
from datetime import datetime

import pandas as pd

# matplotlibのバックエンド設定（GUI不要）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class ChartType(Enum):
    """チャートタイプの列挙"""
    BAR = auto()          # 棒グラフ
    LINE = auto()         # 折れ線グラフ
    PIE = auto()          # 円グラフ
    SCATTER = auto()      # 散布図
    HISTOGRAM = auto()    # ヒストグラム
    HBAR = auto()         # 横棒グラフ


@dataclass
class ChartConfig:
    """チャート設定"""
    chart_type: ChartType = ChartType.BAR
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[int, int] = (10, 6)
    color: str = "#4A90D9"
    show_values: bool = True
    max_categories: int = 10  # 表示する最大カテゴリ数


@dataclass
class ChartResult:
    """チャート生成結果"""
    success: bool
    file_path: Optional[str] = None
    chart_type: Optional[ChartType] = None
    error: Optional[str] = None


class Visualizer:
    """
    データ可視化クラス

    DataFrameからチャートを生成し、画像ファイルとして保存する
    """

    DEFAULT_OUTPUT_DIR = "output"

    def __init__(self, output_dir: Optional[str] = None):
        """
        Args:
            output_dir: 出力ディレクトリ
        """
        self._output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def create_chart(
        self,
        data: pd.DataFrame,
        config: Optional[ChartConfig] = None,
    ) -> ChartResult:
        """
        チャートを生成

        Args:
            data: 可視化するデータ
            config: チャート設定

        Returns:
            生成結果
        """
        if config is None:
            config = self._auto_detect_config(data)

        try:
            # チャートタイプに応じた生成
            fig, ax = plt.subplots(figsize=config.figsize)

            if config.chart_type == ChartType.BAR:
                self._create_bar_chart(ax, data, config)
            elif config.chart_type == ChartType.HBAR:
                self._create_hbar_chart(ax, data, config)
            elif config.chart_type == ChartType.LINE:
                self._create_line_chart(ax, data, config)
            elif config.chart_type == ChartType.PIE:
                self._create_pie_chart(ax, data, config)
            elif config.chart_type == ChartType.HISTOGRAM:
                self._create_histogram(ax, data, config)
            elif config.chart_type == ChartType.SCATTER:
                self._create_scatter(ax, data, config)
            else:
                self._create_bar_chart(ax, data, config)

            # タイトル設定
            if config.title:
                ax.set_title(config.title, fontsize=14, fontweight='bold')

            # レイアウト調整
            plt.tight_layout()

            # ファイル保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.png"
            file_path = self._output_dir / filename

            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            return ChartResult(
                success=True,
                file_path=str(file_path),
                chart_type=config.chart_type,
            )

        except Exception as e:
            plt.close('all')
            return ChartResult(
                success=False,
                error=str(e),
            )

    def _auto_detect_config(self, data: pd.DataFrame) -> ChartConfig:
        """データからチャート設定を自動判定"""
        config = ChartConfig()

        if data.empty:
            return config

        # インデックスがカテゴリ的かどうか
        index_is_categorical = data.index.dtype == 'object' or len(data.index) <= 20

        # 列数
        num_columns = len(data.columns)

        # カテゴリ数に応じたチャートタイプ選択
        num_categories = len(data.index)

        if num_categories <= 6:
            # 少ないカテゴリは円グラフも可
            config.chart_type = ChartType.BAR
        elif num_categories <= 15:
            # 中程度は横棒グラフ
            config.chart_type = ChartType.HBAR
        else:
            # 多い場合は上位のみ表示
            config.chart_type = ChartType.HBAR
            config.max_categories = 10

        # 時系列っぽい場合は折れ線
        if index_is_categorical:
            index_sample = str(data.index[0]) if len(data.index) > 0 else ""
            if any(kw in index_sample for kw in ["月", "年", "日", "Q", "/"]):
                config.chart_type = ChartType.LINE

        return config

    def _create_bar_chart(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        config: ChartConfig,
    ) -> None:
        """棒グラフを作成"""
        # データの準備
        if len(data.columns) > 0:
            col = data.columns[0]
            values = data[col].head(config.max_categories)
        else:
            values = data.iloc[:, 0].head(config.max_categories) if len(data) > 0 else pd.Series()

        # 棒グラフ描画
        bars = ax.bar(range(len(values)), values, color=config.color, edgecolor='white')

        # ラベル設定
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.index, rotation=45, ha='right')

        if config.xlabel:
            ax.set_xlabel(config.xlabel)
        if config.ylabel:
            ax.set_ylabel(config.ylabel)

        # 値を表示
        if config.show_values:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(
                    self._format_value(val),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _create_hbar_chart(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        config: ChartConfig,
    ) -> None:
        """横棒グラフを作成"""
        if len(data.columns) > 0:
            col = data.columns[0]
            values = data[col].head(config.max_categories)
        else:
            values = data.iloc[:, 0].head(config.max_categories) if len(data) > 0 else pd.Series()

        # 降順にソート（上が大きい値）
        values = values.sort_values(ascending=True)

        # 横棒グラフ描画
        bars = ax.barh(range(len(values)), values, color=config.color, edgecolor='white')

        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index)

        if config.xlabel:
            ax.set_xlabel(config.xlabel)

        # 値を表示
        if config.show_values:
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.annotate(
                    self._format_value(val),
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left',
                    va='center',
                    fontsize=9,
                )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _create_line_chart(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        config: ChartConfig,
    ) -> None:
        """折れ線グラフを作成"""
        if len(data.columns) > 0:
            col = data.columns[0]
            values = data[col]
        else:
            values = data.iloc[:, 0] if len(data) > 0 else pd.Series()

        ax.plot(range(len(values)), values, marker='o', color=config.color, linewidth=2)

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.index, rotation=45, ha='right')

        if config.xlabel:
            ax.set_xlabel(config.xlabel)
        if config.ylabel:
            ax.set_ylabel(config.ylabel)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _create_pie_chart(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        config: ChartConfig,
    ) -> None:
        """円グラフを作成"""
        if len(data.columns) > 0:
            col = data.columns[0]
            values = data[col].head(config.max_categories)
        else:
            values = data.iloc[:, 0].head(config.max_categories) if len(data) > 0 else pd.Series()

        # 負の値は扱えないので絶対値
        values = values.abs()

        wedges, texts, autotexts = ax.pie(
            values,
            labels=values.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Set3.colors[:len(values)],
        )

        ax.axis('equal')

    def _create_histogram(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        config: ChartConfig,
    ) -> None:
        """ヒストグラムを作成"""
        if len(data.columns) > 0:
            col = data.columns[0]
            values = data[col].dropna()
        else:
            values = data.iloc[:, 0].dropna() if len(data) > 0 else pd.Series()

        ax.hist(values, bins=20, color=config.color, edgecolor='white')

        if config.xlabel:
            ax.set_xlabel(config.xlabel)
        ax.set_ylabel("頻度")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _create_scatter(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        config: ChartConfig,
    ) -> None:
        """散布図を作成"""
        if len(data.columns) >= 2:
            x = data.iloc[:, 0]
            y = data.iloc[:, 1]
            ax.scatter(x, y, color=config.color, alpha=0.6)
            ax.set_xlabel(data.columns[0])
            ax.set_ylabel(data.columns[1])
        else:
            ax.text(0.5, 0.5, "散布図には2列以上必要です", ha='center', va='center')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _format_value(self, value: Union[int, float]) -> str:
        """値をフォーマット"""
        if pd.isna(value):
            return "N/A"
        if isinstance(value, float):
            if abs(value) >= 1_000_000:
                return f"¥{value / 1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                return f"¥{value / 1_000:.1f}K"
            else:
                return f"{value:,.0f}"
        return f"{value:,}"


def create_chart_from_result(
    data: pd.DataFrame,
    title: str = "",
    output_dir: Optional[str] = None,
) -> ChartResult:
    """分析結果からチャートを作成するヘルパー関数"""
    visualizer = Visualizer(output_dir)
    config = ChartConfig(title=title)
    return visualizer.create_chart(data, config)
