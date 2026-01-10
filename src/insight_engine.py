"""
自動インサイト発見エンジン

データをスキャンし、パターン・異常値・トレンドを自動検出する
Phase 5: 収益価値向上のためのプレミアム機能

収益貢献:
- 有料プラン（Basic/Pro）の差別化機能
- ユーザーが「手間を省ける」価値を提供
- データアナリストの代替として月額課金を正当化
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime


class InsightType(Enum):
    """インサイトの種類"""
    OVERVIEW = "overview"           # データ概要
    TREND = "trend"                 # トレンド
    ANOMALY = "anomaly"             # 異常値
    CORRELATION = "correlation"     # 相関
    DISTRIBUTION = "distribution"   # 分布
    TOP_PERFORMERS = "top"          # 上位項目
    BOTTOM_PERFORMERS = "bottom"    # 下位項目
    SEASONALITY = "seasonality"     # 季節性
    MISSING_DATA = "missing"        # 欠損データ
    RECOMMENDATION = "recommendation"  # 推奨アクション


class InsightSeverity(Enum):
    """インサイトの重要度"""
    INFO = "info"           # 情報
    WARNING = "warning"     # 注意
    CRITICAL = "critical"   # 重要


@dataclass
class Insight:
    """個別のインサイト"""
    insight_type: InsightType
    title: str
    description: str
    severity: InsightSeverity = InsightSeverity.INFO
    confidence: float = 0.8  # 0.0-1.0
    data: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    columns_involved: List[str] = field(default_factory=list)


@dataclass
class InsightReport:
    """インサイトレポート全体"""
    insights: List[Insight]
    generated_at: str
    data_rows: int
    data_columns: int
    analysis_time_ms: float = 0.0

    @property
    def summary(self) -> str:
        """レポートサマリー"""
        critical = sum(1 for i in self.insights if i.severity == InsightSeverity.CRITICAL)
        warning = sum(1 for i in self.insights if i.severity == InsightSeverity.WARNING)
        return f"発見: {len(self.insights)}件（重要: {critical}件, 注意: {warning}件）"


class InsightEngine:
    """
    自動インサイト発見エンジン

    データを多角的に分析し、ビジネスに有用なインサイトを自動生成する

    主要機能:
    - 基本統計・分布分析
    - 異常値検出（IQR, Zスコア）
    - トレンド分析（時系列データ）
    - 相関分析
    - カテゴリ別分析
    - 推奨アクション生成
    """

    # 異常値検出の閾値
    ANOMALY_Z_THRESHOLD = 3.0  # Zスコア閾値
    ANOMALY_IQR_MULTIPLIER = 1.5  # IQR倍率

    # 相関分析の閾値
    CORRELATION_THRESHOLD = 0.7  # 強い相関とみなす閾値

    # 欠損データの閾値
    MISSING_THRESHOLD = 0.05  # 5%以上で警告

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 分析対象のDataFrame
        """
        if df is None or len(df) == 0:
            raise ValueError("有効なDataFrameが必要です")

        self._df = df
        self._numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self._categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self._datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    def generate_report(self, max_insights: int = 20) -> InsightReport:
        """
        包括的なインサイトレポートを生成

        Args:
            max_insights: 最大インサイト数

        Returns:
            インサイトレポート
        """
        import time
        start_time = time.perf_counter()

        insights: List[Insight] = []

        # 各分析を実行
        insights.extend(self._analyze_overview())
        insights.extend(self._analyze_missing_data())
        insights.extend(self._analyze_distributions())
        insights.extend(self._analyze_anomalies())
        insights.extend(self._analyze_correlations())
        insights.extend(self._analyze_top_bottom())
        insights.extend(self._analyze_trends())

        # 重要度でソート（CRITICAL > WARNING > INFO）
        severity_order = {
            InsightSeverity.CRITICAL: 0,
            InsightSeverity.WARNING: 1,
            InsightSeverity.INFO: 2,
        }
        insights.sort(key=lambda x: (severity_order[x.severity], -x.confidence))

        # 最大数で制限
        insights = insights[:max_insights]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return InsightReport(
            insights=insights,
            generated_at=datetime.now().isoformat(),
            data_rows=len(self._df),
            data_columns=len(self._df.columns),
            analysis_time_ms=elapsed_ms,
        )

    def _analyze_overview(self) -> List[Insight]:
        """データ概要の分析"""
        insights = []

        # 基本情報
        insights.append(Insight(
            insight_type=InsightType.OVERVIEW,
            title="データ概要",
            description=f"データには{len(self._df):,}件のレコードと{len(self._df.columns)}個のカラムが含まれています。",
            data={
                "rows": len(self._df),
                "columns": len(self._df.columns),
                "numeric_columns": len(self._numeric_cols),
                "categorical_columns": len(self._categorical_cols),
            }
        ))

        # 数値カラムの合計
        if self._numeric_cols:
            totals = {}
            for col in self._numeric_cols[:5]:  # 最大5カラム
                totals[col] = float(self._df[col].sum())

            # 最大の数値カラムを特定
            if totals:
                max_col = max(totals, key=totals.get)
                insights.append(Insight(
                    insight_type=InsightType.OVERVIEW,
                    title=f"{max_col}の合計",
                    description=f"{max_col}の合計値は ¥{totals[max_col]:,.0f} です。",
                    data={"column": max_col, "total": totals[max_col]},
                    columns_involved=[max_col],
                ))

        return insights

    def _analyze_missing_data(self) -> List[Insight]:
        """欠損データの分析"""
        insights = []

        missing_rates = self._df.isnull().mean()
        missing_cols = missing_rates[missing_rates > 0]

        if len(missing_cols) > 0:
            # 欠損率が高いカラムを警告
            high_missing = missing_cols[missing_cols > self.MISSING_THRESHOLD]

            if len(high_missing) > 0:
                for col in high_missing.index[:3]:  # 最大3カラム
                    rate = high_missing[col]
                    severity = InsightSeverity.CRITICAL if rate > 0.2 else InsightSeverity.WARNING

                    insights.append(Insight(
                        insight_type=InsightType.MISSING_DATA,
                        title=f"{col}に欠損データ",
                        description=f"{col}カラムは{rate:.1%}のデータが欠損しています。分析結果に影響する可能性があります。",
                        severity=severity,
                        confidence=1.0,
                        data={"column": col, "missing_rate": float(rate)},
                        recommendation=f"{col}の欠損値を補完するか、このカラムを除外することを検討してください。",
                        columns_involved=[col],
                    ))

        return insights

    def _analyze_distributions(self) -> List[Insight]:
        """分布の分析"""
        insights = []

        for col in self._numeric_cols[:5]:  # 最大5カラム
            data = self._df[col].dropna()
            if len(data) < 10:
                continue

            # 歪度の分析
            skewness = data.skew()
            if abs(skewness) > 1.0:
                direction = "右に偏っています（大きい値が少数）" if skewness > 0 else "左に偏っています（小さい値が少数）"
                insights.append(Insight(
                    insight_type=InsightType.DISTRIBUTION,
                    title=f"{col}の分布偏り",
                    description=f"{col}の分布は{direction}。歪度: {skewness:.2f}",
                    confidence=0.85,
                    data={"column": col, "skewness": float(skewness)},
                    columns_involved=[col],
                ))

            # 範囲の分析
            min_val = data.min()
            max_val = data.max()
            range_val = max_val - min_val

            if range_val > 0:
                cv = data.std() / data.mean() if data.mean() != 0 else 0  # 変動係数
                if cv > 1.0:
                    insights.append(Insight(
                        insight_type=InsightType.DISTRIBUTION,
                        title=f"{col}のばらつきが大きい",
                        description=f"{col}は非常にばらつきが大きいです（変動係数: {cv:.2f}）。範囲: {min_val:,.0f} 〜 {max_val:,.0f}",
                        severity=InsightSeverity.WARNING,
                        data={"column": col, "cv": float(cv), "min": float(min_val), "max": float(max_val)},
                        columns_involved=[col],
                    ))

        return insights

    def _analyze_anomalies(self) -> List[Insight]:
        """異常値の検出"""
        insights = []

        for col in self._numeric_cols[:5]:  # 最大5カラム
            data = self._df[col].dropna()
            if len(data) < 10:
                continue

            # Zスコアによる異常値検出
            mean = data.mean()
            std = data.std()
            if std == 0:
                continue

            z_scores = np.abs((data - mean) / std)
            anomalies = data[z_scores > self.ANOMALY_Z_THRESHOLD]

            if len(anomalies) > 0:
                anomaly_rate = len(anomalies) / len(data)

                if anomaly_rate > 0.01:  # 1%以上が異常値
                    severity = InsightSeverity.CRITICAL if anomaly_rate > 0.05 else InsightSeverity.WARNING

                    insights.append(Insight(
                        insight_type=InsightType.ANOMALY,
                        title=f"{col}に異常値検出",
                        description=f"{col}には{len(anomalies):,}件（{anomaly_rate:.1%}）の異常値が含まれています。最大異常値: {anomalies.max():,.0f}",
                        severity=severity,
                        confidence=0.9,
                        data={
                            "column": col,
                            "anomaly_count": len(anomalies),
                            "anomaly_rate": float(anomaly_rate),
                            "max_anomaly": float(anomalies.max()),
                            "min_anomaly": float(anomalies.min()),
                        },
                        recommendation="異常値を個別に確認し、データ入力ミスやシステムエラーの可能性を調査してください。",
                        columns_involved=[col],
                    ))

        return insights

    def _analyze_correlations(self) -> List[Insight]:
        """相関分析"""
        insights = []

        if len(self._numeric_cols) < 2:
            return insights

        # 相関行列を計算
        numeric_df = self._df[self._numeric_cols].dropna()
        if len(numeric_df) < 10:
            return insights

        corr_matrix = numeric_df.corr()

        # 強い相関を持つペアを検出
        checked_pairs = set()
        for i, col1 in enumerate(self._numeric_cols):
            for j, col2 in enumerate(self._numeric_cols):
                if i >= j:
                    continue

                pair_key = tuple(sorted([col1, col2]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > self.CORRELATION_THRESHOLD:
                    direction = "正" if corr > 0 else "負"
                    strength = "非常に強い" if abs(corr) > 0.9 else "強い"

                    insights.append(Insight(
                        insight_type=InsightType.CORRELATION,
                        title=f"{col1}と{col2}に{strength}{direction}の相関",
                        description=f"{col1}と{col2}の間には{strength}{direction}の相関（r={corr:.2f}）があります。一方が増加すると他方も{'増加' if corr > 0 else '減少'}する傾向があります。",
                        confidence=0.95,
                        data={"column1": col1, "column2": col2, "correlation": float(corr)},
                        columns_involved=[col1, col2],
                    ))

        return insights

    def _analyze_top_bottom(self) -> List[Insight]:
        """上位・下位項目の分析"""
        insights = []

        if not self._categorical_cols or not self._numeric_cols:
            return insights

        # カテゴリカルカラムの最初と数値カラムの最初で分析
        cat_col = self._categorical_cols[0]
        num_col = self._numeric_cols[0]

        grouped = self._df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)

        if len(grouped) >= 3:
            # 上位3件
            top3 = grouped.head(3)
            total = grouped.sum()
            top3_share = top3.sum() / total if total > 0 else 0

            insights.append(Insight(
                insight_type=InsightType.TOP_PERFORMERS,
                title=f"上位{cat_col}",
                description=f"{cat_col}別{num_col}の上位3件: {', '.join([f'{k}(¥{v:,.0f})' for k, v in top3.items()])}。上位3件で全体の{top3_share:.1%}を占めています。",
                confidence=1.0,
                data={
                    "category_column": cat_col,
                    "value_column": num_col,
                    "top_items": {str(k): float(v) for k, v in top3.items()},
                    "top3_share": float(top3_share),
                },
                columns_involved=[cat_col, num_col],
            ))

            # 下位3件（パフォーマンス改善の機会）
            if len(grouped) >= 6:
                bottom3 = grouped.tail(3)
                bottom3_share = bottom3.sum() / total if total > 0 else 0

                if bottom3_share < 0.05:  # 下位3件が5%未満
                    insights.append(Insight(
                        insight_type=InsightType.BOTTOM_PERFORMERS,
                        title=f"改善機会: 下位{cat_col}",
                        description=f"下位3件の{cat_col}は全体の{bottom3_share:.1%}のみ。これらの改善またはリソース再配分を検討してください。",
                        severity=InsightSeverity.WARNING,
                        data={
                            "category_column": cat_col,
                            "value_column": num_col,
                            "bottom_items": {str(k): float(v) for k, v in bottom3.items()},
                            "bottom3_share": float(bottom3_share),
                        },
                        recommendation="下位カテゴリの原因を分析し、マーケティング強化またはリソース再配分を検討してください。",
                        columns_involved=[cat_col, num_col],
                    ))

        return insights

    def _analyze_trends(self) -> List[Insight]:
        """トレンド分析（時系列データがある場合）"""
        insights = []

        # 日付カラムと数値カラムがある場合のみ
        if not self._datetime_cols and not self._numeric_cols:
            return insights

        # 日付カラムを探す（明示的なdatetime型がない場合は推測）
        date_col = None
        if self._datetime_cols:
            date_col = self._datetime_cols[0]
        else:
            # カラム名から日付を推測
            for col in self._df.columns:
                if any(kw in col.lower() for kw in ["date", "time", "日付", "日時", "年月"]):
                    try:
                        pd.to_datetime(self._df[col])
                        date_col = col
                        break
                    except Exception:
                        continue

        if date_col is None:
            return insights

        try:
            # 日付でソート
            df_sorted = self._df.copy()
            if date_col not in self._datetime_cols:
                df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
            df_sorted = df_sorted.sort_values(date_col)

            # 数値カラムのトレンドを分析
            for num_col in self._numeric_cols[:3]:  # 最大3カラム
                # 月別集計
                df_sorted["_month"] = df_sorted[date_col].dt.to_period("M")
                monthly = df_sorted.groupby("_month")[num_col].sum()

                if len(monthly) >= 3:
                    # トレンド方向を判定
                    first_half = monthly.iloc[:len(monthly)//2].mean()
                    second_half = monthly.iloc[len(monthly)//2:].mean()

                    if second_half > first_half * 1.1:  # 10%以上増加
                        change_rate = (second_half - first_half) / first_half
                        insights.append(Insight(
                            insight_type=InsightType.TREND,
                            title=f"{num_col}が増加傾向",
                            description=f"{num_col}は期間を通じて増加傾向にあります（前半比{change_rate:.1%}増加）。",
                            confidence=0.8,
                            data={
                                "column": num_col,
                                "first_half_avg": float(first_half),
                                "second_half_avg": float(second_half),
                                "change_rate": float(change_rate),
                            },
                            columns_involved=[date_col, num_col],
                        ))
                    elif second_half < first_half * 0.9:  # 10%以上減少
                        change_rate = (first_half - second_half) / first_half
                        insights.append(Insight(
                            insight_type=InsightType.TREND,
                            title=f"{num_col}が減少傾向",
                            description=f"{num_col}は期間を通じて減少傾向にあります（前半比{change_rate:.1%}減少）。",
                            severity=InsightSeverity.WARNING,
                            confidence=0.8,
                            data={
                                "column": num_col,
                                "first_half_avg": float(first_half),
                                "second_half_avg": float(second_half),
                                "change_rate": float(-change_rate),
                            },
                            recommendation="減少の原因を調査し、対策を検討してください。",
                            columns_involved=[date_col, num_col],
                        ))

                # 一時カラムを削除
                df_sorted.drop("_month", axis=1, inplace=True, errors="ignore")

        except Exception:
            # トレンド分析でエラーが発生しても続行
            pass

        return insights


def generate_insights(df: pd.DataFrame, max_insights: int = 20) -> InsightReport:
    """
    便利関数: DataFrameからインサイトレポートを生成

    Args:
        df: 分析対象のDataFrame
        max_insights: 最大インサイト数

    Returns:
        インサイトレポート
    """
    engine = InsightEngine(df)
    return engine.generate_report(max_insights=max_insights)
