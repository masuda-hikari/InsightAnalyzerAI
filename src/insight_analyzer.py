"""
InsightAnalyzerAI メインモジュール

自然言語クエリを受け付け、データ分析を実行する
"""

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import pandas as pd

from .data_loader import DataLoader
from .query_parser import QueryParser, ParsedQuery, QueryType
from .executor import QueryExecutor, SafeExecutor, ExecutionResult
from .visualizer import Visualizer, ChartConfig, ChartResult


@dataclass
class AnalysisResult:
    """分析結果を格納するクラス"""
    answer: str
    data: Optional[pd.DataFrame] = None
    chart_path: Optional[str] = None
    query_used: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    confidence: float = 1.0


class InsightAnalyzer:
    """
    メイン分析クラス

    データを読み込み、自然言語クエリに基づいて分析を実行する

    アーキテクチャ:
        DataLoader -> QueryParser -> QueryExecutor -> Visualizer
    """

    def __init__(
        self,
        data_source: Optional[Union[str, Path, pd.DataFrame]] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            data_source: CSVパス、またはDataFrame
            output_dir: チャート出力ディレクトリ
        """
        self._loader = DataLoader()
        self._df: Optional[pd.DataFrame] = None
        self._parser: Optional[QueryParser] = None
        self._visualizer = Visualizer(output_dir)

        if data_source is not None:
            self.load_data(data_source)

    def load_data(self, source: Union[str, Path, pd.DataFrame]) -> None:
        """
        データを読み込む

        Args:
            source: ファイルパスまたはDataFrame
        """
        if isinstance(source, pd.DataFrame):
            self._df = source
            # DataFrameから直接読み込んだ場合、loaderのメタデータを更新
            self._loader._dataframe = source
            self._loader._update_metadata()
        else:
            self._df = self._loader.load(source)

        # パーサーを初期化
        self._parser = QueryParser()
        self._parser.set_schema(self._df)

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """現在のDataFrameを取得"""
        return self._df

    @property
    def schema(self) -> str:
        """データスキーマを取得"""
        return self._loader.get_schema()

    @property
    def metadata(self) -> dict:
        """データのメタデータを取得"""
        return self._loader.metadata

    def ask(
        self,
        question: str,
        generate_chart: bool = False,
    ) -> AnalysisResult:
        """
        自然言語で質問し、分析結果を取得

        Args:
            question: 自然言語の質問
            generate_chart: チャートを生成するか

        Returns:
            分析結果
        """
        if self._df is None:
            return AnalysisResult(
                answer="",
                success=False,
                error="データが読み込まれていません。load_data()でデータを読み込んでください。"
            )

        try:
            # 1. クエリ解析
            parsed = self._parser.parse(question)

            # 2. クエリ実行
            executor = SafeExecutor(self._df)
            exec_result = executor.execute_safe(parsed)

            if not exec_result.success:
                return AnalysisResult(
                    answer="",
                    success=False,
                    error=exec_result.error,
                    execution_time_ms=exec_result.execution_time_ms,
                )

            # 3. 回答生成
            answer = self._format_answer(parsed, exec_result)

            # 4. チャート生成（オプション）
            chart_path = None
            if generate_chart and exec_result.data is not None:
                chart_result = self._generate_chart(parsed, exec_result.data)
                if chart_result.success:
                    chart_path = chart_result.file_path

            return AnalysisResult(
                answer=answer,
                data=exec_result.data,
                chart_path=chart_path,
                query_used=exec_result.query_code,
                success=True,
                execution_time_ms=exec_result.execution_time_ms,
                confidence=parsed.confidence,
            )

        except Exception as e:
            return AnalysisResult(
                answer="",
                success=False,
                error=f"分析中にエラーが発生しました: {str(e)}"
            )

    def _format_answer(
        self,
        parsed: ParsedQuery,
        result: ExecutionResult,
    ) -> str:
        """実行結果を自然言語の回答にフォーマット"""
        query_type = parsed.query_type

        # スカラー値の場合
        if result.value is not None:
            value = result.value
            col = parsed.target_column or "値"

            # 金額判定
            is_currency = col and any(
                kw in col.lower() for kw in ["price", "sales", "金額", "売上"]
            )

            if query_type == QueryType.SUM:
                if is_currency:
                    return f"{col}の合計: ¥{value:,.0f}"
                return f"{col}の合計: {value:,.2f}"

            elif query_type == QueryType.MEAN:
                if is_currency:
                    return f"{col}の平均: ¥{value:,.0f}"
                return f"{col}の平均: {value:,.2f}"

            elif query_type == QueryType.COUNT:
                return f"データ件数: {value:,}件"

        # DataFrame結果の場合
        if result.data is not None:
            if query_type == QueryType.GROUPBY:
                return self._format_groupby_answer(parsed, result.data)

            elif query_type == QueryType.DESCRIBE:
                return f"データの基本統計:\n{result.data.to_string()}"

            else:
                return result.data.to_string()

        return "結果を取得できませんでした"

    def _format_groupby_answer(
        self,
        parsed: ParsedQuery,
        data: pd.DataFrame,
    ) -> str:
        """グループ別集計結果をフォーマット"""
        group_col = parsed.group_column or "カテゴリ"
        target_col = parsed.target_column

        # 金額判定
        is_currency = target_col and any(
            kw in target_col.lower() for kw in ["price", "sales", "金額", "売上"]
        )

        if target_col:
            lines = [f"{group_col}別の{target_col}合計:"]
        else:
            lines = [f"{group_col}別の件数:"]

        # DataFrameまたはSeriesからデータ取得
        if isinstance(data, pd.DataFrame) and len(data.columns) > 0:
            col = data.columns[0]
            for idx, row in data.iterrows():
                val = row[col]
                if is_currency:
                    lines.append(f"  {idx}: ¥{val:,.0f}")
                else:
                    lines.append(f"  {idx}: {val:,.2f}")
        else:
            for idx, val in data.items():
                if is_currency:
                    lines.append(f"  {idx}: ¥{val:,.0f}")
                else:
                    lines.append(f"  {idx}: {val:,.2f}")

        return "\n".join(lines)

    def _generate_chart(
        self,
        parsed: ParsedQuery,
        data: pd.DataFrame,
    ) -> ChartResult:
        """クエリ結果からチャートを生成"""
        title = parsed.original_question
        config = ChartConfig(title=title)

        return self._visualizer.create_chart(data, config)

    def get_summary(self) -> AnalysisResult:
        """データの要約を取得"""
        return self.ask("データの概要を教えて")

    def get_insights(self) -> list[str]:
        """
        自動インサイト生成（Phase 5で拡張予定）

        現在は基本的な統計情報のみ
        """
        if self._df is None:
            return ["データが読み込まれていません"]

        insights = []

        # 基本情報
        insights.append(f"データ件数: {len(self._df):,}件")
        insights.append(f"カラム数: {len(self._df.columns)}列")

        # 数値カラムの統計
        numeric_cols = self._df.select_dtypes(include=["number"]).columns
        for col in numeric_cols[:3]:  # 最大3カラム
            total = self._df[col].sum()
            mean = self._df[col].mean()
            insights.append(f"{col}: 合計 {total:,.0f}, 平均 {mean:,.0f}")

        return insights


def main():
    """CLIエントリーポイント"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python -m src.insight_analyzer <データファイル>")
        print("例: python -m src.insight_analyzer data/sample_sales.csv")
        print()
        print("オプション:")
        print("  --chart  チャートも生成する")
        sys.exit(1)

    file_path = sys.argv[1]
    generate_chart = "--chart" in sys.argv

    print(f"データを読み込み中: {file_path}")
    analyzer = InsightAnalyzer(file_path)
    print(f"読み込み完了: {len(analyzer.dataframe)}行")
    print()
    print("スキーマ:")
    print(analyzer.schema)
    print()

    # 自動インサイト表示
    print("自動インサイト:")
    for insight in analyzer.get_insights():
        print(f"  - {insight}")
    print()

    print("質問を入力してください（終了: quit, チャート生成: chart）")
    print("-" * 50)

    while True:
        try:
            question = input("> ").strip()

            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            # チャート生成オプション
            with_chart = generate_chart or question.lower().startswith("chart ")
            if question.lower().startswith("chart "):
                question = question[6:].strip()

            result = analyzer.ask(question, generate_chart=with_chart)

            if result.success:
                print(result.answer)
                if result.query_used:
                    print(f"[クエリ: {result.query_used}]")
                if result.execution_time_ms > 0:
                    print(f"[実行時間: {result.execution_time_ms:.2f}ms]")
                if result.chart_path:
                    print(f"[チャート: {result.chart_path}]")
            else:
                print(f"エラー: {result.error}")

            print()

        except KeyboardInterrupt:
            print("\n終了します")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
