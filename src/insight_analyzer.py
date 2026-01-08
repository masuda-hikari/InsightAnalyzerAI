"""
InsightAnalyzerAI メインモジュール

自然言語クエリを受け付け、データ分析を実行する
Phase 2: LLM統合による高度な自然言語処理
"""

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import pandas as pd

from .data_loader import DataLoader
from .query_parser import QueryParser, ParsedQuery, QueryType
from .executor import QueryExecutor, SafeExecutor, ExecutionResult
from .visualizer import Visualizer, ChartConfig, ChartResult
from .llm_handler import LLMHandler, LLMConfig, create_llm_handler


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
    llm_explanation: Optional[str] = None  # LLMによる結果の説明
    llm_used: bool = False  # LLMが使用されたか


class InsightAnalyzer:
    """
    メイン分析クラス

    データを読み込み、自然言語クエリに基づいて分析を実行する

    アーキテクチャ:
        DataLoader -> QueryParser/LLMHandler -> QueryExecutor -> Visualizer

    Phase 2では、LLMを使用して:
    - 自然言語を高精度でPandasコードに変換
    - 分析結果を自然言語で説明
    """

    def __init__(
        self,
        data_source: Optional[Union[str, Path, pd.DataFrame]] = None,
        output_dir: Optional[str] = None,
        use_llm: bool = True,
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        Args:
            data_source: CSVパス、またはDataFrame
            output_dir: チャート出力ディレクトリ
            use_llm: LLM統合を有効にするか（デフォルト: True）
            llm_config: LLM設定（オプション）
        """
        self._loader = DataLoader()
        self._df: Optional[pd.DataFrame] = None
        self._parser: Optional[QueryParser] = None
        self._visualizer = Visualizer(output_dir)

        # LLM統合（Phase 2）
        self._use_llm = use_llm
        self._llm_handler: Optional[LLMHandler] = None
        if use_llm:
            self._llm_handler = LLMHandler(llm_config) if llm_config else create_llm_handler()

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

    @property
    def llm_available(self) -> bool:
        """LLMが利用可能か"""
        return self._llm_handler is not None and self._llm_handler.is_available

    def ask(
        self,
        question: str,
        generate_chart: bool = False,
        use_llm: Optional[bool] = None,
        explain_result: bool = False,
    ) -> AnalysisResult:
        """
        自然言語で質問し、分析結果を取得

        Args:
            question: 自然言語の質問
            generate_chart: チャートを生成するか
            use_llm: LLMを使用するか（Noneの場合はインスタンス設定に従う）
            explain_result: LLMで結果を説明するか

        Returns:
            分析結果
        """
        if self._df is None:
            return AnalysisResult(
                answer="",
                success=False,
                error="データが読み込まれていません。load_data()でデータを読み込んでください。"
            )

        # LLM使用フラグを決定
        should_use_llm = use_llm if use_llm is not None else self._use_llm
        llm_available = self._llm_handler is not None and self._llm_handler.is_available
        use_llm_for_query = should_use_llm and llm_available

        try:
            llm_used = False
            llm_explanation = None

            # Phase 2: LLMによるコード生成を試行
            if use_llm_for_query:
                result = self._ask_with_llm(question, generate_chart, explain_result)
                if result is not None:
                    return result
                # LLM失敗時はフォールバック

            # Phase 1: キーワードベースの解析（フォールバック）
            parsed = self._parser.parse(question)

            # クエリ実行
            executor = SafeExecutor(self._df)
            exec_result = executor.execute_safe(parsed)

            if not exec_result.success:
                return AnalysisResult(
                    answer="",
                    success=False,
                    error=exec_result.error,
                    execution_time_ms=exec_result.execution_time_ms,
                )

            # 回答生成
            answer = self._format_answer(parsed, exec_result)

            # LLMで結果を説明（オプション）
            if explain_result and llm_available and exec_result.data is not None:
                explain_response = self._llm_handler.explain_result(
                    question, exec_result.data, self._df
                )
                if explain_response.success:
                    llm_explanation = explain_response.explanation
                    llm_used = True

            # チャート生成（オプション）
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
                llm_explanation=llm_explanation,
                llm_used=llm_used,
            )

        except Exception as e:
            return AnalysisResult(
                answer="",
                success=False,
                error=f"分析中にエラーが発生しました: {str(e)}"
            )

    def _ask_with_llm(
        self,
        question: str,
        generate_chart: bool = False,
        explain_result: bool = False,
    ) -> Optional[AnalysisResult]:
        """
        LLMを使用して質問に回答（Phase 2）

        Args:
            question: 自然言語の質問
            generate_chart: チャートを生成するか
            explain_result: 結果を説明するか

        Returns:
            分析結果。LLM失敗時はNone（フォールバック用）
        """
        import time

        if self._llm_handler is None or not self._llm_handler.is_available:
            return None

        start_time = time.perf_counter()

        # LLMでPandasコードを生成
        llm_response = self._llm_handler.generate_code(question, self._df)

        if not llm_response.success or not llm_response.pandas_code:
            return None

        # 生成されたコードを安全に実行
        executor = SafeExecutor(self._df)

        # コードの安全性検証
        if not executor.validate_code(llm_response.pandas_code):
            return None

        try:
            # サンドボックス実行
            local_vars = {"df": self._df.copy(), "pd": pd}
            exec(llm_response.pandas_code, {"__builtins__": {}}, local_vars)

            # 結果を取得
            result = local_vars.get("result")

            execution_time = (time.perf_counter() - start_time) * 1000

            # 結果をDataFrameに変換
            result_data = None
            result_value = None

            if isinstance(result, pd.DataFrame):
                result_data = result
            elif isinstance(result, pd.Series):
                result_data = result.to_frame()
            elif isinstance(result, (int, float)):
                result_value = result
                result_data = pd.DataFrame({"result": [result]})
            else:
                result_value = result

            # 回答を生成
            answer = self._format_llm_result(question, result)

            # LLMで結果を説明（オプション）
            llm_explanation = None
            if explain_result and result is not None:
                explain_response = self._llm_handler.explain_result(
                    question, result, self._df
                )
                if explain_response.success:
                    llm_explanation = explain_response.explanation

            # チャート生成（オプション）
            chart_path = None
            if generate_chart and result_data is not None:
                parsed = ParsedQuery(
                    query_type=QueryType.UNKNOWN,
                    original_question=question
                )
                chart_result = self._generate_chart(parsed, result_data)
                if chart_result.success:
                    chart_path = chart_result.file_path

            return AnalysisResult(
                answer=answer,
                data=result_data,
                chart_path=chart_path,
                query_used=llm_response.pandas_code,
                success=True,
                execution_time_ms=execution_time,
                confidence=0.9,  # LLM使用時は高信頼度
                llm_explanation=llm_explanation,
                llm_used=True,
            )

        except Exception as e:
            # 実行エラー時はNoneを返してフォールバック
            return None

    def _format_llm_result(self, question: str, result) -> str:
        """LLM生成コードの実行結果をフォーマット"""
        if result is None:
            return "結果を取得できませんでした"

        if isinstance(result, pd.DataFrame):
            if len(result) <= 10:
                return f"分析結果:\n{result.to_string()}"
            else:
                return f"分析結果（上位10件）:\n{result.head(10).to_string()}\n...（全{len(result)}件）"

        if isinstance(result, pd.Series):
            if len(result) <= 10:
                return f"分析結果:\n{result.to_string()}"
            else:
                return f"分析結果（上位10件）:\n{result.head(10).to_string()}\n...（全{len(result)}件）"

        if isinstance(result, (int, float)):
            # 数値の場合、金額かどうかを推測
            if abs(result) >= 1000:
                return f"結果: ¥{result:,.0f}"
            return f"結果: {result:,.2f}"

        return f"結果: {result}"

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
                # 数値型の場合のみフォーマット
                if isinstance(val, (int, float)):
                    if is_currency:
                        lines.append(f"  {idx}: ¥{val:,.0f}")
                    else:
                        lines.append(f"  {idx}: {val:,.2f}")
                else:
                    lines.append(f"  {idx}: {val}")
        else:
            for idx, val in data.items():
                # 数値型の場合のみフォーマット
                if isinstance(val, (int, float)):
                    if is_currency:
                        lines.append(f"  {idx}: ¥{val:,.0f}")
                    else:
                        lines.append(f"  {idx}: {val:,.2f}")
                else:
                    lines.append(f"  {idx}: {val}")

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
        print("  --chart    チャートも生成する")
        print("  --no-llm   LLM統合を無効化")
        print("  --explain  LLMで結果を説明")
        sys.exit(1)

    file_path = sys.argv[1]
    generate_chart = "--chart" in sys.argv
    use_llm = "--no-llm" not in sys.argv
    explain_result = "--explain" in sys.argv

    print(f"データを読み込み中: {file_path}")
    analyzer = InsightAnalyzer(file_path, use_llm=use_llm)
    print(f"読み込み完了: {len(analyzer.dataframe)}行")

    # LLM状態を表示
    if use_llm:
        if analyzer.llm_available:
            print("LLM統合: 有効（OpenAI API）")
        else:
            print("LLM統合: APIキーなし（フォールバックモード）")
    else:
        print("LLM統合: 無効")
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

            result = analyzer.ask(
                question,
                generate_chart=with_chart,
                explain_result=explain_result,
            )

            if result.success:
                print(result.answer)

                # LLM説明がある場合は表示
                if result.llm_explanation:
                    print()
                    print("【AIによる解説】")
                    print(result.llm_explanation)

                # メタ情報
                meta_info = []
                if result.llm_used:
                    meta_info.append("LLM使用")
                if result.query_used:
                    meta_info.append(f"クエリ: {result.query_used}")
                if result.execution_time_ms > 0:
                    meta_info.append(f"実行時間: {result.execution_time_ms:.2f}ms")
                if result.chart_path:
                    meta_info.append(f"チャート: {result.chart_path}")

                if meta_info:
                    print(f"[{' | '.join(meta_info)}]")
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
