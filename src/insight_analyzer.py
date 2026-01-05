"""
InsightAnalyzerAI メインモジュール

自然言語クエリを受け付け、データ分析を実行する
"""

import os
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import dataclass

import pandas as pd

from .data_loader import DataLoader


@dataclass
class AnalysisResult:
    """分析結果を格納するクラス"""
    answer: str
    data: Optional[pd.DataFrame] = None
    chart_path: Optional[str] = None
    query_used: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class InsightAnalyzer:
    """
    メイン分析クラス

    データを読み込み、自然言語クエリに基づいて分析を実行する
    """

    def __init__(self, data_source: Optional[Union[str, Path, pd.DataFrame]] = None):
        """
        Args:
            data_source: CSVパス、またはDataFrame
        """
        self._loader = DataLoader()
        self._df: Optional[pd.DataFrame] = None

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
        else:
            self._df = self._loader.load(source)

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """現在のDataFrameを取得"""
        return self._df

    @property
    def schema(self) -> str:
        """データスキーマを取得"""
        return self._loader.get_schema()

    def ask(self, question: str) -> AnalysisResult:
        """
        自然言語で質問し、分析結果を取得

        Args:
            question: 自然言語の質問

        Returns:
            分析結果
        """
        if self._df is None:
            return AnalysisResult(
                answer="",
                success=False,
                error="データが読み込まれていません。load_data()でデータを読み込んでください。"
            )

        # Phase 1: 簡易的なキーワードベース処理
        # TODO: Phase 2でLLM統合に置き換え

        question_lower = question.lower()

        try:
            # 合計を求める
            if "合計" in question or "total" in question_lower or "sum" in question_lower:
                result = self._calculate_sum(question)
                return result

            # 平均を求める
            if "平均" in question or "average" in question_lower or "mean" in question_lower:
                result = self._calculate_mean(question)
                return result

            # 件数を数える
            if "件数" in question or "count" in question_lower or "何件" in question:
                result = self._calculate_count(question)
                return result

            # グループ別集計
            if "別" in question or "ごと" in question or "by" in question_lower:
                result = self._calculate_groupby(question)
                return result

            # デフォルト: 基本統計
            stats = self._df.describe()
            return AnalysisResult(
                answer=f"データの基本統計:\n{stats.to_string()}",
                data=stats,
                query_used="df.describe()"
            )

        except Exception as e:
            return AnalysisResult(
                answer="",
                success=False,
                error=f"分析中にエラーが発生しました: {str(e)}"
            )

    def _find_numeric_column(self, question: str) -> Optional[str]:
        """質問から数値カラムを推定"""
        numeric_cols = self._df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            if col.lower() in question.lower() or col in question:
                return col

        # 見つからない場合は最初の数値カラム
        return numeric_cols[0] if len(numeric_cols) > 0 else None

    def _find_category_column(self, question: str) -> Optional[str]:
        """質問からカテゴリカラムを推定"""
        # 「地域別」「製品別」などのパターンを検出
        keywords = {
            "地域": "region",
            "製品": "product",
            "商品": "product",
            "担当": "salesperson",
            "月": "date",
        }

        for jp_key, en_col in keywords.items():
            if jp_key in question:
                for col in self._df.columns:
                    if en_col in col.lower() or jp_key in col:
                        return col

        # カテゴリ型カラムを返す
        cat_cols = self._df.select_dtypes(include=['object']).columns
        return cat_cols[0] if len(cat_cols) > 0 else None

    def _calculate_sum(self, question: str) -> AnalysisResult:
        """合計を計算"""
        col = self._find_numeric_column(question)
        if col is None:
            return AnalysisResult(
                answer="数値カラムが見つかりません",
                success=False,
                error="数値カラムが見つかりません"
            )

        total = self._df[col].sum()
        query = f"df['{col}'].sum()"

        # 金額の場合はフォーマット
        if "price" in col.lower() or "sales" in col.lower() or "金額" in col:
            answer = f"{col}の合計: ¥{total:,.0f}"
        else:
            answer = f"{col}の合計: {total:,.2f}"

        return AnalysisResult(
            answer=answer,
            data=pd.DataFrame({col: [total]}, index=['合計']),
            query_used=query
        )

    def _calculate_mean(self, question: str) -> AnalysisResult:
        """平均を計算"""
        col = self._find_numeric_column(question)
        if col is None:
            return AnalysisResult(
                answer="数値カラムが見つかりません",
                success=False,
                error="数値カラムが見つかりません"
            )

        mean = self._df[col].mean()
        query = f"df['{col}'].mean()"

        if "price" in col.lower() or "sales" in col.lower() or "金額" in col:
            answer = f"{col}の平均: ¥{mean:,.0f}"
        else:
            answer = f"{col}の平均: {mean:,.2f}"

        return AnalysisResult(
            answer=answer,
            data=pd.DataFrame({col: [mean]}, index=['平均']),
            query_used=query
        )

    def _calculate_count(self, question: str) -> AnalysisResult:
        """件数を計算"""
        count = len(self._df)

        return AnalysisResult(
            answer=f"データ件数: {count:,}件",
            data=pd.DataFrame({'count': [count]}, index=['件数']),
            query_used="len(df)"
        )

    def _calculate_groupby(self, question: str) -> AnalysisResult:
        """グループ別集計"""
        cat_col = self._find_category_column(question)
        num_col = self._find_numeric_column(question)

        if cat_col is None:
            return AnalysisResult(
                answer="グループ化するカラムが見つかりません",
                success=False,
                error="カテゴリカラムが見つかりません"
            )

        if num_col is None:
            # 件数でグループ化
            grouped = self._df.groupby(cat_col).size().sort_values(ascending=False)
            query = f"df.groupby('{cat_col}').size()"
            answer = f"{cat_col}別の件数:\n{grouped.to_string()}"
        else:
            # 合計でグループ化
            grouped = self._df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
            query = f"df.groupby('{cat_col}')['{num_col}'].sum()"

            lines = [f"{cat_col}別の{num_col}合計:"]
            for idx, val in grouped.items():
                if "price" in num_col.lower() or "sales" in num_col.lower():
                    lines.append(f"  {idx}: ¥{val:,.0f}")
                else:
                    lines.append(f"  {idx}: {val:,.2f}")
            answer = "\n".join(lines)

        return AnalysisResult(
            answer=answer,
            data=grouped.to_frame(),
            query_used=query
        )


def main():
    """CLIエントリーポイント"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python -m src.insight_analyzer <データファイル>")
        print("例: python -m src.insight_analyzer data/sample_sales.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"データを読み込み中: {file_path}")
    analyzer = InsightAnalyzer(file_path)
    print(f"読み込み完了: {len(analyzer.dataframe)}行")
    print()
    print("スキーマ:")
    print(analyzer.schema)
    print()
    print("質問を入力してください（終了: quit）")
    print("-" * 50)

    while True:
        try:
            question = input("> ").strip()
            if question.lower() in ('quit', 'exit', 'q'):
                break
            if not question:
                continue

            result = analyzer.ask(question)

            if result.success:
                print(result.answer)
                if result.query_used:
                    print(f"[実行クエリ: {result.query_used}]")
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
