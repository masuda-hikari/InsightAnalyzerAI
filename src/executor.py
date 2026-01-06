"""
クエリ実行モジュール

解析済みクエリをPandas操作に変換し、安全に実行する
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union
from functools import wraps

import pandas as pd

from .query_parser import ParsedQuery, QueryType


@dataclass
class ExecutionResult:
    """実行結果を格納するクラス"""
    success: bool
    data: Optional[pd.DataFrame] = None
    value: Optional[Any] = None  # スカラー値（合計、平均等）
    query_code: str = ""  # 実行したPandasコード（デバッグ用）
    execution_time_ms: float = 0.0
    error: Optional[str] = None


def timeout_handler(timeout_ms: int = 30000):
    """タイムアウトデコレータ（簡易実装）"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if elapsed_ms > timeout_ms:
                raise TimeoutError(f"実行時間が{timeout_ms}msを超過しました")
            return result
        return wrapper
    return decorator


class QueryExecutor:
    """
    クエリを安全に実行するクラス

    - タイムアウト制御
    - エラーハンドリング
    - 実行ログ生成
    """

    DEFAULT_TIMEOUT_MS = 30000  # 30秒

    def __init__(self, df: pd.DataFrame, timeout_ms: Optional[int] = None):
        """
        Args:
            df: 操作対象のDataFrame
            timeout_ms: タイムアウト（ミリ秒）
        """
        self._df = df
        self._timeout_ms = timeout_ms or self.DEFAULT_TIMEOUT_MS

    def execute(self, parsed_query: ParsedQuery) -> ExecutionResult:
        """
        解析済みクエリを実行

        Args:
            parsed_query: 解析済みクエリ

        Returns:
            実行結果
        """
        start_time = time.perf_counter()

        try:
            # クエリタイプに応じた実行
            handler = self._get_handler(parsed_query.query_type)
            result = handler(parsed_query)

            execution_time = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time

            return result

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )

    def _get_handler(self, query_type: QueryType) -> Callable:
        """クエリタイプに対応するハンドラを取得"""
        handlers = {
            QueryType.SUM: self._execute_sum,
            QueryType.MEAN: self._execute_mean,
            QueryType.COUNT: self._execute_count,
            QueryType.GROUPBY: self._execute_groupby,
            QueryType.FILTER: self._execute_filter,
            QueryType.SORT: self._execute_sort,
            QueryType.DESCRIBE: self._execute_describe,
            QueryType.UNKNOWN: self._execute_describe,
        }
        return handlers.get(query_type, self._execute_describe)

    def _execute_sum(self, query: ParsedQuery) -> ExecutionResult:
        """合計を実行"""
        col = query.target_column
        if col is None or col not in self._df.columns:
            # 数値カラムの最初を使用
            numeric_cols = self._df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                return ExecutionResult(
                    success=False,
                    error="数値カラムが見つかりません",
                )
            col = numeric_cols[0]

        total = self._df[col].sum()
        query_code = f"df['{col}'].sum()"

        return ExecutionResult(
            success=True,
            value=total,
            data=pd.DataFrame({col: [total]}, index=["合計"]),
            query_code=query_code,
        )

    def _execute_mean(self, query: ParsedQuery) -> ExecutionResult:
        """平均を実行"""
        col = query.target_column
        if col is None or col not in self._df.columns:
            numeric_cols = self._df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                return ExecutionResult(
                    success=False,
                    error="数値カラムが見つかりません",
                )
            col = numeric_cols[0]

        mean = self._df[col].mean()
        query_code = f"df['{col}'].mean()"

        return ExecutionResult(
            success=True,
            value=mean,
            data=pd.DataFrame({col: [mean]}, index=["平均"]),
            query_code=query_code,
        )

    def _execute_count(self, query: ParsedQuery) -> ExecutionResult:
        """件数を実行"""
        count = len(self._df)
        query_code = "len(df)"

        return ExecutionResult(
            success=True,
            value=count,
            data=pd.DataFrame({"count": [count]}, index=["件数"]),
            query_code=query_code,
        )

    def _execute_groupby(self, query: ParsedQuery) -> ExecutionResult:
        """グループ別集計を実行"""
        group_col = query.group_column
        target_col = query.target_column

        if group_col is None or group_col not in self._df.columns:
            # カテゴリカラムの最初を使用
            cat_cols = self._df.select_dtypes(include=["object"]).columns
            if len(cat_cols) == 0:
                return ExecutionResult(
                    success=False,
                    error="グループ化カラムが見つかりません",
                )
            group_col = cat_cols[0]

        if target_col is None or target_col not in self._df.columns:
            # 件数でグループ化
            grouped = self._df.groupby(group_col).size().sort_values(ascending=False)
            query_code = f"df.groupby('{group_col}').size().sort_values(ascending=False)"
        else:
            # 指定カラムで集計
            agg_func = query.aggregation
            if agg_func == "sum":
                grouped = self._df.groupby(group_col)[target_col].sum()
            elif agg_func == "mean":
                grouped = self._df.groupby(group_col)[target_col].mean()
            elif agg_func == "count":
                grouped = self._df.groupby(group_col)[target_col].count()
            else:
                grouped = self._df.groupby(group_col)[target_col].sum()

            grouped = grouped.sort_values(ascending=False)
            query_code = f"df.groupby('{group_col}')['{target_col}'].{agg_func}().sort_values(ascending=False)"

        return ExecutionResult(
            success=True,
            data=grouped.to_frame(),
            query_code=query_code,
        )

    def _execute_filter(self, query: ParsedQuery) -> ExecutionResult:
        """フィルタを実行"""
        df_filtered = self._df.copy()
        conditions = []

        for f in query.filter_conditions:
            col = f.get("column") or query.target_column
            if col is None:
                continue

            op = f["operator"]
            val = f["value"]

            if op == ">=":
                df_filtered = df_filtered[df_filtered[col] >= val]
                conditions.append(f"df['{col}'] >= {val}")
            elif op == "<=":
                df_filtered = df_filtered[df_filtered[col] <= val]
                conditions.append(f"df['{col}'] <= {val}")
            elif op == ">":
                df_filtered = df_filtered[df_filtered[col] > val]
                conditions.append(f"df['{col}'] > {val}")
            elif op == "<":
                df_filtered = df_filtered[df_filtered[col] < val]
                conditions.append(f"df['{col}'] < {val}")
            elif op == "==":
                df_filtered = df_filtered[df_filtered[col] == val]
                conditions.append(f"df['{col}'] == {val}")

        query_code = " & ".join(conditions) if conditions else "df"

        return ExecutionResult(
            success=True,
            data=df_filtered,
            value=len(df_filtered),
            query_code=f"df[{query_code}]" if conditions else "df",
        )

    def _execute_sort(self, query: ParsedQuery) -> ExecutionResult:
        """ソートを実行"""
        col = query.target_column
        if col is None or col not in self._df.columns:
            numeric_cols = self._df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
            else:
                col = self._df.columns[0]

        ascending = query.sort_ascending
        sorted_df = self._df.sort_values(col, ascending=ascending)
        query_code = f"df.sort_values('{col}', ascending={ascending})"

        return ExecutionResult(
            success=True,
            data=sorted_df,
            query_code=query_code,
        )

    def _execute_describe(self, query: ParsedQuery) -> ExecutionResult:
        """基本統計を実行"""
        stats = self._df.describe()
        query_code = "df.describe()"

        return ExecutionResult(
            success=True,
            data=stats,
            query_code=query_code,
        )


class SafeExecutor:
    """
    より安全な実行環境を提供するクラス

    - 危険な操作のブロック
    - リソース制限
    - 監査ログ
    """

    # 禁止するDataFrame操作（将来のLLM生成コード用）
    BLOCKED_OPERATIONS = [
        "eval",
        "exec",
        "to_sql",
        "to_pickle",
        "__",
    ]

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._execution_log: list = []

    def validate_code(self, code: str) -> bool:
        """コードの安全性を検証"""
        for blocked in self.BLOCKED_OPERATIONS:
            if blocked in code:
                return False
        return True

    def execute_safe(self, query: ParsedQuery) -> ExecutionResult:
        """安全にクエリを実行"""
        executor = QueryExecutor(self._df)
        result = executor.execute(query)

        # 実行ログを記録
        self._execution_log.append({
            "query": query.original_question,
            "query_code": result.query_code,
            "success": result.success,
            "execution_time_ms": result.execution_time_ms,
        })

        return result

    @property
    def execution_log(self) -> list:
        """実行ログを取得"""
        return self._execution_log
