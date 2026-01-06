"""Executorのテスト"""

from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.executor import QueryExecutor, SafeExecutor, ExecutionResult
from src.query_parser import ParsedQuery, QueryType


class TestQueryExecutor:
    """QueryExecutorクラスのテスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """テスト用DataFrame"""
        return pd.DataFrame({
            "region": ["東京", "大阪", "東京", "大阪"],
            "sales": [100, 200, 150, 250],
            "quantity": [10, 20, 15, 25],
        })

    @pytest.fixture
    def executor(self, sample_df: pd.DataFrame) -> QueryExecutor:
        """Executorインスタンス"""
        return QueryExecutor(sample_df)

    def test_execute_sum(self, executor: QueryExecutor):
        """合計実行"""
        query = ParsedQuery(
            query_type=QueryType.SUM,
            target_column="sales",
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.value == 700  # 100+200+150+250
        assert "sum()" in result.query_code

    def test_execute_mean(self, executor: QueryExecutor):
        """平均実行"""
        query = ParsedQuery(
            query_type=QueryType.MEAN,
            target_column="quantity",
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.value == 17.5  # (10+20+15+25)/4
        assert "mean()" in result.query_code

    def test_execute_count(self, executor: QueryExecutor):
        """件数実行"""
        query = ParsedQuery(query_type=QueryType.COUNT)
        result = executor.execute(query)

        assert result.success is True
        assert result.value == 4

    def test_execute_groupby(self, executor: QueryExecutor):
        """グループ別集計実行"""
        query = ParsedQuery(
            query_type=QueryType.GROUPBY,
            target_column="sales",
            group_column="region",
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.data is not None
        # 東京: 100+150=250, 大阪: 200+250=450
        assert len(result.data) == 2

    def test_execute_describe(self, executor: QueryExecutor):
        """基本統計実行"""
        query = ParsedQuery(query_type=QueryType.DESCRIBE)
        result = executor.execute(query)

        assert result.success is True
        assert result.data is not None
        assert "mean" in result.data.index

    def test_execution_time_recorded(self, executor: QueryExecutor):
        """実行時間が記録される"""
        query = ParsedQuery(query_type=QueryType.COUNT)
        result = executor.execute(query)

        assert result.execution_time_ms >= 0

    def test_error_handling(self):
        """エラーハンドリング"""
        # 空のDataFrame
        executor = QueryExecutor(pd.DataFrame())
        query = ParsedQuery(
            query_type=QueryType.SUM,
            target_column="nonexistent",
        )
        result = executor.execute(query)

        assert result.success is False
        assert result.error is not None


class TestSafeExecutor:
    """SafeExecutorクラスのテスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })

    def test_validate_safe_code(self, sample_df: pd.DataFrame):
        """安全なコードの検証"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("df['a'].sum()") is True
        assert executor.validate_code("df.groupby('a').mean()") is True

    def test_validate_blocked_code(self, sample_df: pd.DataFrame):
        """危険なコードのブロック"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("eval('code')") is False
        assert executor.validate_code("exec('code')") is False
        assert executor.validate_code("df.__class__") is False

    def test_execution_log(self, sample_df: pd.DataFrame):
        """実行ログの記録"""
        executor = SafeExecutor(sample_df)

        query = ParsedQuery(
            query_type=QueryType.SUM,
            original_question="合計",
        )
        executor.execute_safe(query)

        assert len(executor.execution_log) == 1
        assert "query" in executor.execution_log[0]
        assert "success" in executor.execution_log[0]
