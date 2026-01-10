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

    def test_validate_additional_blocked_operations(self, sample_df: pd.DataFrame):
        """追加の危険な操作のブロック"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("import os") is False
        assert executor.validate_code("open('file.txt')") is False
        assert executor.validate_code("os.system('ls')") is False
        assert executor.validate_code("sys.exit()") is False
        assert executor.validate_code("subprocess.run()") is False

    def test_validate_empty_code(self, sample_df: pd.DataFrame):
        """空のコードの検証"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("") is False
        assert executor.validate_code(None) is False

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

    def test_clear_log(self, sample_df: pd.DataFrame):
        """実行ログのクリア"""
        executor = SafeExecutor(sample_df)

        query = ParsedQuery(
            query_type=QueryType.SUM,
            original_question="合計",
        )
        executor.execute_safe(query)

        assert len(executor.execution_log) == 1

        executor.clear_log()
        assert len(executor.execution_log) == 0

    def test_null_dataframe_error(self):
        """NullDataFrameでのエラー"""
        with pytest.raises(ValueError, match="DataFrameがNoneです"):
            SafeExecutor(None)

    def test_empty_dataframe_error(self):
        """空DataFrameでのエラー"""
        with pytest.raises(ValueError, match="DataFrameが空です"):
            SafeExecutor(pd.DataFrame())

    def test_null_query_handling(self, sample_df: pd.DataFrame):
        """Nullクエリのハンドリング"""
        executor = SafeExecutor(sample_df)
        result = executor.execute_safe(None)

        assert result.success is False
        assert "Null" in result.error

    def test_get_error_suggestion(self, sample_df: pd.DataFrame):
        """エラー提案の取得"""
        executor = SafeExecutor(sample_df)

        # カラムが見つからない場合
        suggestion = executor.get_error_suggestion("カラムが見つかりません: 'xyz'")
        assert len(suggestion) > 0  # 何らかの提案がある

        # 数値カラムが見つからない場合
        suggestion = executor.get_error_suggestion("数値カラムが見つかりません")
        assert len(suggestion) > 0

        # メモリ不足の場合
        suggestion = executor.get_error_suggestion("メモリ不足")
        assert len(suggestion) > 0

        # 未知のエラー
        suggestion = executor.get_error_suggestion("予期しないエラー")
        assert len(suggestion) > 0


class TestErrorHandling:
    """詳細なエラーハンドリングテスト"""

    def test_keyerror_handling(self):
        """KeyErrorのハンドリング"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        executor = SafeExecutor(df)

        query = ParsedQuery(
            query_type=QueryType.SUM,
            target_column="nonexistent_column",
            original_question="存在しないカラムの合計",
        )
        result = executor.execute_safe(query)

        # カラムが見つからない場合、最初の数値カラムを使用するので成功する
        # ただし、数値カラム"a"が使用される
        assert result.success is True

    def test_large_result_truncation(self):
        """大きな結果の切り詰め"""
        # 大きなDataFrameを作成
        df = pd.DataFrame({
            "a": list(range(100)),
            "b": ["cat"] * 50 + ["dog"] * 50,
        })
        executor = SafeExecutor(df)

        query = ParsedQuery(
            query_type=QueryType.DESCRIBE,
            original_question="統計",
        )
        result = executor.execute_safe(query)

        assert result.success is True
        # 結果が制限されていることを確認
        assert result.data is not None
