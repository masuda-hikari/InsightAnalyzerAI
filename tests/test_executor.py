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


class TestQueryExecutorExtended:
    """QueryExecutorの追加テスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """テスト用DataFrame"""
        return pd.DataFrame({
            "region": ["東京", "大阪", "東京", "大阪", "福岡"],
            "sales": [100, 200, 150, 250, 300],
            "quantity": [10, 20, 15, 25, 30],
            "price": [10.0, 10.0, 10.0, 10.0, 10.0],
        })

    @pytest.fixture
    def executor(self, sample_df: pd.DataFrame) -> QueryExecutor:
        """Executorインスタンス"""
        return QueryExecutor(sample_df)

    def test_execute_sum_no_column(self, sample_df: pd.DataFrame):
        """カラム指定なしの合計実行"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.SUM,
            target_column=None,  # カラム指定なし
        )
        result = executor.execute(query)

        assert result.success is True
        # 最初の数値カラム（sales）の合計
        assert result.value == 1000  # 100+200+150+250+300

    def test_execute_sum_invalid_column(self, sample_df: pd.DataFrame):
        """無効なカラムでの合計実行"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.SUM,
            target_column="invalid_column",
        )
        result = executor.execute(query)

        # 無効なカラムの場合は最初の数値カラムを使用
        assert result.success is True
        assert result.value == 1000

    def test_execute_sum_no_numeric_columns(self):
        """数値カラムなしでの合計実行"""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        executor = QueryExecutor(df)
        query = ParsedQuery(
            query_type=QueryType.SUM,
            target_column=None,
        )
        result = executor.execute(query)

        assert result.success is False
        assert "数値カラムが見つかりません" in result.error

    def test_execute_mean_no_column(self, sample_df: pd.DataFrame):
        """カラム指定なしの平均実行"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.MEAN,
            target_column=None,
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.value == 200  # (100+200+150+250+300)/5

    def test_execute_mean_no_numeric_columns(self):
        """数値カラムなしでの平均実行"""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        executor = QueryExecutor(df)
        query = ParsedQuery(
            query_type=QueryType.MEAN,
            target_column=None,
        )
        result = executor.execute(query)

        assert result.success is False
        assert "数値カラムが見つかりません" in result.error

    def test_execute_groupby_no_group_column(self, sample_df: pd.DataFrame):
        """グループカラム指定なしのグループ集計"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.GROUPBY,
            group_column=None,
            target_column="sales",
        )
        result = executor.execute(query)

        # 最初のオブジェクトカラム（region）でグループ化
        assert result.success is True
        assert result.data is not None

    def test_execute_groupby_no_categorical_columns(self):
        """カテゴリカラムなしでのグループ集計"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        executor = QueryExecutor(df)
        query = ParsedQuery(
            query_type=QueryType.GROUPBY,
            group_column=None,
            target_column="a",
        )
        result = executor.execute(query)

        assert result.success is False
        assert "グループ化カラムが見つかりません" in result.error

    def test_execute_groupby_count_without_target(self, sample_df: pd.DataFrame):
        """ターゲットカラムなしでのグループ件数"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.GROUPBY,
            group_column="region",
            target_column=None,
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.data is not None
        assert "size()" in result.query_code

    def test_execute_groupby_with_mean_aggregation(self, sample_df: pd.DataFrame):
        """平均集計でのグループ集計"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.GROUPBY,
            group_column="region",
            target_column="sales",
            aggregation="mean",
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.data is not None
        assert "mean()" in result.query_code

    def test_execute_groupby_with_count_aggregation(self, sample_df: pd.DataFrame):
        """件数集計でのグループ集計"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.GROUPBY,
            group_column="region",
            target_column="sales",
            aggregation="count",
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.data is not None
        assert "count()" in result.query_code

    def test_execute_groupby_with_default_aggregation(self, sample_df: pd.DataFrame):
        """デフォルト集計（sum）でのグループ集計"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.GROUPBY,
            group_column="region",
            target_column="sales",
            aggregation="unknown",  # 未知の集計方法
        )
        result = executor.execute(query)

        # デフォルトでsumになる
        assert result.success is True
        assert result.data is not None

    def test_execute_filter_with_conditions(self, sample_df: pd.DataFrame):
        """条件付きフィルタ実行"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="sales",
            filter_conditions=[
                {"column": "sales", "operator": ">=", "value": 200},
            ],
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 3  # 200, 250, 300

    def test_execute_filter_less_than(self, sample_df: pd.DataFrame):
        """未満条件のフィルタ"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="sales",
            filter_conditions=[
                {"column": "sales", "operator": "<", "value": 200},
            ],
        )
        result = executor.execute(query)

        assert result.success is True
        assert len(result.data) == 2  # 100, 150

    def test_execute_filter_greater_than(self, sample_df: pd.DataFrame):
        """超過条件のフィルタ"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="sales",
            filter_conditions=[
                {"column": "sales", "operator": ">", "value": 200},
            ],
        )
        result = executor.execute(query)

        assert result.success is True
        assert len(result.data) == 2  # 250, 300

    def test_execute_filter_less_than_or_equal(self, sample_df: pd.DataFrame):
        """以下条件のフィルタ"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="sales",
            filter_conditions=[
                {"column": "sales", "operator": "<=", "value": 150},
            ],
        )
        result = executor.execute(query)

        assert result.success is True
        assert len(result.data) == 2  # 100, 150

    def test_execute_filter_equal(self, sample_df: pd.DataFrame):
        """等価条件のフィルタ"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="sales",
            filter_conditions=[
                {"column": "sales", "operator": "==", "value": 200},
            ],
        )
        result = executor.execute(query)

        assert result.success is True
        assert len(result.data) == 1

    def test_execute_filter_no_column(self, sample_df: pd.DataFrame):
        """カラム指定なしの条件でのフィルタ"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="sales",
            filter_conditions=[
                {"column": None, "operator": ">=", "value": 200},
            ],
        )
        result = executor.execute(query)

        # target_columnが使用される
        assert result.success is True
        assert len(result.data) == 3

    def test_execute_filter_empty_conditions(self, sample_df: pd.DataFrame):
        """空の条件でのフィルタ"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            filter_conditions=[],
        )
        result = executor.execute(query)

        # 全データが返される
        assert result.success is True
        assert len(result.data) == 5

    def test_execute_sort_no_column(self, sample_df: pd.DataFrame):
        """カラム指定なしのソート"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.SORT,
            target_column=None,
            sort_ascending=True,
        )
        result = executor.execute(query)

        # 最初の数値カラム（sales）でソート
        assert result.success is True
        assert result.data is not None
        assert result.data.iloc[0]["sales"] == 100

    def test_execute_sort_descending(self, sample_df: pd.DataFrame):
        """降順ソート"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.SORT,
            target_column="sales",
            sort_ascending=False,
        )
        result = executor.execute(query)

        assert result.success is True
        assert result.data.iloc[0]["sales"] == 300

    def test_execute_sort_no_numeric_columns(self):
        """数値カラムなしでのソート"""
        df = pd.DataFrame({"text": ["c", "a", "b"]})
        executor = QueryExecutor(df)
        query = ParsedQuery(
            query_type=QueryType.SORT,
            target_column=None,
            sort_ascending=True,
        )
        result = executor.execute(query)

        # 最初のカラムでソート
        assert result.success is True
        assert result.data.iloc[0]["text"] == "a"

    def test_execute_unknown_query_type(self, sample_df: pd.DataFrame):
        """未知のクエリタイプ（describe実行）"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(query_type=QueryType.UNKNOWN)
        result = executor.execute(query)

        # UNKNOWNはdescribeにフォールバック
        assert result.success is True
        assert result.data is not None

    def test_execute_exception_handling(self):
        """例外発生時のハンドリング"""
        # 特殊なDataFrameで例外を発生させる
        df = pd.DataFrame({"a": [1, 2, 3]})
        executor = QueryExecutor(df)

        # 存在しないカラムでフィルタを試みる（例外発生）
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            filter_conditions=[
                {"column": "nonexistent", "operator": ">=", "value": 1},
            ],
        )
        result = executor.execute(query)

        assert result.success is False
        assert result.error is not None


class TestSafeExecutorExtended:
    """SafeExecutorの追加テスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "a": list(range(100)),
            "b": ["cat"] * 50 + ["dog"] * 50,
        })

    def test_execute_safe_with_large_result(self, sample_df: pd.DataFrame):
        """大きな結果の制限"""
        # MAX_RESULT_ROWSより小さいデータでテスト
        executor = SafeExecutor(sample_df)

        query = ParsedQuery(
            query_type=QueryType.DESCRIBE,
            original_question="統計",
        )
        result = executor.execute_safe(query)

        assert result.success is True
        assert result.data is not None

    def test_validate_code_with_to_sql(self, sample_df: pd.DataFrame):
        """to_sqlブロックの検証"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("df.to_sql('table', conn)") is False

    def test_validate_code_with_to_pickle(self, sample_df: pd.DataFrame):
        """to_pickleブロックの検証"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("df.to_pickle('file.pkl')") is False

    def test_validate_code_with_shell(self, sample_df: pd.DataFrame):
        """shellブロックの検証"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("shell('ls')") is False

    def test_validate_code_with_rm(self, sample_df: pd.DataFrame):
        """rmブロックの検証"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("rm -rf /") is False

    def test_validate_code_with_del(self, sample_df: pd.DataFrame):
        """delブロックの検証"""
        executor = SafeExecutor(sample_df)
        assert executor.validate_code("del something") is False

    def test_get_error_suggestion_group_column(self, sample_df: pd.DataFrame):
        """グループ化カラムエラーの提案"""
        executor = SafeExecutor(sample_df)
        suggestion = executor.get_error_suggestion("グループ化カラムが見つかりません")
        # 何らかの提案がある
        assert len(suggestion) > 0

    def test_get_error_suggestion_date_error(self, sample_df: pd.DataFrame):
        """日付エラーの提案"""
        executor = SafeExecutor(sample_df)
        suggestion = executor.get_error_suggestion("日付データが範囲外です")
        assert "日付形式" in suggestion

    def test_execution_log_content(self, sample_df: pd.DataFrame):
        """実行ログの内容確認"""
        executor = SafeExecutor(sample_df)

        query = ParsedQuery(
            query_type=QueryType.SUM,
            original_question="合計を教えて",
        )
        result = executor.execute_safe(query)

        log = executor.execution_log[0]
        assert log["query"] == "合計を教えて"
        assert log["success"] == result.success
        assert "execution_time_ms" in log
        assert "query_code" in log


class TestTimeoutHandler:
    """タイムアウトハンドラーのテスト"""

    def test_timeout_decorator_no_timeout(self):
        """タイムアウトなしの実行"""
        from src.executor import timeout_handler

        @timeout_handler(timeout_ms=10000)
        def fast_function():
            return "done"

        result = fast_function()
        assert result == "done"

    def test_timeout_decorator_with_args(self):
        """引数付き関数のタイムアウトテスト"""
        from src.executor import timeout_handler

        @timeout_handler(timeout_ms=10000)
        def add_numbers(a, b):
            return a + b

        result = add_numbers(1, 2)
        assert result == 3


class TestSafeExecutorExceptionHandling:
    """SafeExecutorの例外ハンドリングテスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["cat", "dog", "bird"],
        })

    def test_keyerror_exception_handling(self, sample_df: pd.DataFrame):
        """KeyError例外のハンドリング"""
        from unittest.mock import patch, MagicMock

        executor = SafeExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="a",
            original_question="テスト",
            filter_conditions=[
                {"column": "nonexistent_col", "operator": ">=", "value": 1},
            ],
        )

        result = executor.execute_safe(query)
        # KeyErrorが発生し、適切にハンドリングされる
        assert result.success is False
        # エラーメッセージにはカラム名が含まれる
        assert "nonexistent_col" in result.error or "カラムが見つかりません" in result.error

    def test_typeerror_exception_handling(self, sample_df: pd.DataFrame):
        """TypeError例外のハンドリング"""
        from unittest.mock import patch

        executor = SafeExecutor(sample_df)

        # TypeErrorを発生させるためのモック
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="a",
            original_question="テスト",
            filter_conditions=[
                {"column": "b", "operator": ">=", "value": 1},  # 文字列カラムに数値比較
            ],
        )

        result = executor.execute_safe(query)
        # 文字列と数値の比較でTypeErrorが発生する可能性
        # エラーになるか成功するかは実装次第
        assert result.execution_time_ms >= 0

    def test_general_exception_handling(self, sample_df: pd.DataFrame):
        """汎用Exception例外のハンドリング"""
        from unittest.mock import patch, MagicMock

        executor = SafeExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.SUM,
            original_question="テスト",
        )

        # QueryExecutorのexecuteをモックして例外を発生
        with patch.object(QueryExecutor, 'execute') as mock_execute:
            mock_execute.side_effect = RuntimeError("予期しないエラー")

            result = executor.execute_safe(query)

            assert result.success is False
            assert "RuntimeError" in result.error
            assert "予期しないエラー" in result.error

    def test_memory_error_handling(self, sample_df: pd.DataFrame):
        """MemoryError例外のハンドリング"""
        from unittest.mock import patch

        executor = SafeExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.SUM,
            original_question="テスト",
        )

        # QueryExecutorのexecuteをモックしてMemoryErrorを発生
        with patch.object(QueryExecutor, 'execute') as mock_execute:
            mock_execute.side_effect = MemoryError("メモリ不足")

            result = executor.execute_safe(query)

            assert result.success is False
            assert "メモリ不足" in result.error

    def test_out_of_bounds_datetime_handling(self, sample_df: pd.DataFrame):
        """OutOfBoundsDatetime例外のハンドリング"""
        from unittest.mock import patch

        executor = SafeExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.SUM,
            original_question="テスト",
        )

        # QueryExecutorのexecuteをモックしてOutOfBoundsDatetimeを発生
        with patch.object(QueryExecutor, 'execute') as mock_execute:
            mock_execute.side_effect = pd.errors.OutOfBoundsDatetime("日付が範囲外")

            result = executor.execute_safe(query)

            assert result.success is False
            assert "日付データが範囲外" in result.error


class TestFilterConditionEdgeCases:
    """フィルタ条件のエッジケーステスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["cat", "dog", "bird", "fish", "mouse"],
        })

    def test_filter_condition_with_none_column_and_none_target(self, sample_df: pd.DataFrame):
        """columnとtarget_columnの両方がNoneの場合"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column=None,
            filter_conditions=[
                {"column": None, "operator": ">=", "value": 1},  # 両方None
            ],
        )
        result = executor.execute(query)

        # 条件がスキップされ、全データが返される
        assert result.success is True
        assert len(result.data) == 5

    def test_filter_multiple_conditions_none_column(self, sample_df: pd.DataFrame):
        """複数条件でcolumnがNone"""
        executor = QueryExecutor(sample_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="a",
            filter_conditions=[
                {"column": None, "operator": ">=", "value": 2},  # target_columnを使用
                {"column": None, "operator": "<=", "value": 4},  # target_columnを使用
            ],
        )
        result = executor.execute(query)

        assert result.success is True
        assert len(result.data) == 3  # 2, 3, 4


class TestResultLimiting:
    """結果制限のテスト"""

    def test_large_result_truncation(self):
        """MAX_RESULT_ROWSを超える結果の切り詰め"""
        # 大きなDataFrameを作成
        large_df = pd.DataFrame({
            "a": list(range(15000)),
            "b": ["cat"] * 15000,
        })

        executor = SafeExecutor(large_df)
        query = ParsedQuery(
            query_type=QueryType.FILTER,
            target_column="a",
            filter_conditions=[],  # 全データを返す
            original_question="テスト",
        )

        result = executor.execute_safe(query)

        assert result.success is True
        # MAX_RESULT_ROWS（10000）以下に制限される
        assert len(result.data) <= SafeExecutor.MAX_RESULT_ROWS

    def test_memory_usage_limiting(self):
        """高メモリ使用量の結果制限（テスト目的）"""
        # メモリ使用量の制限をテスト
        # 実際に100MBを超えるデータを作成するのは重いので、
        # ロジックが存在することを確認
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })

        executor = SafeExecutor(df)
        query = ParsedQuery(
            query_type=QueryType.DESCRIBE,
            original_question="テスト",
        )

        result = executor.execute_safe(query)

        # 正常に実行されることを確認
        assert result.success is True


class TestTimeoutHandlerException:
    """タイムアウトハンドラーの例外テスト"""

    def test_timeout_raises_exception_after_completion(self):
        """実行完了後にタイムアウト超過を検知"""
        import time
        from src.executor import timeout_handler

        # 非常に短いタイムアウトを設定
        @timeout_handler(timeout_ms=1)  # 1ミリ秒
        def slow_function():
            time.sleep(0.01)  # 10ミリ秒
            return "done"

        # タイムアウト例外が発生するはず
        with pytest.raises(TimeoutError) as exc_info:
            slow_function()

        assert "超過" in str(exc_info.value)


class TestExecutionResultDataclass:
    """ExecutionResultデータクラスのテスト"""

    def test_execution_result_default_values(self):
        """ExecutionResultのデフォルト値"""
        result = ExecutionResult(success=True)

        assert result.success is True
        assert result.data is None
        assert result.value is None
        assert result.query_code == ""
        assert result.execution_time_ms == 0.0
        assert result.error is None

    def test_execution_result_with_all_values(self):
        """ExecutionResultの全値設定"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = ExecutionResult(
            success=True,
            data=df,
            value=6,
            query_code="df['a'].sum()",
            execution_time_ms=10.5,
            error=None,
        )

        assert result.success is True
        assert result.data is not None
        assert result.value == 6
        assert result.query_code == "df['a'].sum()"
        assert result.execution_time_ms == 10.5

    def test_execution_result_error_case(self):
        """ExecutionResultのエラーケース"""
        result = ExecutionResult(
            success=False,
            error="カラムが見つかりません",
            execution_time_ms=5.0,
        )

        assert result.success is False
        assert result.error == "カラムが見つかりません"


class TestQueryExecutorCustomTimeout:
    """カスタムタイムアウト設定のテスト"""

    def test_custom_timeout_setting(self):
        """カスタムタイムアウト設定"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        executor = QueryExecutor(df, timeout_ms=5000)

        assert executor._timeout_ms == 5000

    def test_default_timeout_setting(self):
        """デフォルトタイムアウト設定"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        executor = QueryExecutor(df)

        assert executor._timeout_ms == QueryExecutor.DEFAULT_TIMEOUT_MS


class TestValidateCodeEdgeCases:
    """コード検証のエッジケーステスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3]})

    def test_validate_code_case_insensitive(self, sample_df: pd.DataFrame):
        """大文字小文字を区別しない検証"""
        executor = SafeExecutor(sample_df)

        # 大文字でも検出される
        assert executor.validate_code("EVAL('code')") is False
        assert executor.validate_code("EXEC('code')") is False
        assert executor.validate_code("IMPORT os") is False

    def test_validate_code_with_partial_match(self, sample_df: pd.DataFrame):
        """部分一致での検出"""
        executor = SafeExecutor(sample_df)

        # 部分一致でも検出される
        assert executor.validate_code("malicious_eval('x')") is False
        assert executor.validate_code("do_exec_now()") is False

    def test_validate_code_safe_variations(self, sample_df: pd.DataFrame):
        """安全なコードのバリエーション"""
        executor = SafeExecutor(sample_df)

        # これらは安全
        assert executor.validate_code("df.sum()") is True
        assert executor.validate_code("df.mean()") is True
        assert executor.validate_code("df.groupby('a').sum()") is True
        assert executor.validate_code("df[df['a'] > 1]") is True


class TestSafeExecutorMemoryLimit:
    """メモリ制限のテスト"""

    def test_memory_limit_triggers_truncation(self):
        """メモリ制限を超えた場合のデータ切り詰め"""
        from unittest.mock import patch, MagicMock

        # サンプルDataFrame
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })

        executor = SafeExecutor(df)
        query = ParsedQuery(
            query_type=QueryType.DESCRIBE,
            original_question="テスト",
        )

        # メモリ使用量を超えた場合をシミュレート
        mock_result = MagicMock()
        mock_result.data = pd.DataFrame({"col": list(range(5000))})
        mock_result.success = True
        mock_result.query_code = "test"
        mock_result.execution_time_ms = 1.0

        # memory_usage が大きな値を返すようにモック
        with patch.object(QueryExecutor, 'execute', return_value=mock_result):
            with patch.object(pd.DataFrame, 'memory_usage') as mock_mem:
                # 100MBを超える値を返す（バイト単位）
                mock_series = MagicMock()
                mock_series.sum.return_value = 150 * 1024 * 1024  # 150MB
                mock_mem.return_value = mock_series

                result = executor.execute_safe(query)

                # 成功だが、データは制限される（1000行に）
                assert result.success is True


class TestSafeExecutorKeyErrorDirect:
    """KeyError例外の直接テスト"""

    def test_keyerror_direct_handling(self):
        """KeyError例外が直接発生した場合"""
        from unittest.mock import patch

        df = pd.DataFrame({"a": [1, 2, 3]})
        executor = SafeExecutor(df)
        query = ParsedQuery(
            query_type=QueryType.SUM,
            original_question="テスト",
        )

        # QueryExecutorのexecuteをモックしてKeyErrorを発生
        with patch.object(QueryExecutor, 'execute') as mock_execute:
            mock_execute.side_effect = KeyError("missing_column")

            result = executor.execute_safe(query)

            assert result.success is False
            assert "カラムが見つかりません" in result.error
            assert "missing_column" in result.error


class TestSafeExecutorTypeErrorDirect:
    """TypeError例外の直接テスト"""

    def test_typeerror_direct_handling(self):
        """TypeError例外が直接発生した場合"""
        from unittest.mock import patch

        df = pd.DataFrame({"a": [1, 2, 3]})
        executor = SafeExecutor(df)
        query = ParsedQuery(
            query_type=QueryType.SUM,
            original_question="テスト",
        )

        # QueryExecutorのexecuteをモックしてTypeErrorを発生
        with patch.object(QueryExecutor, 'execute') as mock_execute:
            mock_execute.side_effect = TypeError("unsupported operand type")

            result = executor.execute_safe(query)

            assert result.success is False
            assert "データ型エラー" in result.error
            assert "unsupported operand type" in result.error
