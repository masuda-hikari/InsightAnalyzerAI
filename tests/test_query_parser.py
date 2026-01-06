"""QueryParserのテスト"""

from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_parser import QueryParser, QueryType, ParsedQuery


class TestQueryParser:
    """QueryParserクラスのテスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """テスト用DataFrame"""
        return pd.DataFrame({
            "region": ["東京", "大阪", "名古屋"],
            "product": ["A", "B", "C"],
            "total_sales": [1000, 2000, 1500],
            "quantity": [10, 20, 15],
        })

    @pytest.fixture
    def parser(self, sample_df: pd.DataFrame) -> QueryParser:
        """スキーマ設定済みParser"""
        parser = QueryParser()
        parser.set_schema(sample_df)
        return parser

    def test_detect_sum_query(self, parser: QueryParser):
        """合計クエリの検出"""
        result = parser.parse("売上の合計を教えて")
        assert result.query_type == QueryType.SUM

        result = parser.parse("total sales")
        assert result.query_type == QueryType.SUM

    def test_detect_mean_query(self, parser: QueryParser):
        """平均クエリの検出"""
        result = parser.parse("平均値を計算して")
        assert result.query_type == QueryType.MEAN

        result = parser.parse("average quantity")
        assert result.query_type == QueryType.MEAN

    def test_detect_count_query(self, parser: QueryParser):
        """件数クエリの検出"""
        result = parser.parse("データは何件ありますか？")
        assert result.query_type == QueryType.COUNT

        result = parser.parse("count rows")
        assert result.query_type == QueryType.COUNT

    def test_detect_groupby_query(self, parser: QueryParser):
        """グループ別クエリの検出"""
        result = parser.parse("地域別の売上")
        assert result.query_type == QueryType.GROUPBY

        result = parser.parse("製品ごとの数量")
        assert result.query_type == QueryType.GROUPBY

    def test_detect_describe_query(self, parser: QueryParser):
        """基本統計クエリの検出"""
        result = parser.parse("データの概要")
        assert result.query_type == QueryType.DESCRIBE

    def test_find_target_column(self, parser: QueryParser):
        """対象カラムの検出"""
        result = parser.parse("total_salesの合計")
        assert result.target_column == "total_sales"

    def test_find_group_column(self, parser: QueryParser):
        """グループカラムの検出"""
        result = parser.parse("地域別の売上")
        assert result.group_column == "region"

    def test_confidence_score(self, parser: QueryParser):
        """確信度スコアの計算"""
        # 明確なクエリは高い確信度
        result = parser.parse("total_salesの合計")
        assert result.confidence >= 0.7

        # 曖昧なクエリは低い確信度
        result = parser.parse("なんかいい感じに")
        assert result.confidence < 0.7


class TestParsedQuery:
    """ParsedQueryクラスのテスト"""

    def test_default_values(self):
        """デフォルト値"""
        query = ParsedQuery(query_type=QueryType.SUM)
        assert query.target_column is None
        assert query.group_column is None
        assert query.filter_conditions == []
        assert query.aggregation == "sum"


class TestFilterParsing:
    """フィルタ解析のテスト"""

    @pytest.fixture
    def parser(self) -> QueryParser:
        return QueryParser()

    def test_parse_greater_than(self, parser: QueryParser):
        """以上条件の解析"""
        result = parser.parse("1000以上のデータ")
        assert len(result.filter_conditions) > 0
        assert result.filter_conditions[0]["operator"] == ">="
        assert result.filter_conditions[0]["value"] == 1000

    def test_parse_less_than(self, parser: QueryParser):
        """以下条件の解析"""
        result = parser.parse("500以下のデータ")
        assert len(result.filter_conditions) > 0
        assert result.filter_conditions[0]["operator"] == "<="
        assert result.filter_conditions[0]["value"] == 500
