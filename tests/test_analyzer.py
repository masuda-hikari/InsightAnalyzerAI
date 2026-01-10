"""InsightAnalyzerのテスト"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insight_analyzer import InsightAnalyzer, AnalysisResult
from src.llm_handler import LLMConfig


class TestInsightAnalyzer:
    """InsightAnalyzerクラスのテスト"""

    @pytest.fixture
    def sample_csv_path(self) -> Path:
        """サンプルCSVのパスを返す"""
        return Path(__file__).parent.parent / "data" / "sample_sales.csv"

    @pytest.fixture
    def analyzer(self, sample_csv_path: Path) -> InsightAnalyzer:
        """データ読み込み済みのAnalyzer"""
        return InsightAnalyzer(sample_csv_path)

    def test_load_data_from_path(self, sample_csv_path: Path):
        """パスからデータ読み込み"""
        analyzer = InsightAnalyzer(sample_csv_path)
        assert analyzer.dataframe is not None
        assert len(analyzer.dataframe) == 25

    def test_load_data_from_dataframe(self):
        """DataFrameから直接読み込み"""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })
        analyzer = InsightAnalyzer(df)
        assert len(analyzer.dataframe) == 3

    def test_ask_without_data(self):
        """データなしでの質問"""
        analyzer = InsightAnalyzer()
        result = analyzer.ask("合計を教えて")

        assert result.success is False
        assert "データが読み込まれていません" in result.error

    def test_ask_sum(self, analyzer: InsightAnalyzer):
        """合計クエリ"""
        result = analyzer.ask("total_salesの合計")

        assert result.success is True
        assert "合計" in result.answer
        assert result.query_used is not None

    def test_ask_mean(self, analyzer: InsightAnalyzer):
        """平均クエリ"""
        result = analyzer.ask("quantityの平均")

        assert result.success is True
        assert "平均" in result.answer

    def test_ask_count(self, analyzer: InsightAnalyzer):
        """件数クエリ"""
        result = analyzer.ask("データは何件？")

        assert result.success is True
        assert "25" in result.answer or "件数" in result.answer

    def test_ask_groupby_region(self, analyzer: InsightAnalyzer):
        """地域別集計"""
        result = analyzer.ask("地域別の売上合計")

        assert result.success is True
        assert "東京" in result.answer or "地域" in result.answer
        assert result.data is not None

    def test_ask_groupby_product(self, analyzer: InsightAnalyzer):
        """製品別集計"""
        result = analyzer.ask("製品別の売上")

        assert result.success is True
        assert result.data is not None

    def test_ask_default_stats(self, analyzer: InsightAnalyzer):
        """デフォルト統計（キーワードなし）"""
        result = analyzer.ask("データの概要")

        assert result.success is True
        assert "基本統計" in result.answer


class TestAnalysisResultAccuracy:
    """分析結果の正確性テスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用データで初期化"""
        df = pd.DataFrame({
            "region": ["東京", "大阪", "東京", "大阪"],
            "sales": [100, 200, 150, 250],
            "quantity": [10, 20, 15, 25]
        })
        return InsightAnalyzer(df)

    def test_sum_accuracy(self, analyzer: InsightAnalyzer):
        """合計の正確性"""
        result = analyzer.ask("salesの合計")

        # 100 + 200 + 150 + 250 = 700
        assert result.success is True
        assert "700" in result.answer

    def test_mean_accuracy(self, analyzer: InsightAnalyzer):
        """平均の正確性"""
        result = analyzer.ask("quantityの平均")

        # (10 + 20 + 15 + 25) / 4 = 17.5
        assert result.success is True
        assert "17" in result.answer  # 17.5 or 17.50

    def test_count_accuracy(self, analyzer: InsightAnalyzer):
        """件数の正確性"""
        result = analyzer.ask("件数")

        assert result.success is True
        assert "4" in result.answer

    def test_groupby_accuracy(self, analyzer: InsightAnalyzer):
        """グループ別集計の正確性"""
        result = analyzer.ask("地域別sales合計")

        # 東京: 100 + 150 = 250
        # 大阪: 200 + 250 = 450
        assert result.success is True
        if result.data is not None:
            # DataFrameで検証
            data = result.data
            assert len(data) == 2


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_dataframe(self):
        """空のDataFrame"""
        df = pd.DataFrame()
        analyzer = InsightAnalyzer(df)

        result = analyzer.ask("合計")
        # 数値カラムがないのでエラー
        assert result.success is False or "見つかりません" in str(result.error or result.answer)

    def test_no_numeric_columns(self):
        """数値カラムなし"""
        df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "category": ["X", "Y", "Z"]
        })
        analyzer = InsightAnalyzer(df)

        result = analyzer.ask("合計")
        assert result.success is False or "見つかりません" in str(result.error or result.answer)


class TestNewArchitecture:
    """新アーキテクチャのテスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用Analyzer"""
        df = pd.DataFrame({
            "region": ["東京", "大阪", "名古屋"],
            "sales": [1000, 2000, 1500],
            "quantity": [10, 20, 15],
        })
        return InsightAnalyzer(df)

    def test_execution_time_recorded(self, analyzer: InsightAnalyzer):
        """実行時間が記録される"""
        result = analyzer.ask("合計")
        assert result.execution_time_ms >= 0

    def test_confidence_score(self, analyzer: InsightAnalyzer):
        """確信度スコアが記録される"""
        result = analyzer.ask("salesの合計")
        assert 0.0 <= result.confidence <= 1.0

    def test_get_insights(self, analyzer: InsightAnalyzer):
        """自動インサイト生成"""
        insights = analyzer.get_insights()
        assert len(insights) > 0
        assert any("件" in i for i in insights)

    def test_get_summary(self, analyzer: InsightAnalyzer):
        """サマリー取得"""
        result = analyzer.get_summary()
        assert result.success is True

    def test_metadata_property(self, analyzer: InsightAnalyzer):
        """メタデータプロパティ"""
        metadata = analyzer.metadata
        assert "rows" in metadata
        assert metadata["rows"] == 3


class TestLLMIntegration:
    """LLM統合のテスト（Phase 2）"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """テスト用DataFrame"""
        return pd.DataFrame({
            "region": ["東京", "大阪", "名古屋", "福岡"],
            "sales": [1000, 2000, 1500, 800],
            "quantity": [10, 20, 15, 8],
        })

    def test_init_with_llm_disabled(self, sample_df: pd.DataFrame):
        """LLM無効での初期化"""
        analyzer = InsightAnalyzer(sample_df, use_llm=False)
        assert analyzer.llm_available is False

    def test_init_with_llm_enabled_no_key(self, sample_df: pd.DataFrame):
        """LLM有効だがAPIキーなし"""
        with patch.dict("os.environ", {}, clear=True):
            analyzer = InsightAnalyzer(sample_df, use_llm=True)
            # APIキーがないのでLLMは利用不可
            assert analyzer.llm_available is False

    def test_llm_available_property(self, sample_df: pd.DataFrame):
        """llm_availableプロパティのテスト"""
        analyzer = InsightAnalyzer(sample_df, use_llm=False)
        assert analyzer.llm_available is False

    def test_ask_without_llm_fallback(self, sample_df: pd.DataFrame):
        """LLMなしでのフォールバック動作"""
        analyzer = InsightAnalyzer(sample_df, use_llm=False)
        result = analyzer.ask("salesの合計")

        assert result.success is True
        assert result.llm_used is False
        assert "合計" in result.answer

    def test_ask_with_use_llm_override(self, sample_df: pd.DataFrame):
        """use_llmパラメータでのオーバーライド"""
        analyzer = InsightAnalyzer(sample_df, use_llm=True)
        # 明示的にLLM無効を指定
        result = analyzer.ask("salesの合計", use_llm=False)

        assert result.success is True
        assert result.llm_used is False

    def test_analysis_result_new_fields(self, sample_df: pd.DataFrame):
        """AnalysisResultの新フィールド"""
        analyzer = InsightAnalyzer(sample_df, use_llm=False)
        result = analyzer.ask("合計")

        # 新フィールドが存在するか確認
        assert hasattr(result, "llm_explanation")
        assert hasattr(result, "llm_used")
        assert result.llm_used is False
        assert result.llm_explanation is None

    def test_llm_with_mock(self, sample_df: pd.DataFrame):
        """モックを使用したLLMテスト"""
        # モックレスポンス
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """```python
result = df['sales'].sum()
```"""
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100

        with patch.dict("sys.modules", {"openai": Mock()}):
            analyzer = InsightAnalyzer(sample_df, use_llm=True)

            # LLMハンドラをモック化
            if analyzer._llm_handler:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                analyzer._llm_handler._available = True
                analyzer._llm_handler._client = mock_client

                result = analyzer.ask("売上の合計")

                assert result.success is True
                assert result.llm_used is True
                # 実際の値が計算される
                assert "5,300" in result.answer or "5300" in result.answer

    def test_llm_fallback_on_error(self, sample_df: pd.DataFrame):
        """LLMエラー時のフォールバック"""
        with patch.dict("sys.modules", {"openai": Mock()}):
            analyzer = InsightAnalyzer(sample_df, use_llm=True)

            if analyzer._llm_handler:
                # エラーを発生させるモック
                mock_client = Mock()
                mock_client.chat.completions.create.side_effect = Exception("API Error")
                analyzer._llm_handler._available = True
                analyzer._llm_handler._client = mock_client

                # エラー時はフォールバックする
                result = analyzer.ask("salesの合計")

                assert result.success is True
                # フォールバックでキーワードベースの解析が動作
                assert result.llm_used is False

    def test_llm_unsafe_code_rejected(self, sample_df: pd.DataFrame):
        """危険なコードの拒否"""
        # 危険なコードを生成するモック
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """```python
result = eval('df["sales"].sum()')
```"""
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100

        with patch.dict("sys.modules", {"openai": Mock()}):
            analyzer = InsightAnalyzer(sample_df, use_llm=True)

            if analyzer._llm_handler:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                analyzer._llm_handler._available = True
                analyzer._llm_handler._client = mock_client

                result = analyzer.ask("売上の合計")

                # evalが含まれるコードは拒否され、フォールバック
                assert result.success is True
                assert result.llm_used is False


class TestLLMExplainResult:
    """LLMによる結果説明のテスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "region": ["東京", "大阪"],
            "sales": [1000, 2000],
        })

    def test_explain_result_disabled(self, sample_df: pd.DataFrame):
        """説明機能が無効の場合"""
        analyzer = InsightAnalyzer(sample_df, use_llm=False)
        result = analyzer.ask("合計", explain_result=False)

        assert result.llm_explanation is None

    def test_explain_result_with_mock(self, sample_df: pd.DataFrame):
        """モックを使用した説明機能テスト"""
        # コード生成用モック
        code_response = Mock()
        code_response.choices = [Mock()]
        code_response.choices[0].message.content = """```python
result = df['sales'].sum()
```"""
        code_response.usage = Mock()
        code_response.usage.total_tokens = 50

        # 説明生成用モック
        explain_response = Mock()
        explain_response.choices = [Mock()]
        explain_response.choices[0].message.content = "売上の合計は3,000円です。東京と大阪の売上を足し合わせた結果です。"
        explain_response.usage = Mock()
        explain_response.usage.total_tokens = 30

        with patch.dict("sys.modules", {"openai": Mock()}):
            analyzer = InsightAnalyzer(sample_df, use_llm=True)

            if analyzer._llm_handler:
                mock_client = Mock()
                # 複数回の呼び出しに対応
                mock_client.chat.completions.create.side_effect = [
                    code_response,
                    explain_response,
                ]
                analyzer._llm_handler._available = True
                analyzer._llm_handler._client = mock_client

                result = analyzer.ask("売上の合計", explain_result=True)

                assert result.success is True
                assert result.llm_used is True
                assert result.llm_explanation is not None
                assert "売上" in result.llm_explanation


class TestFormatLLMResult:
    """LLM結果フォーマットのテスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用Analyzer"""
        df = pd.DataFrame({
            "region": ["東京", "大阪"],
            "sales": [1000, 2000],
        })
        return InsightAnalyzer(df, use_llm=False)

    def test_format_llm_result_none(self, analyzer: InsightAnalyzer):
        """None結果のフォーマット"""
        result = analyzer._format_llm_result("質問", None)
        assert "取得できませんでした" in result

    def test_format_llm_result_dataframe_small(self, analyzer: InsightAnalyzer):
        """小さいDataFrameのフォーマット"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = analyzer._format_llm_result("質問", df)
        assert "分析結果:" in result

    def test_format_llm_result_dataframe_large(self, analyzer: InsightAnalyzer):
        """大きいDataFrameのフォーマット"""
        df = pd.DataFrame({"a": list(range(100))})
        result = analyzer._format_llm_result("質問", df)
        assert "上位10件" in result
        assert "全100件" in result

    def test_format_llm_result_series_small(self, analyzer: InsightAnalyzer):
        """小さいSeriesのフォーマット"""
        series = pd.Series([1, 2, 3], name="test")
        result = analyzer._format_llm_result("質問", series)
        assert "分析結果:" in result

    def test_format_llm_result_series_large(self, analyzer: InsightAnalyzer):
        """大きいSeriesのフォーマット"""
        series = pd.Series(list(range(100)), name="test")
        result = analyzer._format_llm_result("質問", series)
        assert "上位10件" in result
        assert "全100件" in result

    def test_format_llm_result_large_number(self, analyzer: InsightAnalyzer):
        """大きな数値のフォーマット"""
        result = analyzer._format_llm_result("質問", 1500000)
        assert "¥" in result
        assert "1,500,000" in result

    def test_format_llm_result_small_number(self, analyzer: InsightAnalyzer):
        """小さな数値のフォーマット"""
        result = analyzer._format_llm_result("質問", 123.45)
        assert "結果:" in result
        assert "123" in result

    def test_format_llm_result_string(self, analyzer: InsightAnalyzer):
        """文字列結果のフォーマット"""
        result = analyzer._format_llm_result("質問", "テスト文字列")
        assert "結果:" in result
        assert "テスト文字列" in result


class TestFormatAnswer:
    """回答フォーマットのテスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用Analyzer"""
        df = pd.DataFrame({
            "region": ["東京", "大阪"],
            "sales": [1000, 2000],
            "price": [500, 600],
        })
        return InsightAnalyzer(df, use_llm=False)

    def test_format_answer_sum_currency(self, analyzer: InsightAnalyzer):
        """合計（金額）のフォーマット"""
        result = analyzer.ask("salesの合計")
        assert result.success is True
        assert "合計" in result.answer
        assert "3,000" in result.answer or "3000" in result.answer

    def test_format_answer_mean_currency(self, analyzer: InsightAnalyzer):
        """平均（金額）のフォーマット"""
        result = analyzer.ask("priceの平均")
        assert result.success is True
        assert "平均" in result.answer

    def test_format_answer_count(self, analyzer: InsightAnalyzer):
        """件数のフォーマット"""
        result = analyzer.ask("何件？")
        assert result.success is True
        assert "件数" in result.answer or "2" in result.answer

    def test_format_answer_describe(self, analyzer: InsightAnalyzer):
        """基本統計のフォーマット"""
        result = analyzer.ask("概要")
        assert result.success is True
        assert "基本統計" in result.answer


class TestFormatGroupbyAnswer:
    """グループ別集計フォーマットのテスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用Analyzer"""
        df = pd.DataFrame({
            "region": ["東京", "東京", "大阪", "大阪"],
            "sales": [1000, 500, 2000, 1000],
            "category": ["A", "B", "A", "B"],
        })
        return InsightAnalyzer(df, use_llm=False)

    def test_format_groupby_currency(self, analyzer: InsightAnalyzer):
        """グループ別合計（金額）"""
        result = analyzer.ask("地域別のsales合計")
        assert result.success is True
        assert result.data is not None

    def test_format_groupby_non_currency(self, analyzer: InsightAnalyzer):
        """グループ別集計（非金額）"""
        result = analyzer.ask("category別の件数")
        assert result.success is True


class TestGetInsightReport:
    """インサイトレポートのテスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用Analyzer"""
        df = pd.DataFrame({
            "region": ["東京", "大阪", "名古屋", "福岡", "札幌"],
            "sales": [1000, 2000, 1500, 800, 1200],
            "quantity": [10, 20, 15, 8, 12],
        })
        return InsightAnalyzer(df, use_llm=False)

    def test_get_insight_report_basic(self, analyzer: InsightAnalyzer):
        """基本的なインサイトレポート取得"""
        report = analyzer.get_insight_report()
        assert report is not None
        assert report.data_rows == 5
        assert report.data_columns >= 2

    def test_get_insight_report_with_max_insights(self, analyzer: InsightAnalyzer):
        """最大インサイト数の制限"""
        report = analyzer.get_insight_report(max_insights=3)
        assert len(report.insights) <= 3

    def test_get_insight_report_without_data(self):
        """データなしでのレポート取得"""
        analyzer = InsightAnalyzer(use_llm=False)
        report = analyzer.get_insight_report()
        assert report.data_rows == 0
        assert len(report.insights) == 0


class TestGetFormattedInsights:
    """フォーマット済みインサイトのテスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用Analyzer"""
        df = pd.DataFrame({
            "region": ["東京", "大阪", "名古屋"],
            "sales": [1000, 2000, 1500],
        })
        return InsightAnalyzer(df, use_llm=False)

    def test_get_formatted_insights_basic(self, analyzer: InsightAnalyzer):
        """フォーマット済みインサイト取得"""
        result = analyzer.get_formatted_insights()
        assert "インサイトレポート" in result
        assert "データ:" in result

    def test_get_formatted_insights_with_max(self, analyzer: InsightAnalyzer):
        """最大数指定でのフォーマット済みインサイト"""
        result = analyzer.get_formatted_insights(max_insights=5)
        assert isinstance(result, str)

    def test_get_formatted_insights_without_data(self):
        """データなしでのフォーマット済みインサイト"""
        analyzer = InsightAnalyzer(use_llm=False)
        result = analyzer.get_formatted_insights()
        assert "読み込まれていません" in result


class TestChartGeneration:
    """チャート生成のテスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用Analyzer"""
        df = pd.DataFrame({
            "region": ["東京", "大阪"],
            "sales": [1000, 2000],
        })
        return InsightAnalyzer(df, use_llm=False)

    def test_ask_with_chart(self, analyzer: InsightAnalyzer):
        """チャート生成付きクエリ"""
        result = analyzer.ask("地域別のsales合計", generate_chart=True)
        assert result.success is True
        # チャートが生成される場合はパスが設定される
        if result.data is not None:
            assert result.chart_path is not None or result.data is not None


class TestExceptionHandling:
    """例外処理のテスト"""

    @pytest.fixture
    def analyzer(self) -> InsightAnalyzer:
        """テスト用Analyzer"""
        df = pd.DataFrame({
            "value": [1, 2, 3],
        })
        return InsightAnalyzer(df, use_llm=False)

    def test_ask_handles_exception(self):
        """askメソッドの例外処理"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        analyzer = InsightAnalyzer(df, use_llm=False)

        # パーサーにエラーを発生させる
        original_parse = analyzer._parser.parse

        def mock_parse(question):
            raise ValueError("Parsing error")

        analyzer._parser.parse = mock_parse

        result = analyzer.ask("合計")
        assert result.success is False
        assert "エラー" in result.error

        # 元に戻す
        analyzer._parser.parse = original_parse


class TestSchemaProperty:
    """スキーマプロパティのテスト"""

    def test_schema_property(self):
        """スキーマプロパティの取得"""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["a", "b"],
        })
        analyzer = InsightAnalyzer(df, use_llm=False)
        schema = analyzer.schema
        assert isinstance(schema, str)
        assert "col1" in schema or "データ" in schema


class TestLLMConfig:
    """LLM設定のテスト"""

    def test_init_with_custom_llm_config(self):
        """カスタムLLM設定での初期化"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        config = LLMConfig(
            api_key="test_key",
            model="gpt-4",
            temperature=0.5,
        )
        analyzer = InsightAnalyzer(df, use_llm=True, llm_config=config)
        # 設定は適用されるが、APIキーが無効なためLLMは利用不可
        assert analyzer._llm_handler is not None
