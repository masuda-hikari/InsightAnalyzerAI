"""LLMHandlerのテスト"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_handler import LLMHandler, LLMConfig, LLMResponse, LLMQueryParser


class TestLLMHandler:
    """LLMHandlerクラスのテスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """テスト用DataFrame"""
        return pd.DataFrame({
            "region": ["東京", "大阪", "名古屋"],
            "sales": [1000, 2000, 1500],
            "quantity": [10, 20, 15],
        })

    def test_init_without_api_key(self):
        """APIキーなしでの初期化"""
        # 環境変数をクリア
        with patch.dict(os.environ, {}, clear=True):
            handler = LLMHandler(LLMConfig(api_key=None))
            assert handler.is_available is False

    def test_init_with_invalid_api_key(self):
        """無効なAPIキーでの初期化"""
        config = LLMConfig(api_key="invalid-key")
        handler = LLMHandler(config)
        # インポートエラーまたは接続エラーでFalse
        # 実際のAPIキー検証は行わない

    def test_generate_code_unavailable(self, sample_df: pd.DataFrame):
        """LLM利用不可時のコード生成"""
        config = LLMConfig(api_key=None)
        handler = LLMHandler(config)

        result = handler.generate_code("売上の合計", sample_df)

        assert result.success is False
        assert "利用できません" in result.error

    def test_explain_result_unavailable(self, sample_df: pd.DataFrame):
        """LLM利用不可時の結果説明"""
        config = LLMConfig(api_key=None)
        handler = LLMHandler(config)

        result = handler.explain_result("売上の合計", 4500, sample_df)

        assert result.success is False

    def test_generate_schema(self, sample_df: pd.DataFrame):
        """スキーマ生成"""
        handler = LLMHandler(LLMConfig(api_key=None))
        schema = handler._generate_schema(sample_df)

        assert "region" in schema
        assert "sales" in schema
        assert "quantity" in schema

    def test_extract_code_python_block(self):
        """Pythonコードブロックの抽出"""
        handler = LLMHandler(LLMConfig(api_key=None))

        content = """説明テキスト
```python
result = df['sales'].sum()
```
追加テキスト"""

        code = handler._extract_code(content)
        assert "result = df['sales'].sum()" in code

    def test_extract_code_generic_block(self):
        """汎用コードブロックの抽出"""
        handler = LLMHandler(LLMConfig(api_key=None))

        content = """
```
result = df.groupby('region')['sales'].sum()
```
"""

        code = handler._extract_code(content)
        assert "groupby" in code

    def test_extract_code_no_block(self):
        """コードブロックなしの場合"""
        handler = LLMHandler(LLMConfig(api_key=None))

        content = "result = df['sales'].mean()"

        code = handler._extract_code(content)
        assert "result = df['sales'].mean()" in code


class TestLLMConfig:
    """LLMConfigのテスト"""

    def test_default_values(self):
        """デフォルト値"""
        config = LLMConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000

    def test_custom_values(self):
        """カスタム値"""
        config = LLMConfig(
            api_key="test-key",
            model="gpt-4o",
            temperature=0.5,
        )
        assert config.api_key == "test-key"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5


class TestLLMResponse:
    """LLMResponseのテスト"""

    def test_success_response(self):
        """成功レスポンス"""
        response = LLMResponse(
            success=True,
            pandas_code="result = df['sales'].sum()",
            tokens_used=50,
        )
        assert response.success is True
        assert response.pandas_code is not None
        assert response.error is None

    def test_error_response(self):
        """エラーレスポンス"""
        response = LLMResponse(
            success=False,
            error="API error",
        )
        assert response.success is False
        assert response.error == "API error"


class TestLLMQueryParser:
    """LLMQueryParserのテスト"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })

    def test_parse_without_llm(self, sample_df: pd.DataFrame):
        """LLM利用不可時のパース"""
        handler = LLMHandler(LLMConfig(api_key=None))
        parser = LLMQueryParser(handler)

        result = parser.parse_with_llm("合計", sample_df)
        assert "error" in result


class TestIntegration:
    """統合テスト（モック使用）"""

    @pytest.fixture
    def mock_openai_response(self):
        """OpenAI応答のモック"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """```python
result = df['sales'].sum()
```"""
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        return mock_response

    def test_generate_code_with_mock(self, mock_openai_response):
        """モックを使用したコード生成テスト"""
        # 実際のAPI呼び出しをモック化
        with patch.dict('sys.modules', {'openai': Mock()}):
            # LLMHandlerを直接設定
            config = LLMConfig(api_key="test-key")
            handler = LLMHandler(config)

            # モッククライアントを設定
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            handler._available = True
            handler._client = mock_client

            df = pd.DataFrame({"sales": [100, 200, 300]})
            result = handler.generate_code("売上の合計", df)

            assert result.success is True
            assert "sum()" in result.pandas_code
