"""
LLM統合モジュール

自然言語クエリをLLM（OpenAI）で解析し、Pandasコードを生成する
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

import pandas as pd

# OpenAI SDKは遅延インポート（インストールされていない場合のフォールバック用）


@dataclass
class LLMConfig:
    """LLM設定"""
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0  # 再現性のため低温度
    max_tokens: int = 1000
    timeout: int = 30


@dataclass
class LLMResponse:
    """LLM応答"""
    success: bool
    pandas_code: Optional[str] = None
    explanation: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0


class LLMHandler:
    """
    LLMを使用した自然言語→Pandasコード変換

    Phase 2で本格実装。現在はスタブ。
    """

    SYSTEM_PROMPT = """あなたはデータ分析の専門家です。
ユーザーの自然言語での質問を、Pandas DataFrameを操作するPythonコードに変換してください。

ルール:
1. 変数名は必ず `df` を使用
2. 結果は `result` 変数に格納
3. 安全なコードのみ生成（eval, exec, __は禁止）
4. 日本語のカラム名にも対応
5. コードのみを返す（説明は別途）

スキーマ情報:
{schema}

サンプルデータ:
{sample}
"""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Args:
            config: LLM設定
        """
        self._config = config or LLMConfig()

        # APIキーを環境変数から取得
        if self._config.api_key is None:
            self._config.api_key = os.getenv("OPENAI_API_KEY")

        self._client = None
        self._available = False
        self._init_client()

    def _init_client(self) -> None:
        """OpenAIクライアントを初期化"""
        if not self._config.api_key:
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._config.api_key)
            self._available = True
        except ImportError:
            # OpenAIライブラリがインストールされていない
            pass
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """LLMが利用可能か"""
        return self._available

    def generate_code(
        self,
        question: str,
        df: pd.DataFrame,
    ) -> LLMResponse:
        """
        自然言語クエリからPandasコードを生成

        Args:
            question: ユーザーの質問
            df: 対象DataFrame

        Returns:
            LLM応答
        """
        if not self._available:
            return LLMResponse(
                success=False,
                error="LLMが利用できません。OPENAI_API_KEYを設定してください。"
            )

        try:
            # スキーマ情報を生成
            schema = self._generate_schema(df)
            sample = df.head(3).to_string()

            # プロンプト構築
            system_prompt = self.SYSTEM_PROMPT.format(
                schema=schema,
                sample=sample,
            )

            # API呼び出し
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            # 応答解析
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            # コード抽出（```python ... ``` の中身）
            code = self._extract_code(content)

            return LLMResponse(
                success=True,
                pandas_code=code,
                raw_response=content,
                tokens_used=tokens_used,
            )

        except Exception as e:
            return LLMResponse(
                success=False,
                error=str(e),
            )

    def explain_result(
        self,
        question: str,
        result: Any,
        df: pd.DataFrame,
    ) -> LLMResponse:
        """
        分析結果を自然言語で説明

        Args:
            question: 元の質問
            result: 分析結果
            df: 対象DataFrame

        Returns:
            説明を含むLLM応答
        """
        if not self._available:
            return LLMResponse(
                success=False,
                error="LLMが利用できません"
            )

        try:
            # 結果を文字列化
            if isinstance(result, pd.DataFrame):
                result_str = result.to_string()
            elif isinstance(result, pd.Series):
                result_str = result.to_string()
            else:
                result_str = str(result)

            prompt = f"""以下の分析結果を、ビジネスパーソン向けに分かりやすく日本語で説明してください。

質問: {question}

結果:
{result_str}

説明（箇条書きで簡潔に）:"""

            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            return LLMResponse(
                success=True,
                explanation=content,
                raw_response=content,
                tokens_used=tokens_used,
            )

        except Exception as e:
            return LLMResponse(
                success=False,
                error=str(e),
            )

    def _generate_schema(self, df: pd.DataFrame) -> str:
        """DataFrameのスキーマ情報を生成"""
        lines = []
        for col in df.columns:
            dtype = df[col].dtype
            sample_values = df[col].dropna().head(3).tolist()
            lines.append(f"- {col} ({dtype}): 例 {sample_values}")
        return "\n".join(lines)

    def _extract_code(self, content: str) -> str:
        """LLM応答からPythonコードを抽出"""
        # ```python ... ``` パターン
        if "```python" in content:
            start = content.find("```python") + len("```python")
            end = content.find("```", start)
            if end > start:
                return content[start:end].strip()

        # ``` ... ``` パターン
        if "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                return content[start:end].strip()

        # そのまま返す
        return content.strip()


class LLMQueryParser:
    """
    LLMを使用した高度なクエリパーサー

    Phase 1のキーワードベースパーサーの代替/補完
    """

    def __init__(self, handler: Optional[LLMHandler] = None):
        self._handler = handler or LLMHandler()

    def parse_with_llm(
        self,
        question: str,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        LLMを使用してクエリを解析

        Returns:
            解析結果（query_type, target_column, group_column等）
        """
        if not self._handler.is_available:
            return {"error": "LLM not available"}

        prompt = f"""以下の質問を分析し、JSON形式で回答してください。

質問: {question}

カラム: {list(df.columns)}

出力形式:
{{
    "query_type": "sum" | "mean" | "count" | "groupby" | "filter" | "describe",
    "target_column": "対象の数値カラム名",
    "group_column": "グループ化カラム名（あれば）",
    "filter_conditions": [条件リスト],
    "confidence": 0.0-1.0の確信度
}}
"""

        try:
            response = self._handler._client.chat.completions.create(
                model=self._handler._config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )

            content = response.choices[0].message.content

            # JSONを抽出・パース
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                return json.loads(json_str)

            return {"error": "JSON parse failed"}

        except Exception as e:
            return {"error": str(e)}


def create_llm_handler() -> LLMHandler:
    """LLMハンドラーを作成するヘルパー関数"""
    config = LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    return LLMHandler(config)
