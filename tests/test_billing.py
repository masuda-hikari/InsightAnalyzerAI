"""
InsightAnalyzerAI - 課金システムテスト

Stripe統合・課金機能のテスト
"""

import pytest
from unittest.mock import MagicMock, patch
import os

# Streamlitのモック
import sys
sys.modules['streamlit'] = MagicMock()

from src.auth import PlanType
from src.billing import (
    PriceConfig,
    PRICE_CONFIGS,
    BillingManager,
)


class TestPriceConfig:
    """価格設定のテスト"""

    def test_free_plan_config(self):
        """Freeプランの価格設定"""
        config = PRICE_CONFIGS[PlanType.FREE]
        assert config.price_jpy == 0
        assert config.stripe_price_id is None
        assert len(config.features) > 0

    def test_basic_plan_config(self):
        """Basicプランの価格設定"""
        config = PRICE_CONFIGS[PlanType.BASIC]
        assert config.price_jpy == 2980
        assert "50MB/ファイル" in config.features

    def test_pro_plan_config(self):
        """Proプランの価格設定"""
        config = PRICE_CONFIGS[PlanType.PRO]
        assert config.price_jpy == 9800
        assert "API連携" in config.features

    def test_enterprise_plan_config(self):
        """Enterpriseプランの価格設定"""
        config = PRICE_CONFIGS[PlanType.ENTERPRISE]
        assert config.price_jpy == 0  # 要見積
        assert "オンプレミス対応" in config.features


class TestBillingManager:
    """BillingManagerのテスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)
        return st.session_state

    @pytest.fixture
    def mock_env_no_stripe(self, monkeypatch):
        """Stripeキーなしの環境"""
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)

    def test_init_without_stripe(self, mock_session_state, mock_env_no_stripe):
        """Stripeキーなしでの初期化"""
        # stripeモジュール自体をモックして、インポートできないようにする
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            assert manager.stripe_available is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_create_checkout_session_without_stripe(self, mock_session_state, mock_env_no_stripe):
        """Stripeなしでのチェックアウトセッション作成"""
        manager = BillingManager()
        url = manager.create_checkout_session(PlanType.BASIC)
        assert url is None

    def test_get_subscription_status_without_stripe(self, mock_session_state, mock_env_no_stripe):
        """Stripeなしでのサブスクリプション状態取得"""
        manager = BillingManager()
        status = manager.get_subscription_status()
        assert status is None

    def test_cancel_subscription_without_stripe(self, mock_session_state, mock_env_no_stripe):
        """Stripeなしでのサブスクリプションキャンセル"""
        manager = BillingManager()
        result = manager.cancel_subscription()
        assert result is False


class TestPlanFeatures:
    """プラン機能のテスト"""

    def test_all_plans_have_features(self):
        """全プランに機能リストがある"""
        for plan_type, config in PRICE_CONFIGS.items():
            assert len(config.features) > 0, f"{plan_type.value}に機能がありません"

    def test_higher_plans_have_more_features(self):
        """上位プランはより多くの機能を持つ"""
        free_features = len(PRICE_CONFIGS[PlanType.FREE].features)
        basic_features = len(PRICE_CONFIGS[PlanType.BASIC].features)
        pro_features = len(PRICE_CONFIGS[PlanType.PRO].features)

        assert basic_features >= free_features
        assert pro_features >= basic_features

    def test_price_ordering(self):
        """価格は上位プランほど高い"""
        free_price = PRICE_CONFIGS[PlanType.FREE].price_jpy
        basic_price = PRICE_CONFIGS[PlanType.BASIC].price_jpy
        pro_price = PRICE_CONFIGS[PlanType.PRO].price_jpy

        assert free_price == 0
        assert basic_price > free_price
        assert pro_price > basic_price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
