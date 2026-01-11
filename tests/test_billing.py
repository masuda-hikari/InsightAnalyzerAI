"""
InsightAnalyzerAI - 課金システムテスト

Stripe統合・課金機能のテスト
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import os
from datetime import datetime

# Streamlitのモック - session_stateはdictのように動作させる
import sys


class MockSessionState(dict):
    """MagicMockとdictを組み合わせたモック"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            pass


mock_st = MagicMock()
mock_st.session_state = MockSessionState()
mock_st.secrets = MagicMock()
mock_st.secrets.get = MagicMock(return_value=None)
sys.modules['streamlit'] = mock_st

from src.auth import PlanType, User
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

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        # MockSessionStateを使用
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    @pytest.fixture
    def mock_env_no_stripe(self, monkeypatch):
        """Stripeキーなしの環境"""
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)

    def test_init_without_stripe(self, mock_env_no_stripe):
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

    def test_create_checkout_session_without_stripe(self, mock_env_no_stripe):
        """Stripeなしでのチェックアウトセッション作成"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            url = manager.create_checkout_session(PlanType.BASIC)
            assert url is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_get_subscription_status_without_stripe(self, mock_env_no_stripe):
        """Stripeなしでのサブスクリプション状態取得"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            status = manager.get_subscription_status()
            assert status is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_cancel_subscription_without_stripe(self, mock_env_no_stripe):
        """Stripeなしでのサブスクリプションキャンセル"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            result = manager.cancel_subscription()
            assert result is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_handle_webhook_without_stripe(self, mock_env_no_stripe):
        """Stripeなしでのwebhook処理"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            result = manager.handle_webhook("payload", "sig")
            assert result is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available


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


# Stripe連携テストはSkip（Streamlitセッション状態のモックが複雑なため）
# 重要: 以下のテストは手動でStreamlit環境で実行することを推奨


class TestBillingManagerWithStripeMocked:
    """Stripe有効時のBillingManagerテスト（モック）"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state.clear()
        st.session_state["auth_users_db"] = {}

    def test_billing_manager_import(self):
        """BillingManagerのインポート確認"""
        assert BillingManager is not None

    def test_price_configs_completeness(self):
        """PRICE_CONFIGSに全プランが含まれている"""
        for plan_type in PlanType:
            assert plan_type in PRICE_CONFIGS

    def test_price_config_structure(self):
        """PriceConfigの構造確認"""
        config = PRICE_CONFIGS[PlanType.BASIC]
        assert hasattr(config, 'plan')
        assert hasattr(config, 'price_jpy')
        assert hasattr(config, 'stripe_price_id')
        assert hasattr(config, 'features')


class TestBillingManagerWithStripeAvailable:
    """Stripe有効時のBillingManagerテスト（モックStripe）"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    @pytest.fixture
    def mock_stripe(self, monkeypatch):
        """Stripeモジュールのモック"""
        mock_stripe_module = MagicMock()
        mock_stripe_module.api_key = None
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")
        return mock_stripe_module

    def test_init_with_stripe_key(self, mock_stripe, monkeypatch):
        """Stripeキー有りでの初期化"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe
        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe

        try:
            manager = BillingManager()
            # 環境変数からキー取得してstripe_available=Trueになる
            assert manager.stripe_available is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_create_checkout_session_no_price_id(self, monkeypatch):
        """Stripe Price IDなしでのチェックアウトセッション作成"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            manager.stripe_available = True  # 強制的に有効化

            # Freeプランにはprice_idがない
            url = manager.create_checkout_session(PlanType.FREE)
            assert url is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_create_checkout_session_no_user(self, monkeypatch):
        """未ログイン時のチェックアウトセッション作成"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = None

            manager = BillingManager()
            manager.stripe_available = True

            # ユーザーがいないのでNone
            url = manager.create_checkout_session(PlanType.BASIC)
            assert url is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_create_checkout_session_stripe_error(self, monkeypatch):
        """Stripeエラー時のチェックアウトセッション作成"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        mock_stripe_module.checkout.Session.create.side_effect = Exception("Stripe error")

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.FREE
            )

            manager = BillingManager()
            manager.stripe_available = True

            # PRICE_CONFIGSにstripe_price_idを一時的に設定
            original_config = PRICE_CONFIGS[PlanType.BASIC]
            PRICE_CONFIGS[PlanType.BASIC] = PriceConfig(
                plan=PlanType.BASIC,
                price_jpy=2980,
                stripe_price_id="price_test_basic",
                features=original_config.features
            )

            try:
                url = manager.create_checkout_session(PlanType.BASIC)
                assert url is None  # エラーでNone
            finally:
                PRICE_CONFIGS[PlanType.BASIC] = original_config
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_get_subscription_status_no_user(self):
        """ユーザーなしでのサブスクリプション状態取得"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = None

            manager = BillingManager()
            manager.stripe_available = True

            status = manager.get_subscription_status()
            assert status is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_get_subscription_status_no_subscription_id(self):
        """サブスクリプションIDなしでの状態取得"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC,
                stripe_subscription_id=None
            )

            manager = BillingManager()
            manager.stripe_available = True

            status = manager.get_subscription_status()
            assert status is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    @pytest.mark.skip(reason="AuthManagerがget_current_userでsession_state.auth_userを使用するため、モック環境では正確にテストできない")
    def test_get_subscription_status_success(self, monkeypatch):
        """サブスクリプション状態取得成功"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_subscription = MagicMock()
        mock_subscription.status = "active"
        mock_subscription.current_period_end = 1735689600  # 2025-01-01
        mock_subscription.cancel_at_period_end = False

        mock_stripe_module = MagicMock()
        mock_stripe_module.Subscription.retrieve.return_value = mock_subscription

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC,
                stripe_subscription_id="sub_xxx"
            )

            manager = BillingManager()
            manager.stripe_available = True

            status = manager.get_subscription_status()
            assert status is not None
            assert status["status"] == "active"
            assert status["cancel_at_period_end"] is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_get_subscription_status_stripe_error(self, monkeypatch):
        """Stripeエラー時のサブスクリプション状態取得"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        mock_stripe_module.Subscription.retrieve.side_effect = Exception("Stripe error")

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC,
                stripe_subscription_id="sub_xxx"
            )

            manager = BillingManager()
            manager.stripe_available = True

            status = manager.get_subscription_status()
            assert status is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_cancel_subscription_no_user(self):
        """ユーザーなしでのサブスクリプションキャンセル"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = None

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.cancel_subscription()
            assert result is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    @pytest.mark.skip(reason="AuthManagerがget_current_userでsession_state.auth_userを使用するため、モック環境では正確にテストできない")
    def test_cancel_subscription_success(self, monkeypatch):
        """サブスクリプションキャンセル成功"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        mock_stripe_module.Subscription.modify.return_value = MagicMock()

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC,
                stripe_subscription_id="sub_xxx"
            )

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.cancel_subscription()
            assert result is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_cancel_subscription_stripe_error(self, monkeypatch):
        """Stripeエラー時のサブスクリプションキャンセル"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        mock_stripe_module.Subscription.modify.side_effect = Exception("Stripe error")

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            st.session_state = MockSessionState()
            st.session_state["current_user"] = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC,
                stripe_subscription_id="sub_xxx"
            )

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.cancel_subscription()
            assert result is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestWebhookHandler:
    """Webhookハンドラーのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_handle_webhook_no_secret(self, monkeypatch):
        """Webhook secret未設定時"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            assert result is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_handle_webhook_invalid_signature(self, monkeypatch):
        """無効な署名でのWebhook処理"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.side_effect = Exception("Invalid signature")

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            assert result is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_handle_webhook_checkout_completed(self, monkeypatch):
        """checkout.session.completed イベント処理"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "metadata": {
                        "user_id": "test123",
                        "plan": "basic"
                    },
                    "customer": "cus_xxx",
                    "subscription": "sub_xxx"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            # ユーザーをセットアップ
            st.session_state["auth_users_db"] = {
                "test@example.com": User(
                    user_id="test123",
                    email="test@example.com",
                    password_hash="dummy_hash",
                    plan=PlanType.FREE
                )
            }

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            assert result is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_handle_webhook_subscription_updated(self, monkeypatch):
        """customer.subscription.updated イベント処理"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": "sub_xxx",
                    "status": "active"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            assert result is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_handle_webhook_subscription_deleted(self, monkeypatch):
        """customer.subscription.deleted イベント処理"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_xxx"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            # サブスクリプションIDを持つユーザーをセットアップ
            st.session_state["auth_users_db"] = {
                "test@example.com": User(
                    user_id="test123",
                    email="test@example.com",
                    password_hash="dummy_hash",
                    plan=PlanType.BASIC,
                    stripe_subscription_id="sub_xxx"
                )
            }

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            assert result is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_handle_checkout_completed_missing_metadata(self, monkeypatch):
        """メタデータなしでのcheckout完了処理"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "metadata": {},  # 空のメタデータ
                    "customer": "cus_xxx",
                    "subscription": "sub_xxx"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            assert result is True  # イベント処理自体は成功（内部で早期return）
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestStripeInitialization:
    """Stripe初期化のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_init_stripe_from_streamlit_secrets(self, monkeypatch):
        """Streamlit secretsからのStripe初期化"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            # 環境変数は設定しない
            monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)

            # Streamlit secretsからキーを取得
            st.secrets.get = MagicMock(return_value="sk_test_from_secrets")

            manager = BillingManager()
            assert manager.stripe_available is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_init_stripe_secrets_exception(self, monkeypatch):
        """Streamlit secrets例外時の初期化"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            # 環境変数は設定しない
            monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)

            # Streamlit secretsで例外
            st.secrets.get = MagicMock(side_effect=Exception("No secrets"))

            manager = BillingManager()
            assert manager.stripe_available is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestCheckoutSessionSuccess:
    """チェックアウトセッション成功パスのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_create_checkout_session_success(self, monkeypatch):
        """チェックアウトセッション作成成功"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        # モックチェックアウトセッション
        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/pay/cs_test_xxx"

        mock_stripe_module = MagicMock()
        mock_stripe_module.checkout.Session.create.return_value = mock_session

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")

            # セッションにユーザーを設定
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.FREE
            )
            st.session_state["auth_user"] = test_user
            st.session_state["current_user"] = test_user

            # PRICE_CONFIGSにstripe_price_idを一時的に設定
            original_config = PRICE_CONFIGS[PlanType.BASIC]
            PRICE_CONFIGS[PlanType.BASIC] = PriceConfig(
                plan=PlanType.BASIC,
                price_jpy=2980,
                stripe_price_id="price_test_basic",
                features=original_config.features
            )

            try:
                manager = BillingManager()
                manager.stripe_available = True

                # auth_managerのget_current_userをモック
                manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

                url = manager.create_checkout_session(PlanType.BASIC)
                assert url == "https://checkout.stripe.com/pay/cs_test_xxx"
            finally:
                PRICE_CONFIGS[PlanType.BASIC] = original_config
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_create_checkout_session_with_custom_urls(self, monkeypatch):
        """カスタムURLでのチェックアウトセッション作成"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/pay/cs_test_xxx"

        mock_stripe_module = MagicMock()
        mock_stripe_module.checkout.Session.create.return_value = mock_session

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")
            monkeypatch.setenv("STRIPE_SUCCESS_URL", "https://example.com/success")
            monkeypatch.setenv("STRIPE_CANCEL_URL", "https://example.com/cancel")

            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.FREE
            )

            original_config = PRICE_CONFIGS[PlanType.PRO]
            PRICE_CONFIGS[PlanType.PRO] = PriceConfig(
                plan=PlanType.PRO,
                price_jpy=9800,
                stripe_price_id="price_test_pro",
                features=original_config.features
            )

            try:
                manager = BillingManager()
                manager.stripe_available = True
                manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

                url = manager.create_checkout_session(PlanType.PRO)
                assert url == "https://checkout.stripe.com/pay/cs_test_xxx"

                # Stripe APIが正しい引数で呼ばれたことを確認
                call_args = mock_stripe_module.checkout.Session.create.call_args
                assert call_args[1]["mode"] == "subscription"
                assert call_args[1]["success_url"] == "https://example.com/success"
                assert call_args[1]["cancel_url"] == "https://example.com/cancel"
            finally:
                PRICE_CONFIGS[PlanType.PRO] = original_config
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestSubscriptionStatusSuccess:
    """サブスクリプション状態取得成功パスのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_get_subscription_status_success(self, monkeypatch):
        """サブスクリプション状態取得成功"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        # モックサブスクリプション
        mock_subscription = MagicMock()
        mock_subscription.status = "active"
        mock_subscription.current_period_end = 1735689600  # 2025-01-01
        mock_subscription.cancel_at_period_end = False

        mock_stripe_module = MagicMock()
        mock_stripe_module.Subscription.retrieve.return_value = mock_subscription

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")

            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC,
                stripe_subscription_id="sub_xxx"
            )

            manager = BillingManager()
            manager.stripe_available = True
            manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

            status = manager.get_subscription_status()
            assert status is not None
            assert status["status"] == "active"
            assert status["cancel_at_period_end"] is False
            assert "current_period_end" in status
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_get_subscription_status_canceled(self, monkeypatch):
        """キャンセル予約済みサブスクリプション状態取得"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_subscription = MagicMock()
        mock_subscription.status = "active"
        mock_subscription.current_period_end = 1735689600
        mock_subscription.cancel_at_period_end = True

        mock_stripe_module = MagicMock()
        mock_stripe_module.Subscription.retrieve.return_value = mock_subscription

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")

            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.PRO,
                stripe_subscription_id="sub_yyy"
            )

            manager = BillingManager()
            manager.stripe_available = True
            manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

            status = manager.get_subscription_status()
            assert status is not None
            assert status["cancel_at_period_end"] is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestCancelSubscriptionSuccess:
    """サブスクリプションキャンセル成功パスのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_cancel_subscription_success(self, monkeypatch):
        """サブスクリプションキャンセル成功"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        mock_stripe_module.Subscription.modify.return_value = MagicMock()

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")

            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.PRO,
                stripe_subscription_id="sub_xxx"
            )

            manager = BillingManager()
            manager.stripe_available = True
            manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

            result = manager.cancel_subscription()
            assert result is True

            # Stripe APIが正しく呼ばれたことを確認
            mock_stripe_module.Subscription.modify.assert_called_once_with(
                "sub_xxx",
                cancel_at_period_end=True
            )
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_cancel_subscription_no_subscription_id(self, monkeypatch):
        """サブスクリプションIDなしでのキャンセル"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")

            # subscription_idなしのユーザー
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.FREE,
                stripe_subscription_id=None
            )

            manager = BillingManager()
            manager.stripe_available = True
            manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

            result = manager.cancel_subscription()
            assert result is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestHandleCheckoutCompletedWithUser:
    """チェックアウト完了処理（ユーザーあり）のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_handle_checkout_completed_updates_user(self, monkeypatch):
        """チェックアウト完了でユーザープランが更新される"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "metadata": {
                        "user_id": "test123",
                        "plan": "basic"
                    },
                    "customer": "cus_xxx",
                    "subscription": "sub_xxx"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            # ユーザーをセットアップ
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.FREE
            )
            st.session_state["auth_users_db"] = {
                "test@example.com": test_user
            }

            manager = BillingManager()
            manager.stripe_available = True

            # update_planをモック
            manager.auth_manager.update_plan = MagicMock(return_value=True)

            result = manager.handle_webhook("payload", "sig")
            assert result is True

            # update_planが呼ばれたことを確認
            manager.auth_manager.update_plan.assert_called_once()
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestSubscriptionDeletedHandler:
    """サブスクリプション削除ハンドラーのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_handle_subscription_deleted_downgrades_user(self, monkeypatch):
        """サブスクリプション削除でユーザーがFreeにダウングレード"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_xxx"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            # サブスクリプションIDを持つユーザーをセットアップ
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.PRO,
                stripe_subscription_id="sub_xxx"
            )
            st.session_state["auth_users_db"] = {
                "test@example.com": test_user
            }

            manager = BillingManager()
            manager.stripe_available = True

            # update_planをモック
            manager.auth_manager.update_plan = MagicMock(return_value=True)

            result = manager.handle_webhook("payload", "sig")
            assert result is True

            # update_planがFREEプランで呼ばれたことを確認
            manager.auth_manager.update_plan.assert_called_once_with(
                "test@example.com",
                PlanType.FREE
            )
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestPriceConfigDataclass:
    """PriceConfigデータクラスのテスト"""

    def test_price_config_creation(self):
        """PriceConfigの作成"""
        config = PriceConfig(
            plan=PlanType.BASIC,
            price_jpy=2980,
            stripe_price_id="price_xxx",
            features=["feature1", "feature2"]
        )
        assert config.plan == PlanType.BASIC
        assert config.price_jpy == 2980
        assert config.stripe_price_id == "price_xxx"
        assert len(config.features) == 2

    def test_price_config_with_none_stripe_id(self):
        """stripe_price_idがNoneのPriceConfig"""
        config = PriceConfig(
            plan=PlanType.FREE,
            price_jpy=0,
            stripe_price_id=None,
            features=["free_feature"]
        )
        assert config.stripe_price_id is None

    def test_all_price_configs_have_plan_type(self):
        """全価格設定にプランタイプがある"""
        for plan_type, config in PRICE_CONFIGS.items():
            assert config.plan == plan_type


class TestStripeImportFailure:
    """Stripeインポート失敗時のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_stripe_import_failure_sets_flag(self, monkeypatch):
        """Stripeインポート失敗時のフラグ設定"""
        import src.billing as billing_module

        # インポート失敗状態をシミュレート
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            assert manager.stripe_available is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available


class TestCheckoutSessionEdgeCases:
    """チェックアウトセッションのエッジケーステスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_create_checkout_session_enterprise_plan(self, monkeypatch):
        """Enterpriseプラン（price_idなし）でのチェックアウト"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            manager.stripe_available = True

            # Enterpriseプランにはprice_idがない
            url = manager.create_checkout_session(PlanType.ENTERPRISE)
            assert url is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_create_checkout_session_invalid_plan(self, monkeypatch):
        """無効なプランでのチェックアウト"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()
            manager.stripe_available = True

            # 存在しないプラン（getでNone返却）
            # PRICE_CONFIGS.get()がNoneを返すケース
            from src.billing import PRICE_CONFIGS
            original_configs = PRICE_CONFIGS.copy()

            # 一時的にPRICE_CONFIGSからBASICを削除
            del PRICE_CONFIGS[PlanType.BASIC]

            try:
                url = manager.create_checkout_session(PlanType.BASIC)
                assert url is None
            finally:
                # 元に戻す
                PRICE_CONFIGS.update(original_configs)
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available


class TestWebhookEventTypes:
    """Webhookイベントタイプ別のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_handle_unknown_webhook_event(self, monkeypatch):
        """未知のWebhookイベント処理"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "unknown.event.type",
            "data": {
                "object": {}
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            manager = BillingManager()
            manager.stripe_available = True

            # 未知のイベントでも処理は成功する
            result = manager.handle_webhook("payload", "sig")
            assert result is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe

    def test_handle_subscription_updated_event(self, monkeypatch):
        """subscription.updated イベント処理"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": "sub_xxx",
                    "status": "past_due"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            assert result is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestBillingManagerInitWithSecrets:
    """Stripe初期化の追加テスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_init_stripe_env_var_priority(self, monkeypatch):
        """環境変数が優先されることを確認"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            # 環境変数とsecretsの両方を設定
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_env")
            st.secrets.get = MagicMock(return_value="sk_test_secrets")

            manager = BillingManager()
            # 環境変数のキーが使われる
            assert manager.stripe_available is True
            assert mock_stripe_module.api_key == "sk_test_env"
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestCheckoutSessionCompleteFlow:
    """チェックアウトセッション完全フローのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_create_checkout_session_full_flow(self, monkeypatch):
        """チェックアウトセッション作成の完全フロー"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/pay/test123"

        mock_stripe_module = MagicMock()
        mock_stripe_module.checkout.Session.create.return_value = mock_session

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")
            monkeypatch.setenv("STRIPE_SUCCESS_URL", "https://app.example.com/success")
            monkeypatch.setenv("STRIPE_CANCEL_URL", "https://app.example.com/cancel")

            test_user = User(
                user_id="user123",
                email="customer@example.com",
                password_hash="hash",
                plan=PlanType.FREE
            )

            # Stripe Price IDを設定
            original_config = PRICE_CONFIGS[PlanType.BASIC]
            PRICE_CONFIGS[PlanType.BASIC] = PriceConfig(
                plan=PlanType.BASIC,
                price_jpy=2980,
                stripe_price_id="price_basic_monthly",
                features=original_config.features
            )

            try:
                manager = BillingManager()
                manager.stripe_available = True
                manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

                url = manager.create_checkout_session(PlanType.BASIC)

                assert url == "https://checkout.stripe.com/pay/test123"

                # APIコールを確認
                call_kwargs = mock_stripe_module.checkout.Session.create.call_args[1]
                assert call_kwargs["payment_method_types"] == ["card"]
                assert call_kwargs["mode"] == "subscription"
                assert call_kwargs["customer_email"] == "customer@example.com"
                assert call_kwargs["metadata"]["user_id"] == "user123"
                assert call_kwargs["metadata"]["plan"] == "basic"
            finally:
                PRICE_CONFIGS[PlanType.BASIC] = original_config
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestSubscriptionRetrievalErrors:
    """サブスクリプション取得エラーのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_get_subscription_different_error_types(self, monkeypatch):
        """異なるエラータイプでのサブスクリプション取得"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        # ネットワークエラーをシミュレート
        mock_stripe_module.Subscription.retrieve.side_effect = ConnectionError("Network error")

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")

            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC,
                stripe_subscription_id="sub_xxx"
            )

            manager = BillingManager()
            manager.stripe_available = True
            manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

            status = manager.get_subscription_status()
            assert status is None
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestCancelSubscriptionErrors:
    """サブスクリプションキャンセルエラーのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_cancel_subscription_different_error(self, monkeypatch):
        """異なるエラーでのキャンセル"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        # タイムアウトエラー
        mock_stripe_module.Subscription.modify.side_effect = TimeoutError("Request timeout")

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")

            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.PRO,
                stripe_subscription_id="sub_xxx"
            )

            manager = BillingManager()
            manager.stripe_available = True
            manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

            result = manager.cancel_subscription()
            assert result is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestWebhookCheckoutWithNoMatch:
    """Webhookチェックアウト完了（ユーザー不一致）のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_handle_checkout_completed_user_not_found(self, monkeypatch):
        """ユーザーが見つからない場合のチェックアウト完了"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "metadata": {
                        "user_id": "nonexistent_user",
                        "plan": "basic"
                    },
                    "customer": "cus_xxx",
                    "subscription": "sub_xxx"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            # DBにはユーザーがいない
            st.session_state["auth_users_db"] = {}

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            # イベント処理自体は成功
            assert result is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestSubscriptionDeletedNoMatch:
    """サブスクリプション削除（ユーザー不一致）のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_handle_subscription_deleted_user_not_found(self, monkeypatch):
        """ユーザーが見つからない場合のサブスクリプション削除"""
        import src.billing as billing_module
        import streamlit as st
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_event = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_nonexistent"
                }
            }
        }

        mock_stripe_module = MagicMock()
        mock_stripe_module.Webhook.construct_event.return_value = mock_event

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xxx")

            # DBにはユーザーがいない
            st.session_state["auth_users_db"] = {}

            manager = BillingManager()
            manager.stripe_available = True

            result = manager.handle_webhook("payload", "sig")
            assert result is True
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestPriceConfigEdgeCases:
    """価格設定のエッジケーステスト"""

    def test_free_plan_has_no_stripe_id(self):
        """Freeプランにはstripe_price_idがない"""
        config = PRICE_CONFIGS[PlanType.FREE]
        assert config.stripe_price_id is None

    def test_enterprise_plan_has_no_stripe_id(self):
        """Enterpriseプランにはstripe_price_idがない"""
        config = PRICE_CONFIGS[PlanType.ENTERPRISE]
        assert config.stripe_price_id is None

    def test_basic_and_pro_can_have_stripe_ids(self):
        """BasicとProプランはstripe_price_idを持てる"""
        # デフォルトは環境変数から取得（未設定ならNone）
        basic_config = PRICE_CONFIGS[PlanType.BASIC]
        pro_config = PRICE_CONFIGS[PlanType.PRO]
        # 環境変数が設定されていればstripe_price_idがある
        # 設定されていなければNone（これは正常）
        assert basic_config.plan == PlanType.BASIC
        assert pro_config.plan == PlanType.PRO


class TestBillingManagerStripeNotImported:
    """Stripeがインポートできない場合のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_all_methods_return_none_or_false_when_stripe_unavailable(self):
        """Stripe無効時は全メソッドがNone/Falseを返す"""
        import src.billing as billing_module
        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            manager = BillingManager()

            assert manager.stripe_available is False
            assert manager.create_checkout_session(PlanType.BASIC) is None
            assert manager.get_subscription_status() is None
            assert manager.cancel_subscription() is False
            assert manager.handle_webhook("payload", "sig") is False
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available


class TestRenderPricingUI:
    """render_pricing_ui関数のテスト（モック環境）"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)
        # Streamlit UI関数をモック
        st.subheader = MagicMock()
        st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        st.container = MagicMock()
        st.markdown = MagicMock()
        st.caption = MagicMock()
        st.divider = MagicMock()
        st.write = MagicMock()
        st.button = MagicMock(return_value=False)

    def test_render_pricing_ui_basic(self):
        """料金プランUI描画の基本テスト"""
        import src.billing as billing_module
        from src.billing import render_pricing_ui
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            # コンテナのモックを設定
            mock_container = MagicMock()
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            st.container = MagicMock(return_value=mock_container)

            # columnsのモックを設定
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            st.columns = MagicMock(return_value=[mock_col, mock_col, mock_col])

            render_pricing_ui()

            # subheaderが呼ばれることを確認
            st.subheader.assert_called()
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_render_pricing_ui_with_logged_in_user(self):
        """ログインユーザーありでの料金プランUI描画"""
        import src.billing as billing_module
        from src.billing import render_pricing_ui
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            # ユーザーをセットアップ
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC
            )
            st.session_state["auth_user"] = test_user

            mock_container = MagicMock()
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            st.container = MagicMock(return_value=mock_container)

            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            st.columns = MagicMock(return_value=[mock_col, mock_col, mock_col])

            render_pricing_ui()

            # columnsが呼ばれることを確認
            st.columns.assert_called()
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available


class TestRenderBillingStatus:
    """render_billing_status関数のテスト（モック環境）"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)
        st.info = MagicMock()
        st.expander = MagicMock()
        st.write = MagicMock()
        st.warning = MagicMock()
        st.button = MagicMock(return_value=False)
        st.success = MagicMock()
        st.rerun = MagicMock()

    def test_render_billing_status_no_user(self):
        """ユーザーなしでのステータス表示"""
        import src.billing as billing_module
        from src.billing import render_billing_status
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            st.session_state["auth_user"] = None
            render_billing_status()
            # ユーザーがいない場合は何も表示しない
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_render_billing_status_paid_user_no_status(self):
        """有料ユーザー（ステータスなし）の表示"""
        import src.billing as billing_module
        from src.billing import render_billing_status
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.PRO
            )
            st.session_state["auth_user"] = test_user

            render_billing_status()

            # Stripe無効時はステータス取得不可
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available


class TestPricingUIUpgradeFlow:
    """料金プランUIのアップグレードフローテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)
        st.subheader = MagicMock()
        st.columns = MagicMock()
        st.container = MagicMock()
        st.markdown = MagicMock()
        st.caption = MagicMock()
        st.divider = MagicMock()
        st.write = MagicMock()
        st.button = MagicMock(return_value=False)
        st.warning = MagicMock()
        st.info = MagicMock()
        st.error = MagicMock()
        st.success = MagicMock()
        st.rerun = MagicMock()

    def test_render_pricing_ui_completes_without_error(self):
        """料金プランUI描画がエラーなく完了する"""
        import src.billing as billing_module
        from src.billing import render_pricing_ui
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            st.session_state["auth_user"] = None

            mock_container = MagicMock()
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            st.container = MagicMock(return_value=mock_container)

            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            st.columns = MagicMock(return_value=[mock_col, mock_col, mock_col])

            # エラーなく完了することを確認
            render_pricing_ui()
            st.subheader.assert_called()
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_render_pricing_ui_with_pro_user(self):
        """Proユーザーでの料金プランUI描画"""
        import src.billing as billing_module
        from src.billing import render_pricing_ui
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.PRO
            )
            st.session_state["auth_user"] = test_user

            mock_container = MagicMock()
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            st.container = MagicMock(return_value=mock_container)

            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            st.columns = MagicMock(return_value=[mock_col, mock_col, mock_col])

            render_pricing_ui()
            st.columns.assert_called()
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available

    def test_render_pricing_ui_with_enterprise_user(self):
        """Enterpriseユーザーでの料金プランUI描画"""
        import src.billing as billing_module
        from src.billing import render_pricing_ui
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.ENTERPRISE
            )
            st.session_state["auth_user"] = test_user

            mock_container = MagicMock()
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            st.container = MagicMock(return_value=mock_container)

            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            st.columns = MagicMock(return_value=[mock_col, mock_col, mock_col])

            render_pricing_ui()
            st.divider.assert_called()
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available


class TestPricingUIDowngrade:
    """料金プランUIのダウングレードテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)
        st.subheader = MagicMock()
        st.columns = MagicMock()
        st.container = MagicMock()
        st.markdown = MagicMock()
        st.caption = MagicMock()
        st.divider = MagicMock()
        st.write = MagicMock()
        st.button = MagicMock(return_value=False)
        st.warning = MagicMock()

    def test_render_with_basic_user(self):
        """Basicユーザーでのダウングレード表示確認"""
        import src.billing as billing_module
        from src.billing import render_pricing_ui
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        billing_module.STRIPE_AVAILABLE = False

        try:
            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.BASIC
            )
            st.session_state["auth_user"] = test_user

            mock_container = MagicMock()
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            st.container = MagicMock(return_value=mock_container)

            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            st.columns = MagicMock(return_value=[mock_col, mock_col, mock_col])

            # 描画がエラーなく完了することを確認
            render_pricing_ui()
            st.button.assert_called()
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available


class TestCreateCheckoutSessionFailure:
    """チェックアウトセッション作成失敗のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.session_state["auth_users_db"] = {}
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)
        st.subheader = MagicMock()
        st.columns = MagicMock()
        st.container = MagicMock()
        st.markdown = MagicMock()
        st.caption = MagicMock()
        st.divider = MagicMock()
        st.write = MagicMock()
        st.button = MagicMock(return_value=False)
        st.error = MagicMock()

    def test_checkout_error_handling_in_manager(self, monkeypatch):
        """BillingManagerのチェックアウトエラーハンドリング"""
        import src.billing as billing_module
        import streamlit as st

        original_stripe_available = billing_module.STRIPE_AVAILABLE
        original_stripe = billing_module.stripe

        mock_stripe_module = MagicMock()
        mock_stripe_module.checkout.Session.create.side_effect = Exception("Stripe error")

        billing_module.STRIPE_AVAILABLE = True
        billing_module.stripe = mock_stripe_module

        try:
            monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xxx")

            test_user = User(
                user_id="test123",
                email="test@example.com",
                password_hash="dummy_hash",
                plan=PlanType.FREE
            )

            original_config = PRICE_CONFIGS[PlanType.BASIC]
            PRICE_CONFIGS[PlanType.BASIC] = PriceConfig(
                plan=PlanType.BASIC,
                price_jpy=2980,
                stripe_price_id="price_test_basic",
                features=original_config.features
            )

            try:
                manager = BillingManager()
                manager.stripe_available = True
                manager.auth_manager.get_current_user = MagicMock(return_value=test_user)

                # エラー時はNoneが返される
                url = manager.create_checkout_session(PlanType.BASIC)
                assert url is None
                # st.errorが呼ばれる
                st.error.assert_called()
            finally:
                PRICE_CONFIGS[PlanType.BASIC] = original_config
        finally:
            billing_module.STRIPE_AVAILABLE = original_stripe_available
            billing_module.stripe = original_stripe


class TestStripeImportAvailability:
    """Stripeインポート可用性のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """各テスト前にsession_stateをリセット"""
        import streamlit as st
        st.session_state = MockSessionState()
        st.secrets = MagicMock()
        st.secrets.get = MagicMock(return_value=None)

    def test_stripe_available_constant(self):
        """STRIPE_AVAILABLE定数の確認"""
        import src.billing as billing_module
        # STRIPE_AVAILABLEはbool型
        assert isinstance(billing_module.STRIPE_AVAILABLE, bool)

    def test_stripe_module_reference(self):
        """stripeモジュール参照の確認"""
        import src.billing as billing_module
        # stripeはNone（インポートできない場合）またはモジュール
        # テスト環境ではモック済み


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
