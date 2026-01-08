"""
InsightAnalyzerAI - 認証システムテスト

認証・プラン管理機能のテスト
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Streamlitのモック
import sys
sys.modules['streamlit'] = MagicMock()

from src.auth import (
    PlanType,
    PlanLimits,
    PLAN_LIMITS,
    User,
    UsageTracker,
    AuthManager,
)


class TestPlanLimits:
    """プラン制限のテスト"""

    def test_free_plan_limits(self):
        """Freeプランの制限"""
        limits = PLAN_LIMITS[PlanType.FREE]
        assert limits.max_file_size_mb == 1
        assert limits.daily_queries == 10
        assert limits.charts_enabled is False
        assert limits.llm_enabled is False

    def test_basic_plan_limits(self):
        """Basicプランの制限"""
        limits = PLAN_LIMITS[PlanType.BASIC]
        assert limits.max_file_size_mb == 50
        assert limits.daily_queries == 100
        assert limits.charts_enabled is True
        assert limits.llm_enabled is True

    def test_pro_plan_limits(self):
        """Proプランの制限"""
        limits = PLAN_LIMITS[PlanType.PRO]
        assert limits.max_file_size_mb == 500
        assert limits.daily_queries == 10000
        assert limits.charts_enabled is True
        assert limits.api_access is True


class TestUser:
    """Userデータクラスのテスト"""

    def test_user_creation(self):
        """ユーザー作成"""
        user = User(
            user_id="test123",
            email="test@example.com",
            password_hash="hash123",
        )
        assert user.user_id == "test123"
        assert user.email == "test@example.com"
        assert user.plan == PlanType.FREE  # デフォルト

    def test_user_with_plan(self):
        """プラン指定でユーザー作成"""
        user = User(
            user_id="pro123",
            email="pro@example.com",
            password_hash="hash456",
            plan=PlanType.PRO,
        )
        assert user.plan == PlanType.PRO


class TestAuthManager:
    """AuthManagerのテスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_password_hash(self, mock_session_state):
        """パスワードハッシュのテスト"""
        manager = AuthManager()
        hash1, salt1 = manager._hash_password("password123")
        hash2, _ = manager._hash_password("password123", salt1)

        # 同じソルトなら同じハッシュ
        assert hash1 == hash2

    def test_password_hash_different_salts(self, mock_session_state):
        """異なるソルトでは異なるハッシュ"""
        manager = AuthManager()
        hash1, salt1 = manager._hash_password("password123")
        hash2, salt2 = manager._hash_password("password123")

        # 異なるソルトなら異なるハッシュ
        if salt1 != salt2:
            assert hash1 != hash2

    def test_verify_password(self, mock_session_state):
        """パスワード検証"""
        manager = AuthManager()
        hash_value, salt = manager._hash_password("mypassword")
        stored_hash = f"{salt}:{hash_value}"

        assert manager._verify_password("mypassword", stored_hash) is True
        assert manager._verify_password("wrongpassword", stored_hash) is False

    def test_register_success(self, mock_session_state):
        """ユーザー登録成功"""
        manager = AuthManager()
        success, message = manager.register("new@example.com", "password123")

        assert success is True
        assert "完了" in message
        # 登録成功の確認: ログインできれば登録成功
        success2, _ = manager.login("new@example.com", "password123")
        assert success2 is True

    def test_register_invalid_email(self, mock_session_state):
        """無効なメールアドレスでの登録"""
        manager = AuthManager()
        success, message = manager.register("invalid-email", "password123")

        assert success is False
        assert "メールアドレス" in message

    def test_register_short_password(self, mock_session_state):
        """短いパスワードでの登録"""
        manager = AuthManager()
        success, message = manager.register("test@example.com", "short")

        assert success is False
        assert "8文字" in message

    def test_register_duplicate_email(self, mock_session_state):
        """重複メールアドレスでの登録"""
        manager = AuthManager()
        manager.register("dup@example.com", "password123")
        success, message = manager.register("dup@example.com", "password456")

        assert success is False
        assert "既に登録" in message

    def test_login_success(self, mock_session_state):
        """ログイン成功"""
        manager = AuthManager()
        manager.register("login@example.com", "password123")
        success, message = manager.login("login@example.com", "password123")

        assert success is True
        assert manager.is_authenticated() is True

    def test_login_wrong_password(self, mock_session_state):
        """間違ったパスワードでのログイン"""
        manager = AuthManager()
        manager.register("test@example.com", "password123")
        success, message = manager.login("test@example.com", "wrongpassword")

        assert success is False

    def test_login_nonexistent_user(self, mock_session_state):
        """存在しないユーザーでのログイン"""
        manager = AuthManager()
        success, message = manager.login("nonexistent@example.com", "password123")

        assert success is False

    def test_logout(self, mock_session_state):
        """ログアウト"""
        manager = AuthManager()
        manager.register("logout@example.com", "password123")
        manager.login("logout@example.com", "password123")

        assert manager.is_authenticated() is True

        manager.logout()

        assert manager.is_authenticated() is False

    def test_get_plan_limits_unauthenticated(self, mock_session_state):
        """未認証時のプラン制限"""
        manager = AuthManager()
        limits = manager.get_plan_limits()

        assert limits == PLAN_LIMITS[PlanType.FREE]

    def test_can_execute_query_within_limit(self, mock_session_state):
        """クエリ制限内での実行"""
        manager = AuthManager()
        manager.usage_tracker._init_session_state()

        can_execute, message = manager.can_execute_query()

        assert can_execute is True

    def test_can_upload_file_within_limit(self, mock_session_state):
        """ファイルサイズ制限内でのアップロード"""
        manager = AuthManager()

        # Freeプラン: 1MB制限
        can_upload, message = manager.can_upload_file(500 * 1024)  # 500KB

        assert can_upload is True

    def test_can_upload_file_exceeds_limit(self, mock_session_state):
        """ファイルサイズ制限超過"""
        manager = AuthManager()

        # Freeプラン: 1MB制限
        can_upload, message = manager.can_upload_file(2 * 1024 * 1024)  # 2MB

        assert can_upload is False
        assert "上限" in message

    def test_update_plan(self, mock_session_state):
        """プラン更新"""
        manager = AuthManager()
        manager.register("upgrade@example.com", "password123")
        manager.login("upgrade@example.com", "password123")

        success = manager.update_plan("upgrade@example.com", PlanType.PRO)

        assert success is True
        assert manager.get_current_user().plan == PlanType.PRO


class TestUsageTracker:
    """UsageTrackerのテスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_increment_query_count(self, mock_session_state):
        """クエリカウント増加"""
        tracker = UsageTracker()
        tracker._init_session_state()

        initial_count = tracker.get_query_count()
        tracker.increment_query_count()
        new_count = tracker.get_query_count()

        assert new_count == initial_count + 1

    def test_add_file_upload(self, mock_session_state):
        """ファイルアップロード記録"""
        tracker = UsageTracker()
        tracker._init_session_state()

        tracker.add_file_upload(1024)
        tracker.add_file_upload(2048)

        assert tracker.get_total_upload_size() == 3072


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
