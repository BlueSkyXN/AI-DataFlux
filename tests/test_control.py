"""
Control Server 测试

被测模块: src/control/config_api.py, src/control/server.py, src/control/process_manager.py

测试 Web GUI 控制面板的核心功能：
- 配置文件 API (读取/写入)
- 进程管理 (状态查询)
- 路径安全验证

测试类/函数清单:
    TestConfigAPI                              配置文件 API 测试
        test_validate_path_inside_project      验证项目内路径通过校验
        test_validate_path_traversal_blocked   验证路径穿越（../）被阻止并返回 403
        test_read_config_existing_file         验证读取存在的配置文件返回内容
        test_read_config_missing_file          验证读取不存在的文件返回 404
        test_read_config_blocks_non_yaml_file  验证读取非 YAML 文件返回 403
        test_write_config_creates_backup       验证写入前创建 .bak 备份文件
        test_write_config_blocks_non_yaml_file 验证写入非 YAML 文件返回 403
    TestControlServerAuth                      控制面板鉴权测试
        test_extract_bearer_token              验证 Bearer Token 解析
        test_authorized_token_validation       验证 Token 校验逻辑
        test_authorized_from_candidates        验证多来源 Token 任一通过即鉴权成功
        test_mask_token_for_logs               验证 Token 脱敏显示
        test_create_app_initializes_token      验证应用创建时初始化鉴权 Token
        test_decode_base64url_token            验证 base64url Token 解码
        test_extract_ws_token                  验证 WebSocket Token 提取（优先 b64）
        test_test_feishu_connection_success    验证飞书连接测试成功路径
        test_test_feishu_connection_api_error  验证飞书连接测试 API 错误路径
    TestProcessManager                         进程管理器测试
        test_initial_status                    验证初始状态为 stopped / pid=None
        test_get_all_status                    验证获取所有进程状态
        test_log_buffer                        验证日志缓冲区初始为空
    TestManagedProcess                         ManagedProcess 数据类测试
        test_to_dict                           验证转换为字典包含完整字段
        test_default_values                    验证默认值正确
    TestKillTree                               进程树清理测试
        test_kill_tree_nonexistent_process     验证清理不存在的进程不抛异常
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import base64


class TestConfigAPI:
    """配置文件 API 测试"""

    def test_validate_path_inside_project(self):
        """测试路径校验 - 项目内部路径"""
        from src.control.config_api import _validate_path, PROJECT_ROOT

        # 项目内部路径应该通过
        result = _validate_path("config-example.yaml")
        assert result.startswith(PROJECT_ROOT)

    def test_validate_path_traversal_blocked(self):
        """测试路径校验 - 阻止路径穿越"""
        from fastapi import HTTPException
        from src.control.config_api import _validate_path

        # 尝试访问项目外部应该被阻止
        with pytest.raises(HTTPException) as exc_info:
            _validate_path("../../../etc/passwd")

        assert exc_info.value.status_code == 403

    def test_read_config_existing_file(self):
        """测试读取存在的配置文件"""
        from src.control.config_api import read_config

        content = read_config("config-example.yaml")
        assert "global" in content
        assert "datasource" in content

    def test_read_config_missing_file(self):
        """测试读取不存在的配置文件"""
        from fastapi import HTTPException
        from src.control.config_api import read_config

        with pytest.raises(HTTPException) as exc_info:
            read_config("nonexistent.yaml")

        assert exc_info.value.status_code == 404

    def test_read_config_blocks_non_yaml_file(self, tmp_path):
        """测试读取非 YAML 文件会被阻止"""
        from fastapi import HTTPException
        from src.control.config_api import read_config

        test_file = tmp_path / "secrets.txt"
        test_file.write_text("secret")

        with patch("src.control.config_api.PROJECT_ROOT", str(tmp_path)):
            with pytest.raises(HTTPException) as exc_info:
                read_config("secrets.txt")

        assert exc_info.value.status_code == 403

    def test_write_config_creates_backup(self, tmp_path):
        """测试写入配置文件会创建备份"""
        from src.control.config_api import write_config

        # 使用临时目录中的文件
        test_file = tmp_path / "test_config.yaml"
        test_file.write_text("original: content")

        # Mock PROJECT_ROOT to use tmp_path
        with patch("src.control.config_api.PROJECT_ROOT", str(tmp_path)):
            result = write_config("test_config.yaml", "new: content")

        assert result["success"] is True
        assert result["backed_up"] is True
        assert (tmp_path / "test_config.yaml.bak").exists()
        assert (tmp_path / "test_config.yaml").read_text() == "new: content"

    def test_write_config_blocks_non_yaml_file(self, tmp_path):
        """测试写入非 YAML 文件会被阻止"""
        from fastapi import HTTPException
        from src.control.config_api import write_config

        with patch("src.control.config_api.PROJECT_ROOT", str(tmp_path)):
            with pytest.raises(HTTPException) as exc_info:
                write_config("malicious.py", "print('bad')")

        assert exc_info.value.status_code == 403


class TestControlServerAuth:
    """控制面板鉴权测试"""

    @pytest.fixture
    def auth_app(self, monkeypatch):
        from src.control import server as control_server

        monkeypatch.setenv("DATAFLUX_CONTROL_TOKEN", "unit-test-token")
        monkeypatch.setattr(control_server, "_CONTROL_AUTH_TOKEN", None)
        monkeypatch.setattr(control_server, "_CONTROL_AUTH_TOKEN_SOURCE", "env")
        return control_server

    def test_extract_bearer_token(self, auth_app):
        """测试 Bearer Token 解析"""
        assert auth_app._extract_bearer_token("Bearer abc123") == "abc123"
        assert auth_app._extract_bearer_token("Basic abc123") == ""

    def test_authorized_token_validation(self, auth_app):
        """测试 Token 校验"""
        assert auth_app._is_authorized_token("unit-test-token") is True
        assert auth_app._is_authorized_token("invalid-token") is False
        assert auth_app._is_authorized_token("") is False

    def test_authorized_from_candidates(self, auth_app):
        """测试多来源 token 任一通过即可鉴权成功"""
        assert (
            auth_app._is_authorized_from_candidates("invalid-token", "unit-test-token")
            is True
        )
        assert (
            auth_app._is_authorized_from_candidates("unit-test-token", "invalid-token")
            is True
        )
        assert (
            auth_app._is_authorized_from_candidates("invalid-token", "also-invalid")
            is False
        )

    def test_mask_token_for_logs(self, auth_app):
        """测试日志 token 脱敏"""
        assert auth_app._mask_token("abcdefghijkl") == "abc...jkl"
        assert auth_app._mask_token("short") == "***"

    def test_create_app_initializes_token(self, auth_app):
        """测试创建应用时会初始化鉴权 Token"""
        app = auth_app.create_control_app()
        assert app is not None
        assert auth_app.get_control_auth_token() == "unit-test-token"

    def test_decode_base64url_token(self, auth_app):
        """测试 base64url token 解码"""
        encoded = base64.urlsafe_b64encode("unit-test-token".encode("utf-8")).decode(
            "ascii"
        )
        encoded = encoded.rstrip("=")
        assert auth_app._decode_base64url_token(encoded) == "unit-test-token"
        assert auth_app._decode_base64url_token(encoded + "!") == ""
        assert auth_app._decode_base64url_token("bad%%%") == ""

    def test_extract_ws_token(self, auth_app):
        """测试 WebSocket token 提取（优先 b64）"""
        encoded = base64.urlsafe_b64encode("unit-test-token".encode("utf-8")).decode(
            "ascii"
        )
        encoded = encoded.rstrip("=")
        protocol, token = auth_app._extract_ws_token(
            f"chat, dataflux-token-b64.{encoded}, other"
        )
        assert protocol.startswith("dataflux-token-b64.")
        assert token == "unit-test-token"

    @pytest.mark.asyncio
    async def test_test_feishu_connection_success(self):
        """测试飞书连接测试成功路径"""
        from src.control.server import _test_feishu_connection

        with patch("src.data.feishu.client.FeishuClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.ensure_token = AsyncMock(return_value="token_1234567890")
            mock_client.close = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await _test_feishu_connection("cli_test", "sec_test")

            assert result["success"] is True
            assert result["message"] == "Connected successfully"
            assert "token_preview" in result
            mock_client_cls.assert_called_once_with(
                app_id="cli_test",
                app_secret="sec_test",
                max_retries=0,
            )
            mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_test_feishu_connection_api_error(self):
        """测试飞书连接测试 API 错误路径"""
        from src.control.server import _test_feishu_connection
        from src.data.feishu.client import FeishuAPIError

        with patch("src.data.feishu.client.FeishuClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.ensure_token = AsyncMock(
                side_effect=FeishuAPIError(code=1000, msg="invalid app_secret")
            )
            mock_client.close = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await _test_feishu_connection("cli_test", "wrong_secret")

            assert result["success"] is False
            assert "invalid app_secret" in result["message"]
            mock_client.close.assert_awaited_once()


class TestProcessManager:
    """进程管理器测试"""

    def test_initial_status(self):
        """测试初始状态"""
        from src.control.process_manager import ProcessManager

        manager = ProcessManager()

        gateway_status = manager.get_status("gateway")
        process_status = manager.get_status("process")

        assert gateway_status["status"] == "stopped"
        assert gateway_status["pid"] is None
        assert process_status["status"] == "stopped"
        assert process_status["pid"] is None

    def test_get_all_status(self):
        """测试获取所有状态"""
        from src.control.process_manager import ProcessManager

        manager = ProcessManager()
        all_status = manager.get_all_status()

        assert "gateway" in all_status
        assert "process" in all_status
        assert "working_directory" in all_status
        assert "managed" in all_status["gateway"]
        assert "managed" in all_status["process"]
        # 验证工作目录路径存在
        assert len(all_status["working_directory"]) > 0

    def test_log_buffer(self):
        """测试日志缓冲区"""
        from src.control.process_manager import ProcessManager

        manager = ProcessManager()

        # 初始应该为空
        gateway_logs = manager.get_log_buffer("gateway")
        assert gateway_logs == []

        process_logs = manager.get_log_buffer("process")
        assert process_logs == []


class TestManagedProcess:
    """ManagedProcess 数据类测试"""

    def test_to_dict(self):
        """测试转换为字典"""
        from src.control.process_manager import ManagedProcess

        proc = ManagedProcess(
            name="gateway",
            status="running",
            pid=12345,
            start_time=1707000000.0,
            config_path="config.yaml",
            port=8787,
        )

        result = proc.to_dict()

        assert result["status"] == "running"
        assert result["pid"] == 12345
        assert result["start_time"] == 1707000000.0
        assert result["config_path"] == "config.yaml"
        assert result["port"] == 8787

    def test_default_values(self):
        """测试默认值"""
        from src.control.process_manager import ManagedProcess

        proc = ManagedProcess(name="process")

        assert proc.status == "stopped"
        assert proc.pid is None
        assert proc.start_time is None
        assert proc.exit_code is None
        assert proc.command == []
        assert proc.config_path is None
        assert proc.port is None


class TestKillTree:
    """进程树清理测试"""

    def test_kill_tree_nonexistent_process(self):
        """测试清理不存在的进程"""
        from src.control.process_manager import kill_tree

        # 不应该抛出异常
        kill_tree(99999999)
