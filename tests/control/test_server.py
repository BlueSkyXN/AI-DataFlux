from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.control.server import create_control_app

client = TestClient(create_control_app())

def test_read_config():
    response = client.get("/api/config?path=config.yaml")
    assert response.status_code == 200
    assert "content" in response.json()
    assert "path" in response.json()

def test_validate_config_valid():
    valid_yaml = """
    datasource:
      type: excel
    """
    response = client.post("/api/config/validate", json={"content": valid_yaml})
    assert response.status_code == 200
    assert response.json()["valid"] is True

def test_validate_config_invalid():
    invalid_yaml = """
    datasource:
      type: [unclosed list
    """
    response = client.post("/api/config/validate", json={"content": invalid_yaml})
    assert response.status_code == 200
    assert response.json()["valid"] is False

@patch("src.data.feishu.client.FeishuClient")
def test_feishu_test_connection_success(mock_feishu_client_cls):
    # Mock FeishuClient instance and its methods
    mock_client_instance = MagicMock()

    # ensure_token is async, so it must return an awaitable (or use AsyncMock)
    # Since we are patching the class, the instance methods are standard Mocks.
    # We can use AsyncMock for the instance.
    from unittest.mock import AsyncMock
    mock_client_instance.ensure_token = AsyncMock(return_value="fake_token_12345")
    mock_client_instance.close = AsyncMock(return_value=None)

    mock_feishu_client_cls.return_value = mock_client_instance

    response = client.post("/api/feishu/test_connection", json={
        "app_id": "cli_test_id",
        "app_secret": "test_secret"
    })

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert "Connected successfully" in result["message"]

    # Check if FeishuClient was called with correct args
    mock_feishu_client_cls.assert_called_with(
        app_id="cli_test_id",
        app_secret="test_secret",
        max_retries=0
    )

@patch("src.data.feishu.client.FeishuClient")
def test_feishu_test_connection_failure(mock_feishu_client_cls):
    # Mock exception
    from src.data.feishu.client import FeishuAPIError
    from unittest.mock import AsyncMock

    mock_client_instance = MagicMock()
    mock_client_instance.ensure_token = AsyncMock(side_effect=FeishuAPIError(code=9999, msg="Invalid secret"))
    mock_client_instance.close = AsyncMock(return_value=None)

    mock_feishu_client_cls.return_value = mock_client_instance

    response = client.post("/api/feishu/test_connection", json={
        "app_id": "cli_test_id",
        "app_secret": "wrong_secret"
    })

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is False
    assert "Invalid secret" in result["message"]
