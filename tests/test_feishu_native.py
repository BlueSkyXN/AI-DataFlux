import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.data.feishu.client import FeishuClient, FeishuAPIError

# ==================== Mocks ====================

@pytest.fixture
def mock_aioresponse():
    """Mock aiohttp responses"""
    with patch("aiohttp.ClientSession.request") as mock_request:
        yield mock_request

@pytest.fixture
def mock_post():
    """Mock aiohttp post"""
    with patch("aiohttp.ClientSession.post") as mock_post:
        yield mock_post

# ==================== Tests ====================

@pytest.mark.asyncio
async def test_ensure_token_success(mock_post):
    """Test token acquisition success"""
    # Setup mock response
    mock_resp = MagicMock()
    mock_resp.status = 200

    # Mock async json()
    f = asyncio.Future()
    f.set_result({
        "code": 0,
        "tenant_access_token": "fake_token_123",
        "expire": 7200
    })
    mock_resp.json.return_value = f

    # Context manager mock
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_resp
    mock_ctx.__aexit__.return_value = None
    mock_post.return_value = mock_ctx

    client = FeishuClient(app_id="cli_123", app_secret="sec_123")
    token = await client.ensure_token()

    assert token == "fake_token_123"
    assert client._token == "fake_token_123"
    await client.close()

@pytest.mark.asyncio
async def test_ensure_token_failure(mock_post):
    """Test token acquisition failure"""
    mock_resp = MagicMock()
    mock_resp.status = 200

    f = asyncio.Future()
    f.set_result({
        "code": 1000,
        "msg": "Invalid app_id"
    })
    mock_resp.json.return_value = f

    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_resp
    mock_ctx.__aexit__.return_value = None
    mock_post.return_value = mock_ctx

    client = FeishuClient(app_id="cli_123", app_secret="sec_123")

    with pytest.raises(FeishuAPIError) as exc:
        await client.ensure_token()

    assert exc.value.code == 1000
    await client.close()

@pytest.mark.asyncio
async def test_batch_update_concurrency(mock_post, mock_aioresponse):
    """Test batch update concurrency logic"""
    # 1. Mock Token
    token_resp = MagicMock()
    f_token = asyncio.Future()
    f_token.set_result({"code": 0, "tenant_access_token": "t", "expire": 7200})
    token_resp.json.return_value = f_token

    token_ctx = MagicMock()
    token_ctx.__aenter__.return_value = token_resp
    token_ctx.__aexit__.return_value = None
    mock_post.return_value = token_ctx

    # 2. Mock Batch Request
    batch_resp = MagicMock()
    batch_resp.status = 200

    f_batch = asyncio.Future()
    f_batch.set_result({
        "code": 0,
        "data": {
            "records": [{"record_id": "r1"}]
        }
    })
    batch_resp.json.return_value = f_batch

    batch_ctx = MagicMock()
    batch_ctx.__aenter__.return_value = batch_resp
    batch_ctx.__aexit__.return_value = None
    mock_aioresponse.return_value = batch_ctx

    client = FeishuClient(
        app_id="id",
        app_secret="sec",
        concurrency=2 # Limit concurrency to 2
    )

    # Generate 2500 records (should split into 3 chunks: 1000, 1000, 500)
    records = [{"record_id": f"rec_{i}", "fields": {"f": 1}} for i in range(2500)]

    await client.bitable_batch_update("app_token", "table_id", records)

    # Verify calls
    # 1 call for token
    # 3 calls for batch update
    assert mock_aioresponse.call_count == 3
    await client.close()

if __name__ == "__main__":
    # Manually run if needed
    pass
