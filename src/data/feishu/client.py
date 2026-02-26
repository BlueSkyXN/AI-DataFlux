"""
飞书原生异步 HTTP 客户端

本模块实现 AI-DataFlux 专用的飞书 OpenAPI 客户端，不依赖飞书官方 SDK。
参考 XTF 项目的 api/auth.py 和 api/base.py 设计，使用 aiohttp 实现异步请求。

核心能力:
    1. tenant_access_token 自动获取与刷新（有效期 2 小时，提前 5 分钟刷新）
    2. 指数退避重试（429 限流、5xx 服务端错误、网络超时）
    3. 速率限制（可配置 QPS 上限）
    4. 统一错误分类与日志

API 端点:
    认证: POST https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal
    多维表格: https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records
    电子表格: https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{spreadsheet_token}/values

异常类清单:
    FeishuAPIError(Exception)         — 飞书 API 通用错误（含 code/msg/url）
    FeishuRateLimitError(FeishuAPIError) — 429 限流错误（附 retry_after 秒数）
    FeishuPermissionError(FeishuAPIError) — 权限不足错误（code 99991401/99991403）

类清单:
    FeishuClient
        飞书原生异步 HTTP 客户端，管理 Token、速率控制、重试与 HTTP 会话

关键常量:
    FEISHU_BASE_URL              — 飞书 OpenAPI 根地址
    TOKEN_REFRESH_MARGIN (300s)  — Token 提前刷新裕量（5 分钟）
    BITABLE_BATCH_UPDATE_LIMIT   — 多维表格批量更新上限（1000 条/次）
    BITABLE_LIST_PAGE_SIZE       — 多维表格列表分页大小（500 条/页）
    SHEET_MAX_ROWS_PER_READ      — 电子表格单次读取上限（5000 行）
    FEISHU_TOO_LARGE_CODES       — 请求/响应过大错误码 (90221, 90227)

关键变量:
    _token            — str|None: 当前 tenant_access_token
    _token_expires_at — float: Token 过期时间戳（Unix）
    _session          — aiohttp.ClientSession|None: 复用的 HTTP 会话
    _semaphore        — asyncio.Semaphore: 并发请求数限制信号量
    _rate_limit_lock  — asyncio.Lock: 速率控制互斥锁

方法清单:
    生命周期:
        _get_session() → ClientSession     — [async] 获取/创建 HTTP 会话（检测事件循环切换）
        close()                            — [async] 关闭 HTTP 会话

    Token 管理:
        ensure_token() → str               — [async] 确保 Token 有效，必要时刷新
        check_connection() → bool          — [async] 通过获取 Token 验证连接
        _auth_headers() → dict             — [async] 构建 Bearer 认证请求头

    底层请求:
        _request(method, url, ...) → dict  — [async] 带重试的 HTTP 请求（核心方法）
            重试策略: 429→读 Retry-After, 5xx→指数退避, Token 失效→刷新,
                     业务层频控(99991400)→退避, 网络错误→退避

    Bitable (多维表格) API:
        bitable_list_records(app_token, table_id, ...) → list  — [async] 拉取全部记录（自动翻页）
        bitable_batch_update(app_token, table_id, records) → list — [async] 批量更新（自动分块+并发）
        bitable_batch_create(app_token, table_id, records) → list — [async] 批量创建（自动分块+并发）
        _bitable_batch_with_split(url, records) → list          — [async] 执行批量操作，过大时栈式二分
        bitable_list_fields(app_token, table_id) → list         — [async] 列出字段定义
        bitable_get_meta(app_token) → dict                      — [async] 获取元数据

    Sheet (电子表格) API:
        sheet_get_meta(spreadsheet_token) → dict                — [async] 获取元数据（含工作表列表）
        sheet_read_range(spreadsheet_token, range_str) → list   — [async] 读取范围（过大时自动行二分）
        _sheet_read_single(spreadsheet_token, range_str) → list — [async] 读取单个范围
        _sheet_read_chunked(spreadsheet_token, range_str) → list — [async] 分块读取（行数减半策略）
        sheet_write_range(spreadsheet_token, range_str, values) → dict — [async] 写入范围（过大时栈式二分）
        _sheet_write_single(spreadsheet_token, range_str, values) → dict — [async] 写入单个范围
        sheet_get_info(spreadsheet_token) → dict                — [async] 获取电子表格基本信息

模块级辅助函数:
    _is_too_large_error(e) → bool         — 检测请求/响应过大错误（码+关键词双重匹配）
    _parse_range(range_str) → tuple|None  — 解析 "Sheet1!A1:Z1000" 为 (sheet_id, 起始列, 起始行, 结束列, 结束行)

模块依赖:
    asyncio, logging, re, time  — 标准库
    aiohttp                     — 异步 HTTP 客户端

使用示例:
    client = FeishuClient(app_id="cli_xxx", app_secret="xxx")
    await client.ensure_token()
    records = await client.bitable_list_records(app_token, table_id)
    await client.close()
"""

import asyncio
import logging
import re
import time
from typing import Any

import aiohttp

# ==================== 常量 ====================

FEISHU_BASE_URL = "https://open.feishu.cn/open-apis"
TOKEN_URL = f"{FEISHU_BASE_URL}/auth/v3/tenant_access_token/internal"

# Token 提前刷新裕量（秒）
TOKEN_REFRESH_MARGIN = 300  # 5 分钟

# 默认重试配置
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # 秒

# Bitable 批量操作上限（对齐官方文档）
BITABLE_BATCH_CREATE_LIMIT = 1000
BITABLE_BATCH_UPDATE_LIMIT = 1000
BITABLE_BATCH_DELETE_LIMIT = 500
BITABLE_LIST_PAGE_SIZE = 500

# Sheet 操作常量
SHEET_MAX_ROWS_PER_READ = 5000
SHEET_MAX_ROWS_PER_WRITE = 5000

# 请求/响应过大错误码（Sheet API 常见）
FEISHU_TOO_LARGE_CODES = (90221, 90227)
_TOO_LARGE_KEYWORDS = (
    "90221",
    "90227",
    "TooLargeResponse",
    "TooLargeRequest",
    "data exceeded",
)

# 范围字符串解析正则: "Sheet1!A1:Z1000"
_RANGE_RE = re.compile(r"^(.+?)!([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)$")


class FeishuAPIError(Exception):
    """飞书 API 调用错误"""

    def __init__(self, code: int, msg: str, url: str = ""):
        self.code = code
        self.msg = msg
        self.url = url
        super().__init__(f"飞书 API 错误 [{code}]: {msg} (URL: {url})")


class FeishuRateLimitError(FeishuAPIError):
    """飞书 429 限流错误"""

    def __init__(self, msg: str, retry_after: float = 1.0, url: str = ""):
        self.retry_after = retry_after
        super().__init__(code=429, msg=msg, url=url)


class FeishuPermissionError(FeishuAPIError):
    """飞书权限不足错误"""

    pass


class FeishuClient:
    """
    飞书原生异步 HTTP 客户端

    Attributes:
        app_id: 飞书应用 ID
        app_secret: 飞书应用密钥
        max_retries: 最大重试次数
        qps_limit: 每秒最大请求数（0 表示不限制）
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        qps_limit: float = 0,
        concurrency: int = 5,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.max_retries = max_retries
        self.qps_limit = qps_limit
        self.concurrency = concurrency

        # Token 状态
        self._token: str | None = None
        self._token_expires_at: float = 0  # Unix 时间戳

        # 速率控制
        self._last_request_time: float = 0
        self._min_interval = 1.0 / qps_limit if qps_limit > 0 else 0
        self._rate_limit_lock = asyncio.Lock()  # 保护速率控制状态
        self._semaphore = asyncio.Semaphore(concurrency) # 限制并发请求数

        # HTTP 会话（延迟创建）
        self._session: aiohttp.ClientSession | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None

        self._logger = logging.getLogger("feishu.client")

    # ==================== 生命周期 ====================

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        current_loop = asyncio.get_running_loop()

        # asyncio.run() 会为每次调用创建新事件循环。
        # 复用旧 loop 上创建的 ClientSession 会触发 RuntimeError（如 Event loop is closed）。
        if self._session and not self._session.closed:
            if self._session_loop is current_loop:
                return self._session

            self._logger.warning("检测到事件循环切换，重建飞书 HTTP 会话")
            try:
                await self._session.close()
            except Exception as e:  # pragma: no cover - 兜底清理
                self._logger.warning(f"关闭旧会话失败（将继续重建）: {e}")
            finally:
                self._session = None
                self._session_loop = None

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
            )
            self._session_loop = current_loop
        return self._session

    async def close(self) -> None:
        """关闭 HTTP 会话"""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._session_loop = None

    # ==================== Token 管理 ====================

    async def ensure_token(self) -> str:
        """
        确保 Token 有效，必要时自动刷新

        Returns:
            有效的 tenant_access_token
        """
        now = time.time()
        if self._token and now < self._token_expires_at - TOKEN_REFRESH_MARGIN:
            return self._token

        self._logger.info("正在获取/刷新飞书 tenant_access_token ...")
        session = await self._get_session()

        async with session.post(
            TOKEN_URL,
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            headers={"Content-Type": "application/json; charset=utf-8"},
        ) as resp:
            body = await resp.json()

        code = body.get("code", -1)
        if code != 0:
            raise FeishuAPIError(
                code=code,
                msg=body.get("msg", "获取 Token 失败"),
                url=TOKEN_URL,
            )

        self._token = body["tenant_access_token"]
        expires_in = body.get("expire", 7200)
        self._token_expires_at = time.time() + expires_in
        self._logger.info(f"Token 获取成功，有效期 {expires_in} 秒")
        return self._token

    async def check_connection(self) -> bool:
        """
        检查连接是否正常（通过获取 Token 验证）

        Returns:
            bool: 连接成功返回 True，失败抛出异常或返回 False
        """
        try:
            await self.ensure_token()
            return True
        except Exception as e:
            self._logger.error(f"连接检查失败: {e}")
            raise

    async def _auth_headers(self) -> dict[str, str]:
        """构建认证请求头"""
        token = await self.ensure_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    # ==================== 底层请求 ====================

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """
        发送带重试的 HTTP 请求

        自动处理:
            - Token 过期 → 刷新后重试
            - 429 限流 → 读取 retry-after / x-ogw-ratelimit-reset 等待后重试
            - 5xx 服务端错误 → 指数退避重试
            - 网络超时 → 指数退避重试

        Args:
            method: HTTP 方法
            url: 请求 URL
            json_data: JSON 请求体
            params: URL 查询参数

        Returns:
            飞书 API 响应的 data 字段（已去掉外层 code/msg）

        Raises:
            FeishuPermissionError: 权限不足
            FeishuRateLimitError: 重试耗尽后仍被限流
            FeishuAPIError: 其他 API 错误
        """
        last_error: Exception | None = None

        # 信号量控制并发
        async with self._semaphore:
            for attempt in range(self.max_retries + 1):
                # 速率控制 (线程安全)
                async with self._rate_limit_lock:
                    if self._min_interval > 0:
                        now = time.time()
                        elapsed = now - self._last_request_time
                        if elapsed < self._min_interval:
                            await asyncio.sleep(self._min_interval - elapsed)
                        self._last_request_time = time.time()

                try:
                    headers = await self._auth_headers()
                    session = await self._get_session()

                    async with session.request(
                        method,
                        url,
                        json=json_data,
                        params=params,
                        headers=headers,
                    ) as resp:
                        # 处理 429 限流
                        if resp.status == 429:
                            retry_after = float(
                                resp.headers.get(
                                    "x-ogw-ratelimit-reset",
                                    resp.headers.get("Retry-After", "1"),
                                )
                            )
                            if attempt < self.max_retries:
                                self._logger.warning(
                                    f"飞书 429 限流，等待 {retry_after}s 后重试 "
                                    f"(第 {attempt + 1}/{self.max_retries} 次)"
                                )
                                await asyncio.sleep(retry_after)
                                continue
                            raise FeishuRateLimitError(
                                msg="重试耗尽后仍被限流",
                                retry_after=retry_after,
                                url=url,
                            )

                        # 处理 5xx 服务端错误
                        if resp.status >= 500:
                            if attempt < self.max_retries:
                                wait = DEFAULT_RETRY_BASE_DELAY * (2**attempt)
                                self._logger.warning(
                                    f"飞书服务端 {resp.status}，等待 {wait}s 后重试"
                                )
                                await asyncio.sleep(wait)
                                continue
                            body_text = await resp.text()
                            raise FeishuAPIError(
                                code=resp.status,
                                msg=f"服务端错误: {body_text[:200]}",
                                url=url,
                            )

                        body = await resp.json()

                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                ) as e:
                    last_error = e
                    if attempt < self.max_retries:
                        wait = DEFAULT_RETRY_BASE_DELAY * (2**attempt)
                        self._logger.warning(f"网络错误: {e}，等待 {wait}s 后重试")
                        await asyncio.sleep(wait)
                        continue
                    raise FeishuAPIError(
                        code=-1,
                        msg=f"网络错误: {e}",
                        url=url,
                    ) from last_error

                # 解析飞书业务层 code
                biz_code = body.get("code", 0)
                if biz_code == 0:
                    return body.get("data", {})

                # Token 失效 → 刷新后重试
                if biz_code in (99991661, 99991668):
                    self._token = None
                    if attempt < self.max_retries:
                        self._logger.warning("Token 失效，正在刷新 ...")
                        continue
                    raise FeishuAPIError(code=biz_code, msg=body.get("msg", ""), url=url)

                # 业务层频控（非 HTTP 429，而是 code=99991400）→ 等待后重试
                if biz_code == 99991400:
                    if attempt < self.max_retries:
                        wait = DEFAULT_RETRY_BASE_DELAY * (2**attempt)
                        self._logger.warning(
                            f"飞书业务层频控 (code={biz_code})，等待 {wait}s 后重试"
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise FeishuRateLimitError(
                        msg=body.get("msg", "业务层频控"),
                        retry_after=DEFAULT_RETRY_BASE_DELAY,
                        url=url,
                    )

                # 权限不足 / 配额耗尽 → 终止
                if biz_code in (99991401, 99991403):
                    raise FeishuPermissionError(
                        code=biz_code,
                        msg=body.get("msg", "权限不足"),
                        url=url,
                    )

                # Sheet 单元格超限 (Sheet 写入常见)
                if biz_code == 131002: # 单元格数量超过限制
                    raise FeishuAPIError(
                        code=biz_code,
                        msg=f"单元格超限: {body.get('msg')}",
                        url=url
                    )

                # 其他错误
                raise FeishuAPIError(
                    code=biz_code,
                    msg=body.get("msg", "未知错误"),
                    url=url,
                )

        # 理论上不会执行到此处
        raise FeishuAPIError(
            code=-1,
            msg=f"重试 {self.max_retries} 次后仍然失败",
            url=url,
        )

    # ==================== Bitable (多维表格) API ====================

    async def bitable_list_records(
        self,
        app_token: str,
        table_id: str,
        *,
        page_size: int = BITABLE_LIST_PAGE_SIZE,
        filter_expr: str | None = None,
        field_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        拉取多维表格全部记录（自动翻页）

        Args:
            app_token: 多维表格 App Token
            table_id: 数据表 ID
            page_size: 每页记录数（最大 500）
            filter_expr: 可选筛选表达式
            field_names: 可选字段名列表

        Returns:
            所有记录列表 [{"record_id": "recXXX", "fields": {...}}, ...]
        """
        url = (
            f"{FEISHU_BASE_URL}/bitable/v1/apps/{app_token}"
            f"/tables/{table_id}/records/search"
        )
        all_records: list[dict[str, Any]] = []
        page_token: str | None = None

        while True:
            body: dict[str, Any] = {"page_size": page_size}
            if page_token:
                body["page_token"] = page_token
            if filter_expr:
                body["filter"] = filter_expr
            if field_names:
                body["field_names"] = field_names

            data = await self._request("POST", url, json_data=body)
            items = data.get("items", [])
            all_records.extend(items)

            if not data.get("has_more", False):
                break
            page_token = data.get("page_token")
            if not page_token:
                break

        self._logger.info(f"Bitable 共拉取 {len(all_records)} 条记录")
        return all_records

    async def bitable_batch_update(
        self,
        app_token: str,
        table_id: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        批量更新多维表格记录（单次上限 1000 条，过大时自动二分，支持并发）

        Args:
            app_token: 多维表格 App Token
            table_id: 数据表 ID
            records: 记录列表 [{"record_id": "recXXX", "fields": {...}}, ...]

        Returns:
            更新后的记录列表
        """
        url = (
            f"{FEISHU_BASE_URL}/bitable/v1/apps/{app_token}"
            f"/tables/{table_id}/records/batch_update"
        )

        tasks = []
        for i in range(0, len(records), BITABLE_BATCH_UPDATE_LIMIT):
            chunk = records[i : i + BITABLE_BATCH_UPDATE_LIMIT]
            tasks.append(self._bitable_batch_with_split(url, chunk))

        # 并发执行所有块
        results_list = await asyncio.gather(*tasks)

        # 合并结果
        all_results = []
        for res in results_list:
            all_results.extend(res)

        return all_results

    async def bitable_batch_create(
        self,
        app_token: str,
        table_id: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        批量创建多维表格记录（单次上限 1000 条，过大时自动二分，支持并发）

        Args:
            app_token: 多维表格 App Token
            table_id: 数据表 ID
            records: 记录列表 [{"fields": {...}}, ...]

        Returns:
            创建后的记录列表
        """
        url = (
            f"{FEISHU_BASE_URL}/bitable/v1/apps/{app_token}"
            f"/tables/{table_id}/records/batch_create"
        )

        tasks = []
        for i in range(0, len(records), BITABLE_BATCH_CREATE_LIMIT):
            chunk = records[i : i + BITABLE_BATCH_CREATE_LIMIT]
            tasks.append(self._bitable_batch_with_split(url, chunk))

        # 并发执行所有块
        results_list = await asyncio.gather(*tasks)

        # 合并结果
        all_results = []
        for res in results_list:
            all_results.extend(res)

        return all_results

    async def _bitable_batch_with_split(
        self,
        url: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        执行 Bitable 批量操作，遇到请求过大时自动二分重试

        参考 XTF engine.py 的栈式迭代，避免递归栈溢出。
        """
        stack = [records]
        all_results: list[dict[str, Any]] = []

        while stack:
            chunk = stack.pop()
            try:
                data = await self._request("POST", url, json_data={"records": chunk})
                all_results.extend(data.get("records", []))
            except FeishuAPIError as e:
                if _is_too_large_error(e) and len(chunk) > 1:
                    mid = len(chunk) // 2
                    self._logger.warning(
                        f"Bitable 请求过大 (code={e.code})，"
                        f"自动二分: {len(chunk)} → {mid} + {len(chunk) - mid}"
                    )
                    # LIFO: 先压后半，再压前半 → 先执行前半
                    stack.append(chunk[mid:])
                    stack.append(chunk[:mid])
                else:
                    raise

        return all_results

    async def bitable_list_fields(
        self, app_token: str, table_id: str
    ) -> list[dict[str, Any]]:
        """
        列出多维表格字段

        Returns:
            字段列表 [{"field_id": "fldXXX", "field_name": "xxx", "type": 1}, ...]
        """
        url = (
            f"{FEISHU_BASE_URL}/bitable/v1/apps/{app_token}"
            f"/tables/{table_id}/fields"
        )
        data = await self._request("GET", url, params={"page_size": "100"})
        return data.get("items", [])

    async def bitable_get_meta(self, app_token: str) -> dict[str, Any]:
        """获取多维表格元数据"""
        url = f"{FEISHU_BASE_URL}/bitable/v1/apps/{app_token}"
        return await self._request("GET", url)

    # ==================== Sheet (电子表格) API ====================

    async def sheet_get_meta(self, spreadsheet_token: str) -> dict[str, Any]:
        """获取电子表格元数据（包含工作表列表）"""
        url = (
            f"{FEISHU_BASE_URL}/sheets/v3/spreadsheets/{spreadsheet_token}/sheets/query"
        )
        return await self._request("GET", url)

    async def sheet_read_range(
        self,
        spreadsheet_token: str,
        range_str: str,
    ) -> list[list[Any]]:
        """
        读取电子表格指定范围的数据（过大时自动行二分）

        Args:
            spreadsheet_token: 电子表格 Token
            range_str: 范围字符串，如 "Sheet1!A1:Z1000"

        Returns:
            二维数组 [[cell, cell, ...], ...]
        """
        try:
            return await self._sheet_read_single(spreadsheet_token, range_str)
        except FeishuAPIError as e:
            if not _is_too_large_error(e):
                raise
            self._logger.warning(
                f"Sheet 读取范围过大 (code={e.code})，启用自动分块: {range_str}"
            )
            return await self._sheet_read_chunked(spreadsheet_token, range_str)

    async def _sheet_read_single(
        self,
        spreadsheet_token: str,
        range_str: str,
    ) -> list[list[Any]]:
        """读取单个范围（无回退）"""
        url = (
            f"{FEISHU_BASE_URL}/sheets/v2/spreadsheets"
            f"/{spreadsheet_token}/values/{range_str}"
        )
        data = await self._request(
            "GET",
            url,
            params={
                "valueRenderOption": "ToString",
                "dateTimeRenderOption": "FormattedString",
            },
        )
        return data.get("valueRange", {}).get("values", [])

    async def _sheet_read_chunked(
        self,
        spreadsheet_token: str,
        range_str: str,
    ) -> list[list[Any]]:
        """
        分块读取电子表格，遇到 90221/90227 自动行数减半

        参考 XTF api/sheet.py 的行二分策略。
        """
        parsed = _parse_range(range_str)
        if parsed is None:
            raise FeishuAPIError(code=-1, msg=f"无法解析范围用于分块: {range_str}")

        sheet_id, start_col, start_row, end_col, end_row = parsed
        chunk_size = max(1, (end_row - start_row + 1) // 2)
        all_rows: list[list[Any]] = []
        row_cursor = start_row

        while row_cursor <= end_row:
            current_end = min(row_cursor + chunk_size - 1, end_row)
            sub_range = f"{sheet_id}!{start_col}{row_cursor}:{end_col}{current_end}"

            try:
                rows = await self._sheet_read_single(spreadsheet_token, sub_range)
                all_rows.extend(rows)
                row_cursor = current_end + 1
            except FeishuAPIError as e:
                if _is_too_large_error(e) and chunk_size > 1:
                    old_size = chunk_size
                    chunk_size = max(1, chunk_size // 2)
                    self._logger.warning(
                        f"Sheet 分块仍过大，行数减半: {old_size} → {chunk_size}"
                    )
                else:
                    raise

        self._logger.info(
            f"Sheet 分块读取完成，共 {len(all_rows)} 行 (块大小: {chunk_size})"
        )
        return all_rows

    async def sheet_write_range(
        self,
        spreadsheet_token: str,
        range_str: str,
        values: list[list[Any]],
    ) -> dict[str, Any]:
        """
        向电子表格指定范围写入数据（过大时自动二分）

        Args:
            spreadsheet_token: 电子表格 Token
            range_str: 范围字符串
            values: 二维数组

        Returns:
            写入结果
        """
        if not values:
            return {}

        # 尝试直接写入
        try:
            return await self._sheet_write_single(spreadsheet_token, range_str, values)
        except FeishuAPIError as e:
            if not _is_too_large_error(e) or len(values) <= 1:
                raise
            self._logger.warning(
                f"Sheet 写入过大 (code={e.code})，启用自动二分: " f"{len(values)} 行"
            )

        # 栈式二分写入
        parsed = _parse_range(range_str)
        if parsed is None:
            raise FeishuAPIError(code=-1, msg=f"无法解析范围用于分块写入: {range_str}")

        sheet_id, start_col, start_row, end_col, _end_row = parsed
        stack = [(start_row, values)]
        last_result: dict[str, Any] = {}

        while stack:
            row_start, chunk_values = stack.pop()
            row_end = row_start + len(chunk_values) - 1
            sub_range = f"{sheet_id}!{start_col}{row_start}:{end_col}{row_end}"
            try:
                last_result = await self._sheet_write_single(
                    spreadsheet_token, sub_range, chunk_values
                )
            except FeishuAPIError as e:
                if _is_too_large_error(e) and len(chunk_values) > 1:
                    mid = len(chunk_values) // 2
                    mid_row = row_start + mid
                    self._logger.warning(
                        f"Sheet 写入二分: {len(chunk_values)} → "
                        f"{mid} + {len(chunk_values) - mid}"
                    )
                    # LIFO: 先压后半，再压前半
                    stack.append((mid_row, chunk_values[mid:]))
                    stack.append((row_start, chunk_values[:mid]))
                else:
                    raise

        return last_result

    async def _sheet_write_single(
        self,
        spreadsheet_token: str,
        range_str: str,
        values: list[list[Any]],
    ) -> dict[str, Any]:
        """写入单个范围（无回退）"""
        url = f"{FEISHU_BASE_URL}/sheets/v2/spreadsheets" f"/{spreadsheet_token}/values"
        body = {
            "valueRange": {
                "range": range_str,
                "values": values,
            }
        }
        return await self._request("PUT", url, json_data=body)

    async def sheet_get_info(self, spreadsheet_token: str) -> dict[str, Any]:
        """获取电子表格基本信息"""
        url = f"{FEISHU_BASE_URL}/sheets/v3/spreadsheets/{spreadsheet_token}"
        return await self._request("GET", url)


# ==================== 模块级辅助函数 ====================


def _is_too_large_error(e: Exception) -> bool:
    """
    检测请求/响应过大错误（90221 TooLargeResponse / 90227 TooLargeRequest）

    参考 XTF api/sheet.py 的 _is_too_large_response，
    同时检查错误码和错误消息字符串以增强兼容性。
    """
    if isinstance(e, FeishuAPIError) and e.code in FEISHU_TOO_LARGE_CODES:
        return True
    msg = str(e)
    return any(kw in msg for kw in _TOO_LARGE_KEYWORDS)


def _parse_range(
    range_str: str,
) -> tuple[str, str, int, str, int] | None:
    """
    解析电子表格范围字符串

    "Sheet1!A1:Z1000" → ("Sheet1", "A", 1, "Z", 1000)
    无法解析时返回 None
    """
    m = _RANGE_RE.match(range_str)
    if not m:
        return None
    return (m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5)))
