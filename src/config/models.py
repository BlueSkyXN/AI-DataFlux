"""Canonical AI-DataFlux v4 configuration models.

The public YAML contract is intentionally represented here instead of being
spread across command entry points.  Every model rejects unknown fields so a
configuration can never be silently interpreted as a different schema.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base for the canonical configuration tree."""

    model_config = ConfigDict(extra="forbid", strict=True)


def _nonempty(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


class LogConfig(StrictModel):
    level: Literal["debug", "info", "warning", "error"] = "info"
    format: Literal["text", "json"] = "text"
    output: Literal["console", "file"] = "console"
    file_path: str = "./logs/ai_dataflux.log"


class AuthConfig(StrictModel):
    token: str = Field(default="", repr=False)


class WorkspaceRoots(StrictModel):
    project: str = "."


class WorkspaceConfig(StrictModel):
    roots: dict[str, str] = Field(default_factory=lambda: {"project": "."})
    state_dir: str = "./.dataflux/jobs"

    @field_validator("roots")
    @classmethod
    def validate_roots(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("roots must not be empty")
        for root_id, path in value.items():
            if not root_id or not root_id.replace("_", "a").replace("-", "a").isalnum():
                raise ValueError(f"invalid workspace root id: {root_id!r}")
            _nonempty(path, f"roots.{root_id}")
        return value


class SchedulerConfig(StrictModel):
    max_active_jobs: Union[Literal["auto"], Annotated[int, Field(ge=1)]] = "auto"
    cpu_high_watermark: float = Field(default=85, gt=0, le=100)
    memory_high_watermark: float = Field(default=80, gt=0, le=100)
    min_free_memory_mb: int = Field(default=512, ge=0)
    sample_interval_seconds: float = Field(default=2, gt=0)


class TokenEstimationConfig(StrictModel):
    mode: Literal["in", "out", "io"] = "io"
    sample_size: int = -1
    encoding: str = "o200k_base"

    @field_validator("sample_size")
    @classmethod
    def validate_sample_size(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("sample_size must be -1 or a positive integer")
        return value

    @field_validator("encoding")
    @classmethod
    def validate_encoding(cls, value: str) -> str:
        return _nonempty(value, "encoding")


class RuntimeConfig(StrictModel):
    log: LogConfig = Field(default_factory=LogConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    token_estimation: TokenEstimationConfig = Field(
        default_factory=TokenEstimationConfig
    )


class ExcelDatasourceConfig(StrictModel):
    type: Literal["excel"]
    input_path: str
    output_path: str
    engine: Literal["auto", "pandas", "polars"] = "auto"
    reader: Literal["auto", "openpyxl", "calamine"] = "auto"
    writer: Literal["auto", "openpyxl", "xlsxwriter"] = "auto"
    require_all_input_fields: bool = True


class CsvDatasourceConfig(StrictModel):
    type: Literal["csv"]
    input_path: str
    output_path: str
    engine: Literal["auto", "pandas", "polars"] = "auto"
    require_all_input_fields: bool = True


class SqliteDatasourceConfig(StrictModel):
    type: Literal["sqlite"]
    db_path: str
    table_name: str
    require_all_input_fields: bool = True


class DatabaseDatasourceConfig(StrictModel):
    host: str
    port: int = Field(ge=1, le=65535)
    user: str
    password: str = Field(repr=False)
    database: str
    table_name: str
    pool_size: int = Field(default=10, ge=1)
    require_all_input_fields: bool = True


class MysqlDatasourceConfig(DatabaseDatasourceConfig):
    type: Literal["mysql"]
    port: int = Field(default=3306, ge=1, le=65535)


class PostgreSQLDatasourceConfig(DatabaseDatasourceConfig):
    type: Literal["postgresql"]
    port: int = Field(default=5432, ge=1, le=65535)
    schema_name: str = "public"


class FeishuBitableDatasourceConfig(StrictModel):
    type: Literal["feishu_bitable"]
    app_id: str
    app_secret: str = Field(repr=False)
    app_token: str
    table_id: str
    max_retries: int = Field(default=3, ge=0)
    qps_limit: float = Field(default=0, ge=0)
    require_all_input_fields: bool = True


class FeishuSheetDatasourceConfig(StrictModel):
    type: Literal["feishu_sheet"]
    app_id: str
    app_secret: str = Field(repr=False)
    spreadsheet_token: str
    sheet_id: str
    max_retries: int = Field(default=3, ge=0)
    qps_limit: float = Field(default=0, ge=0)
    require_all_input_fields: bool = True


DatasourceConfig = Annotated[
    Union[
        ExcelDatasourceConfig,
        CsvDatasourceConfig,
        SqliteDatasourceConfig,
        MysqlDatasourceConfig,
        PostgreSQLDatasourceConfig,
        FeishuBitableDatasourceConfig,
        FeishuSheetDatasourceConfig,
    ],
    Field(discriminator="type"),
]


class ColumnsConfig(StrictModel):
    extract: list[str] = Field(min_length=1)
    write: dict[str, str] = Field(min_length=1)

    @field_validator("extract")
    @classmethod
    def validate_extract(cls, value: list[str]) -> list[str]:
        normalized = [_nonempty(item, "extract item") for item in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("extract must not contain duplicate columns")
        return normalized

    @field_validator("write")
    @classmethod
    def validate_write(cls, value: dict[str, str]) -> dict[str, str]:
        for alias, column in value.items():
            _nonempty(alias, "write alias")
            _nonempty(column, f"write.{alias}")
        return value


class ModelSelectionConfig(StrictModel):
    mode: Literal["auto", "strict", "fallback_group"] = "auto"
    route_id: str | None = None
    group: str | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "ModelSelectionConfig":
        if self.mode == "strict" and not (self.route_id and self.route_id.strip()):
            raise ValueError("route_id is required when mode=strict")
        if self.mode == "fallback_group" and not (self.group and self.group.strip()):
            raise ValueError("group is required when mode=fallback_group")
        if self.mode == "auto" and (
            self.route_id is not None or self.group is not None
        ):
            raise ValueError("auto mode does not accept route_id or group")
        if self.mode == "strict" and self.group is not None:
            raise ValueError("strict mode does not accept group")
        if self.mode == "fallback_group" and self.route_id is not None:
            raise ValueError("fallback_group mode does not accept route_id")
        return self


class PromptConfig(StrictModel):
    required_fields: list[str] = Field(default_factory=list)
    use_json_schema: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)
    temperature_override: bool = True
    system_prompt: str | None = None
    template: str = ""


class PromptOverride(StrictModel):
    required_fields: list[str] | None = None
    use_json_schema: bool | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    temperature_override: bool | None = None
    system_prompt: str | None = None
    template: str | None = None


class ValidationConfig(StrictModel):
    enabled: bool = False
    field_rules: dict[str, list[Any]] = Field(default_factory=dict)


class ValidationOverride(StrictModel):
    enabled: bool | None = None
    field_rules: dict[str, list[Any]] | None = None


class RoutingRule(StrictModel):
    match: Any
    profile: str

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        return _nonempty(value, "profile")


class RoutingConfig(StrictModel):
    enabled: bool = False
    field: str | None = None
    subtasks: list[RoutingRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_enabled(self) -> "RoutingConfig":
        if self.enabled:
            if not self.field or not self.field.strip():
                raise ValueError("field is required when routing is enabled")
            if not self.subtasks:
                raise ValueError("subtasks are required when routing is enabled")
        return self


class RoutingProfile(StrictModel):
    prompt: PromptOverride | None = None
    validation: ValidationOverride | None = None

    @model_validator(mode="after")
    def validate_nonempty(self) -> "RoutingProfile":
        if self.prompt is None and self.validation is None:
            raise ValueError("routing profile must contain prompt or validation")
        return self


class ConcurrencyConfig(StrictModel):
    batch_size: int = Field(default=100, ge=1)
    max_in_flight: int = Field(default=100, ge=1)
    save_interval: int = Field(default=300, ge=1)
    shard_size: int = Field(default=10000, ge=1)
    min_shard_size: int = Field(default=1000, ge=1)
    max_shard_size: int = Field(default=50000, ge=1)
    max_connections: int = Field(default=1000, ge=1)
    max_connections_per_host: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_shard_bounds(self) -> "ConcurrencyConfig":
        if self.min_shard_size > self.max_shard_size:
            raise ValueError("min_shard_size must not exceed max_shard_size")
        return self


class TaskAttemptsConfig(StrictModel):
    api_error: int = Field(default=4, ge=1)
    content_error: int = Field(default=2, ge=1)
    system_error: int = Field(default=3, ge=1)
    source_error: int = Field(default=3, ge=1)


class JobRetryConfig(StrictModel):
    task_max_attempts: TaskAttemptsConfig = Field(default_factory=TaskAttemptsConfig)
    model_max_attempts: int = Field(default=3, ge=1)
    api_pause_duration_seconds: float = Field(default=2, gt=0)
    api_error_trigger_window_seconds: float = Field(default=2, gt=0)


class WritebackConfig(StrictModel):
    commit_max_attempts: int = Field(default=3, ge=1)
    reconciliation_max_attempts: int = Field(default=3, ge=1)
    backoff_initial_seconds: float = Field(default=1, gt=0)
    backoff_max_seconds: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def validate_backoff(self) -> "WritebackConfig":
        if self.backoff_initial_seconds > self.backoff_max_seconds:
            raise ValueError(
                "backoff_initial_seconds must not exceed backoff_max_seconds"
            )
        return self


class JobConfig(StrictModel):
    gateway_url: str = "http://127.0.0.1:8787"
    datasource: DatasourceConfig
    columns: ColumnsConfig
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    retry: JobRetryConfig = Field(default_factory=JobRetryConfig)
    writeback: WritebackConfig = Field(default_factory=WritebackConfig)

    @field_validator("gateway_url")
    @classmethod
    def validate_gateway_url(cls, value: str) -> str:
        return _nonempty(value, "gateway_url")


class GatewayListenConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=1, le=65535)
    workers: Literal[1] = 1


class GatewayConnectionPoolConfig(StrictModel):
    max_connections: int = Field(default=1000, ge=1)
    max_connections_per_host: int = Field(default=1000, ge=1)


class GatewayEndpointsConfig(StrictModel):
    chat_completions: str | None = None
    responses: str | None = None

    @model_validator(mode="after")
    def validate_endpoint(self) -> "GatewayEndpointsConfig":
        if self.chat_completions is None and self.responses is None:
            raise ValueError("at least one gateway endpoint is required")
        for field_name in ("chat_completions", "responses"):
            value = getattr(self, field_name)
            if value is not None:
                _nonempty(value, field_name)
        return self


class GatewayChannelConfig(StrictModel):
    base_url: str
    endpoints: GatewayEndpointsConfig
    timeout_seconds: int = Field(default=600, ge=1)
    proxy: str = ""
    ssl_verify: bool = True
    ip_pool: list[str] = Field(default_factory=list)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _nonempty(value, "base_url")


RouteCapability = Literal[
    "chat_completions",
    "responses",
    "stream",
    "multimodal",
    "tools",
    "n",
    "json_schema",
    "logprobs",
    "previous_response_id",
]


class GatewayRouteConfig(StrictModel):
    id: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    upstream_model: str
    channel_id: str
    api_key: str = Field(default="", repr=False)
    capabilities: list[RouteCapability] = Field(min_length=1)
    weight: int = Field(default=1, ge=0)
    safe_rps: float = Field(default=1, gt=0)
    timeout_seconds: int = Field(default=300, ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2)

    @model_validator(mode="after")
    def validate_route(self) -> "GatewayRouteConfig":
        for field_name in ("id", "display_name", "upstream_model", "channel_id"):
            _nonempty(getattr(self, field_name), field_name)
        aliases = [alias.strip() for alias in self.aliases]
        if any(not alias for alias in aliases):
            raise ValueError("aliases must contain non-empty strings")
        if len(set(aliases)) != len(aliases):
            raise ValueError("aliases must not contain duplicates")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must not contain duplicates")
        capabilities = set(self.capabilities)
        if not capabilities.intersection({"chat_completions", "responses"}):
            raise ValueError("capabilities must include chat_completions or responses")
        if "previous_response_id" in capabilities and "responses" not in capabilities:
            raise ValueError("previous_response_id requires responses")
        return self


class GatewayRetryConfig(StrictModel):
    max_attempts_per_request: int = Field(default=3, ge=1)


class GatewayAffinityConfig(StrictModel):
    ttl_seconds: int = Field(default=3600, ge=1)
    max_entries: int = Field(default=10000, ge=1)


class GatewayConfig(StrictModel):
    listen: GatewayListenConfig = Field(default_factory=GatewayListenConfig)
    connection_pool: GatewayConnectionPoolConfig = Field(
        default_factory=GatewayConnectionPoolConfig
    )
    channels: dict[str, GatewayChannelConfig] = Field(default_factory=dict)
    routes: list[GatewayRouteConfig] = Field(default_factory=list)
    fallback_groups: dict[str, list[str]] = Field(default_factory=dict)
    retry: GatewayRetryConfig = Field(default_factory=GatewayRetryConfig)
    affinity: GatewayAffinityConfig = Field(default_factory=GatewayAffinityConfig)

    @model_validator(mode="after")
    def validate_references(self) -> "GatewayConfig":
        route_ids: set[str] = set()
        aliases: dict[str, str] = {}
        for route in self.routes:
            if route.id in route_ids:
                raise ValueError(f"duplicate route id: {route.id}")
            route_ids.add(route.id)
            if route.channel_id not in self.channels:
                raise ValueError(
                    f"route {route.id} references unknown channel: {route.channel_id}"
                )
            channel_endpoints = self.channels[route.channel_id].endpoints
            for endpoint in set(route.capabilities).intersection(
                {"chat_completions", "responses"}
            ):
                if getattr(channel_endpoints, endpoint) is None:
                    raise ValueError(
                        f"route {route.id} requires missing channel endpoint: {endpoint}"
                    )
            for alias in [
                route.id,
                route.display_name,
                route.upstream_model,
                *route.aliases,
            ]:
                owner = aliases.get(alias)
                if owner is not None and owner != route.id:
                    raise ValueError(f"route alias {alias!r} conflicts with {owner}")
                aliases[alias] = route.id

        for group, members in self.fallback_groups.items():
            _nonempty(group, "fallback group name")
            if not members:
                raise ValueError(f"fallback group {group!r} must not be empty")
            if len(set(members)) != len(members):
                raise ValueError(f"fallback group {group!r} contains duplicate routes")
            unknown = [route_id for route_id in members if route_id not in route_ids]
            if unknown:
                raise ValueError(
                    f"fallback group {group!r} references unknown routes: {unknown}"
                )
        return self


class ControlListenConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8790, ge=1, le=65535)


class ControlConfig(StrictModel):
    listen: ControlListenConfig = Field(default_factory=ControlListenConfig)


class RootConfig(StrictModel):
    schema_version: Literal[4]
    runtime: RuntimeConfig
    job: JobConfig | None = None
    gateway: GatewayConfig | None = None
    control: ControlConfig | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: Any) -> int:
        if type(value) is not int or value != 4:
            raise ValueError("schema_version must be the integer 4")
        return value

    @model_validator(mode="after")
    def validate_components(self) -> "RootConfig":
        if self.job is None and self.gateway is None and self.control is None:
            raise ValueError("at least one of job, gateway, or control is required")
        return self
