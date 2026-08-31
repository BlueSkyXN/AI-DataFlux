# 飞书数据源指南

AI-DataFlux 通过原生 `aiohttp` 客户端访问 Feishu Bitable 和 Sheet。4.0 的飞书凭据与资源 ID 全部位于 `job.datasource` discriminated union 中，不存在顶层 `feishu` section 或旧路径 fallback。

## Bitable 配置

```yaml
schema_version: 4
runtime:
  auth: {token: ""}
  workspace:
    roots: {project: .}
    state_dir: ./.dataflux/jobs
job:
  gateway_url: http://127.0.0.1:8787
  datasource:
    type: feishu_bitable
    app_id: cli_xxx
    app_secret: "..."
    app_token: basc_xxx
    table_id: tbl_xxx
    max_retries: 3
    qps_limit: 5
    require_all_input_fields: true
  columns:
    extract: [问题描述, 上下文信息]
    write:
      ai_analysis: AI分析结果
      category: 分类标签
  prompt:
    template: "处理记录：{record_json}"
```

## Sheet 配置

```yaml
job:
  datasource:
    type: feishu_sheet
    app_id: cli_xxx
    app_secret: "..."
    spreadsheet_token: shtcn_xxx
    sheet_id: "0"
    max_retries: 3
    qps_limit: 5
    require_all_input_fields: true
```

## 权限

Bitable 测试应用需要记录读取和更新权限；Sheet 测试应用需要 spreadsheet 读取和写入权限。权限名称以 Feishu 开放平台当前控制台为准。

## 运行语义

- Token 在内存中缓存并在过期前刷新。
- 限流和明确的可重试服务错误采用有界退避。
- Bitable 按 chunk 写入；每个 chunk 独立形成明确成功、明确失败或结果不明状态。
- Sheet 对单个文档保持串行写入；同一 record 跨多个 cell/segment 时，只有全部期望 cell 确认后才提交。
- timeout、连接中断或部分写入不能直接当作 retryable rejection，必须进入 reconciliation。
- Reconciliation 按 record/range 回读所有期望字段并逐值比较。

## GUI 连接测试

配置编辑器中的“测试连接”只校验 App ID / App Secret 能否取得 tenant token，不读取或写入业务表，也不代表 datasource writeback UAT 已完成。

## 验收边界

Mock contract test 只能证明请求、错误分类和 receipt 映射。真实 Feishu acceptance 还需要隔离测试应用和测试表的 read/write/reconcile 证据；未经授权不得对真实表执行写回。
