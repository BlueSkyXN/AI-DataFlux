# AI-DataFlux 文档中心

欢迎阅读 AI-DataFlux 的技术文档。

## 4.0 开发治理

- [V4_EXECUTION_CONTRACT.md](./V4_EXECUTION_CONTRACT.md) - 4.0 clean break 的版本边界、运行语义、锁与恢复合同、Hardening 切片和 Promotion 门禁
- [BOOTSTRAP_ACCEPTANCE.md](./BOOTSTRAP_ACCEPTANCE.md) - Bootstrap 模块的接受、修订、替换或删除状态，以及对应 commit 和验证证据

> 冻结的 Bootstrap commit 仍保留 `3.2.0-dev` 历史身份；当前 Integration 源码为 `4.0.0-dev`。治理文档中的接受状态以对应 commit 和验证证据为准，不等同于 Release、部署或业务 UAT。

## 📚 核心文档

### 4.0 开发版契约

- [CONFIG.md](./CONFIG.md) - canonical v4 RootConfig、组件配置、datasource discriminator、hash/ETag 边界
- [JOBS.md](./JOBS.md) - durable Job Repository、Worker、state/event/lease/command schema 与对外交付边界
- [CONTROL_API.md](./CONTROL_API.md) - workspace、ETag 配置读写、Job REST/SSE、认证、错误和 CLI 自动化边界
- [GATEWAY_API.md](./GATEWAY_API.md) - Chat Completions / Responses 透传、capability 矩阵、错误和 failover 边界
- [RELEASE_GATES.md](./RELEASE_GATES.md) - CI、coverage、dependency audit、打包 smoke、Release/部署/UAT 证据层

> 这些文档描述 `4.0.0-dev` 当前本地源码合同。页面、类或测试存在不等于已发布、已部署或已通过业务 UAT。

### [ARCH.md](./ARCH.md) - 系统架构文档 ⭐️

**完整的系统架构说明**，包含 13 个章节，2048 行详细内容：

1. **系统概览** - 定位、设计哲学、架构视图、依赖关系、使用场景
2. **项目结构** - 目录布局、关键文件说明、组织原则
3. **架构分层** - 四层架构详解
4. **双组件架构** - 批处理引擎 + API 网关
5. **数据源层架构** - 7 种数据源实现（Excel/CSV/MySQL/PostgreSQL/SQLite/Feishu Bitable/Feishu Sheet）
6. **核心处理引擎** - 组件化设计（Content/State/Retry/Clients）
7. **API 网关架构** - 模型调度、限流、故障转移
8. **数据流与生命周期** - 完整的数据处理流程
9. **核心设计决策** - 连续任务流、元数据分离、向量化优化
10. **并发与性能** - 异步模型、内存管理、性能优化技术
11. **错误处理体系** - 分类重试、熔断机制
12. **扩展性设计** - 如何添加新数据源/引擎/组件
13. **技术栈** - 完整的依赖列表和架构模式

**特色**：
- ✨ 5 个 Mermaid 图增强可视化
- 📊 详细的代码示例和流程图
- 🔧 实用的扩展指南

---

### [GUI.md](./GUI.md) - Web GUI 控制面板 🖥️

**本地 Web GUI 控制面板使用指南**：

- **功能概述** - workspace 选择、配置编辑、durable Jobs、进程管理和日志查看
- **快速开始** - 启动控制面板的命令和参数
- **架构设计** - Control Server 与子进程的关系
- **API 接口** - REST API 和 WebSocket 接口说明
- **进程状态** - 三态状态机（stopped/running/exited）
- **开发说明** - 前端和后端的开发指南
- **跨平台支持** - Linux/macOS/Windows 兼容性

**使用方式**：
```bash
python cli.py gui              # 启动控制面板
python cli.py gui --port 8080  # 指定端口
```

---

### [CONFIG.md](./CONFIG.md) - 配置参数详解

**配置文件完全指南**，涵盖所有配置节：

- **Runtime** - 日志、认证、workspace、scheduler、token estimation
- **Job** - datasource、columns、model selection、prompt、validation、routing、retry/writeback
- **Gateway** - listen、connection pool、channels、routes、fallback groups、retry、affinity
- **Control** - 本地监听参数

每个参数都包含：
- 📍 代码位置（文件:行号）
- 🎯 实际影响（带代码片段）
- 🔗 相关文件列表
- 💡 调优建议

---

### [ROUTING.md](./ROUTING.md) - 规则路由配置指南

**规则路由功能完整指南**，适用于单文件多业务场景：

- **概述** - 规则路由的核心特性和适用场景
- **配置结构** - 主配置文件和子配置文件的组织方式
- **配置详解** - routing、subtask 配置项说明
- **处理流程** - 初始化和记录处理的完整流程图
- **使用示例** - 多业务单元工单分类的完整示例
- **配置合并规则** - 深度合并策略说明
- **错误处理** - 字段不存在、无匹配规则等情况的处理
- **性能考虑** - 预加载、缓存策略
- **向后兼容** - 不使用路由时的兼容性保证

**适用场景**：
- 单个数据文件包含多个业务单元（BU）的数据
- 不同业务单元需要不同的分类标签体系
- 希望一次处理完成，无需手动拆分文件

---

### [DATA_SOURCE.md](./DATA_SOURCE.md) - 数据源读写回机制

**从代码逻辑出发**，说明当前支持的数据源类型（Excel/CSV/MySQL/PostgreSQL/SQLite/Feishu Bitable/Feishu Sheet）以及：

- 任务“未处理/已处理”的判定规则
- 分片读取方案与 `task_id` 定位方式
- 各数据源写回策略与关键差异（如缺失字段写回语义）
- Excel/CSV 原子替换与写回失败语义

---

### [FEISHU.md](./FEISHU.md) - 飞书数据源指南

**飞书数据源接入文档**，包含：

- 飞书应用权限与 Token 获取方式
- `feishu_bitable` / `feishu_sheet` 配置示例
- 原生异步客户端的限流、重试、分块机制说明
- GUI 连接测试接口说明

---

### [../CLAUDE.md](../CLAUDE.md) - Claude Code 开发指南

**专为 Claude Code CLI 准备的开发文档**：

- 📦 项目概览和架构
- 🛠️ 开发命令速查
- 🏗️ 核心组件说明
- ✅ 测试指南
- 📝 常见模式和最佳实践
- ⚙️ 配置系统详解

---

## 📦 归档文档

[legacy/](./legacy/) 目录包含已整合的旧版文档，仅供历史参考：

- `LOGIC_FRAMEWORK.md` - 逻辑框架图（已整合到 ARCH.md 第1.4和1.5节）
- `architecture_diagram.md` - 架构图文档（已整合到 ARCH.md）
- `DESIGN.md` - 设计文档（已整合到 ARCH.md 第2章）

---

## 🚀 快速开始

1. **了解系统** → 阅读 [ARCH.md](./ARCH.md) 第1章「系统概览」
2. **配置项目** → 参考 [CONFIG.md](./CONFIG.md) 和 `config-example.yaml`
3. **切换 4.0 配置** → 直接按 [CONFIG.md](./CONFIG.md) 重写；4.0 不加载或迁移旧 YAML
4. **使用 GUI / Control API** → 运行 `python cli.py gui`，参考 [GUI.md](./GUI.md) 和 [CONTROL_API.md](./CONTROL_API.md)
5. **规则路由** → 查看 [ROUTING.md](./ROUTING.md) 和 `config-example.yaml`
6. **验证发布边界** → 阅读 [RELEASE_GATES.md](./RELEASE_GATES.md)
7. **运行测试** → `pytest tests/` 确保代码质量

---

## 📊 文档统计

| 文档 | 行数 | 章节数 | 用途 |
|------|------|--------|------|
| ARCH.md | 2048 | 13 | 系统架构完整说明 |
| CONFIG.md | 1200+ | 11 | 配置参数详解 |
| GUI.md | 200+ | 8 | Web GUI 控制面板 |
| ROUTING.md | 300+ | 8 | 规则路由指南 |
| CLAUDE.md | 600+ | 多个 | 开发指南 |

---

*最后更新: 2026-08-30*
