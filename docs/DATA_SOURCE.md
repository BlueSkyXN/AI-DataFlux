# AI-DataFlux 数据源（Data Source）读写回机制

本文档**单纯从代码逻辑**出发，总结当前程序支持的数据源类型，以及“读取任务 → 写回结果”的实现方式与差异点。

> 主要代码入口：
> - 数据源工厂：`src/data/factory.py`
> - 抽象接口：`src/data/base.py`
> - 具体实现：`src/data/{mysql,postgresql,sqlite,excel}.py`
> - 飞书实现：`src/data/feishu/{client,bitable,sheet}.py`

---

## 支持的数据源类型

由 `create_task_pool()` 根据 `datasource.type` 选择实现（`src/data/factory.py`）：

| `datasource.type` | 读取来源 | 写回目标 | `task_id`/定位键 | 依赖 |
|---|---|---|---|---|
| `mysql` | MySQL 表 | 同表 `UPDATE` | `id` 主键 | `mysql-connector-python` |
| `postgresql` | PostgreSQL 表 | 同表 `UPDATE`（批量） | `id` 主键 | `psycopg2` |
| `sqlite` | SQLite 表 | 同表 `UPDATE`（事务） | `id` 主键 | Python 标准库 `sqlite3` |
| `excel` | Excel 文件（`.xlsx/.xls`） | 输出文件（可原地） | DataFrame 行索引 | `pandas` + `openpyxl`（工厂检测） |
| `csv` | CSV 文件（`.csv`） | 输出文件（可原地） | DataFrame 行索引 | **复用 `ExcelTaskPool`**，同上（工厂检测） |
| `feishu_bitable` | 飞书多维表格 | 同表 `batch_update` | 连续整数 ↔ `record_id` 映射 | `aiohttp` |
| `feishu_sheet` | 飞书电子表格 | 同表单元格写入 | 数据行索引 ↔ 实际行号映射 | `aiohttp` |

> 注：`csv` 在工厂层面同样要求 `pandas` + `openpyxl` 可用（`EXCEL_ENABLED`），即使实际读写只用到了 CSV 能力（`src/data/factory.py`）。
> 注：飞书数据源详细文档请参考 [FEISHU.md](./FEISHU.md)。

---

## 通用配置语义（所有数据源一致）

### 1) 输入列与输出列

- `columns_to_extract`：从数据源读取的输入字段列表（如 `question`、`context`）
- `columns_to_write`：写回映射（别名 → 真实列名），例如：

```yaml
columns_to_write:
  answer: "ai_answer"
  category: "ai_category"
```

程序内部写回时，使用 “别名” 从结果里取值，再落到 “真实列名”。

### 2) 何为“未处理 / 已处理”

系统以“输出列是否为空”作为任务状态判断依据：

- **未处理（unprocessed）**：输入满足有效性规则 + **任一**输出列为空
- **已处理（processed）**：**所有**输出列非空

输入有效性由 `datasource.require_all_input_fields` 决定：

- `true`：要求 `columns_to_extract` 中**所有**输入列非空（AND）
- `false`：只要**至少一个**输入列非空（OR）

### 3) `_error` 结果不会写回

`update_task_results()` 会跳过包含 `_error` 键的结果（多数据源实现一致），因此这些行/记录仍可能被判定为“未处理”，后续可能再次被加载处理。

---

## 处理流程（高层）

数据源被封装为任务池（TaskPool），典型调用链为：

1. `create_task_pool(config, columns_to_extract, columns_to_write)`
2. `get_id_boundaries()` 获取 ID/索引范围，用于分片
3. `initialize_shard(shard_id, min_id, max_id)` 将该范围内“未处理”的任务加载进内存队列 `tasks`
4. 多次 `get_task_batch(batch_size)` 拉取批次并调用 AI 处理
5. `update_task_results(results)` 批量写回
6. `close()` 释放资源 / 最终保存

---

## 各数据源：读取与写回方案

### MySQL（`mysql`）

**读取**：
- 按 `id BETWEEN min_id AND max_id` 查询，并叠加“未处理条件”，加载到内存队列（`src/data/mysql.py`）

**写回**：
- 对每条结果执行一次 `UPDATE ... WHERE id = %s`
- 通过连接池封装事务：全部执行完后统一 `commit()`（`src/data/mysql.py`）
- 仅更新“结果里提供了别名值”的列（缺失别名不会触发该列更新）

### PostgreSQL（`postgresql`）

**读取**：
- 按 `id BETWEEN ...` + “未处理条件”查询（`src/data/postgresql.py`）

**写回**：
- 使用 `psycopg2.extras.execute_batch()` 批量执行 `UPDATE`（`src/data/postgresql.py`）
- 注意：更新语句会覆盖 `columns_to_write` 中的**所有输出列**
  - 如果某条结果缺少某个别名，参数会变成 `None`，可能把对应列写成 `NULL`

### SQLite（`sqlite`）

**读取**：
- 按 `id BETWEEN ? AND ?` + “未处理条件”查询（`src/data/sqlite.py`）

**写回**：
- 显式事务 `BEGIN TRANSACTION` / `COMMIT` / `ROLLBACK`
- 仅更新“结果里提供了别名值”的列（缺失别名不会触发该列更新）

### Excel / CSV（`excel` / `csv`）

两者共用同一个实现类 `ExcelTaskPool`（`src/data/excel.py`），区别仅在于文件后缀与读写路径。

**读取**：
- 启动时把文件读入 DataFrame（CSV 用 `read_csv`，Excel 用 `read_excel`）
- 分片时在 `[min_idx, max_idx]` 范围内向量化筛选“未处理行”，并以行索引作为 `task_id`（`src/data/excel.py`）

**写回**：
- 先把结果写入**内存 DataFrame**
- 达到 `save_interval`（`datasource.concurrency.save_interval`）后触发落盘；`close()` 时做最终保存（`src/data/excel.py`）
- `output_path` 不配置时默认原地写回 `input_path`（工厂逻辑：`src/data/factory.py`）
- Excel 保存遇到 `UnicodeEncodeError` 时，会尝试清空 AI 输出列的“问题单元格”，仍失败则回退保存为 CSV（`src/data/excel.py`）

**重要差异点**：
- 写回时对每个输出别名使用 `row_result.get(alias, "")`
  - 若结果缺少某个别名，会写入空字符串 `""`（可能覆盖已有值并导致该行继续被判定为“未处理”）

### 飞书多维表格（`feishu_bitable`）

实现类：`FeishuBitableTaskPool`（`src/data/feishu/bitable.py`），使用 `FeishuClient`（`src/data/feishu/client.py`）。

**读取**：
- 初始化时通过 Bitable search API 分页拉取**全部记录**到内存快照
- 建立连续整数 `task_id` ↔ 字符串 `record_id` 的映射表
- 分片时从内存快照中筛选“未处理行”

**写回**：
- 通过 `batch_update` API 批量更新（单次上限 1000 条），自动分块
- 遇到请求过大（90221/90227）自动二分重试（参考 XTF 栈式迭代策略）
- 按 `record_id` 覆盖写入，天然幂等
- 仅更新"结果里提供了别名值"的列

**关键差异**：
- `record_id` 是字符串（如 `recXXXXXX`），不是连续数字
- 需要网络请求，有延迟和失败可能
- Token 有效期 2 小时，由客户端自动刷新

### 飞书电子表格（`feishu_sheet`）

实现类：`FeishuSheetTaskPool`（`src/data/feishu/sheet.py`），使用 `FeishuClient`（`src/data/feishu/client.py`）。

**读取**：
- 初始化时通过 values API 一次性读取全部工作表数据到内存二维数组
- 第一行为表头，建立列名 → 列索引映射
- 数据行从第 2 行开始，行索引（0-based）作为 `task_id`

**写回**：
- 按列分组，连续行合并为一次范围写入，减少 API 调用
- 遇到请求过大（90227）自动二分重试
- 实际行号 = `task_id + 2`（跳过表头行，1-based）
- 列号使用 A/B/C...AA/AB 字母格式

**关键差异**：
- 读取过大范围时自动行二分（90221/90227 检测 + 减半重试）
- 行号可能因其他用户插入/删除而偏移（快照模式避免了此问题）
- 每次写入是独立 HTTP 请求，无事务回滚

---

## 常见坑/约束（基于现有代码行为）

- **状态完全由输出列是否为空决定**：如果你配置了多个输出列，只要有一个为空就会被当作“未处理”再次加载。
- **输出字段缺失的写回语义不一致**：
  - MySQL/SQLite：缺失别名 → 不更新该列
  - PostgreSQL：缺失别名 → 可能写成 `NULL`（覆盖）
  - Excel/CSV：缺失别名 → 写成 `""`（覆盖）
  - 飞书多维表格：缺失别名 → 不更新该列（与 MySQL/SQLite 一致）
  - 飞书电子表格：缺失别名 → 不更新该列
- **CSV 也要求 `pandas+openpyxl`**：当前是工厂层面的依赖检测策略，后续如要“纯 CSV + polars”需要调整 `EXCEL_ENABLED` 判定逻辑。
- **飞书数据源是云端操作**：写入有网络延迟、频控限制和部分失败的可能，详见 [FEISHU.md](./FEISHU.md)。

