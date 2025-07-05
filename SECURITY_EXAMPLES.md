# 安全配置管理示例

## 1. 环境变量配置 (.env 文件)

```bash
# API密钥配置
API_KEY_MODEL_1=sk-1234567890abcdef
API_KEY_MODEL_2=claude-key-abcdef123456
API_KEY_MODEL_3=llama-key-xyz789

# 数据库配置
DB_HOST=localhost
DB_USER=ai_user
DB_PASSWORD=secure_password_123
DB_NAME=ai_dataflux

# 加密密钥 (用于敏感配置加密)
ENCRYPTION_KEY=fernet-key-base64-encoded-here
```

## 2. 安全配置管理器实现

```python
# security/config_manager.py
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from cryptography.fernet import Fernet

class SecureConfigManager:
    """安全的配置管理器，支持环境变量和加密存储"""
    
    def __init__(self, config_path: str):
        load_dotenv()  # 加载.env文件
        self.config_path = config_path
        self.encryption_key = self._get_encryption_key()
        self.fernet = Fernet(self.encryption_key) if self.encryption_key else None
        
    def _get_encryption_key(self) -> Optional[bytes]:
        """获取加密密钥"""
        key_str = os.getenv('ENCRYPTION_KEY')
        if key_str:
            return key_str.encode()
        return None
        
    def get_api_key(self, model_id: str) -> Optional[str]:
        """安全获取API密钥"""
        key = os.getenv(f'API_KEY_MODEL_{model_id}')
        if not key:
            logging.warning(f"API密钥未找到: MODEL_{model_id}")
        return key
        
    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
            'port': int(os.getenv('DB_PORT', '3306'))
        }
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        if self.fernet:
            return self.fernet.encrypt(data.encode()).decode()
        return data
        
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        if self.fernet:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        return encrypted_data
```

## 3. SQL安全查询构建器

```python
# security/sql_builder.py
from typing import List, Dict, Any, Tuple

class SafeSQLBuilder:
    """安全的SQL查询构建器，防止SQL注入"""
    
    @staticmethod
    def build_select_query(
        table_name: str,
        columns: List[str],
        where_conditions: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = None
    ) -> Tuple[str, List[Any]]:
        """构建安全的SELECT查询"""
        
        # 验证表名和列名 (只允许字母数字和下划线)
        if not SafeSQLBuilder._is_safe_identifier(table_name):
            raise ValueError(f"不安全的表名: {table_name}")
            
        safe_columns = []
        for col in columns:
            if not SafeSQLBuilder._is_safe_identifier(col):
                raise ValueError(f"不安全的列名: {col}")
            safe_columns.append(f"`{col}`")
            
        # 构建基础查询
        query = f"SELECT {', '.join(safe_columns)} FROM `{table_name}`"
        params = []
        
        # 添加WHERE条件
        if where_conditions:
            where_parts = []
            for column, value in where_conditions.items():
                if not SafeSQLBuilder._is_safe_identifier(column):
                    raise ValueError(f"不安全的列名: {column}")
                where_parts.append(f"`{column}` = %s")
                params.append(value)
            query += f" WHERE {' AND '.join(where_parts)}"
            
        # 添加ORDER BY
        if order_by:
            if not SafeSQLBuilder._is_safe_identifier(order_by):
                raise ValueError(f"不安全的排序列名: {order_by}")
            query += f" ORDER BY `{order_by}`"
            
        # 添加LIMIT
        if limit:
            query += " LIMIT %s"
            params.append(limit)
            
        return query, params
        
    @staticmethod
    def build_update_query(
        table_name: str,
        updates: Dict[str, Any],
        where_conditions: Dict[str, Any]
    ) -> Tuple[str, List[Any]]:
        """构建安全的UPDATE查询"""
        
        if not SafeSQLBuilder._is_safe_identifier(table_name):
            raise ValueError(f"不安全的表名: {table_name}")
            
        # 构建SET部分
        set_parts = []
        params = []
        for column, value in updates.items():
            if not SafeSQLBuilder._is_safe_identifier(column):
                raise ValueError(f"不安全的列名: {column}")
            set_parts.append(f"`{column}` = %s")
            params.append(value)
            
        query = f"UPDATE `{table_name}` SET {', '.join(set_parts)}"
        
        # 添加WHERE条件
        if where_conditions:
            where_parts = []
            for column, value in where_conditions.items():
                if not SafeSQLBuilder._is_safe_identifier(column):
                    raise ValueError(f"不安全的列名: {column}")
                where_parts.append(f"`{column}` = %s")
                params.append(value)
            query += f" WHERE {' AND '.join(where_parts)}"
        else:
            raise ValueError("UPDATE查询必须包含WHERE条件")
            
        return query, params
        
    @staticmethod
    def _is_safe_identifier(identifier: str) -> bool:
        """检查标识符是否安全 (只包含字母、数字、下划线)"""
        import re
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier))
```

## 4. 敏感信息过滤器

```python
# security/log_filter.py
import logging
import re
from typing import Set

class SensitiveDataFilter(logging.Filter):
    """日志敏感信息过滤器"""
    
    def __init__(self):
        super().__init__()
        # 定义敏感信息的正则表达式模式
        self.sensitive_patterns = [
            r'api_key["\']?\s*[:=]\s*["\']?([^"\'\s]+)',  # API密钥
            r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)',  # 密码
            r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)',    # Token
            r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)',   # 密钥
            r'Authorization:\s*Bearer\s+([^\s]+)',        # Bearer token
        ]
        
    def filter(self, record):
        """过滤日志记录中的敏感信息"""
        if hasattr(record, 'msg'):
            original_msg = str(record.msg)
            filtered_msg = self._mask_sensitive_data(original_msg)
            record.msg = filtered_msg
            
        # 同样处理args中的敏感信息
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    filtered_args.append(self._mask_sensitive_data(arg))
                else:
                    filtered_args.append(arg)
            record.args = tuple(filtered_args)
            
        return True
        
    def _mask_sensitive_data(self, text: str) -> str:
        """用星号替换敏感数据"""
        for pattern in self.sensitive_patterns:
            text = re.sub(
                pattern, 
                lambda m: m.group(0).replace(m.group(1), '*' * len(m.group(1))),
                text,
                flags=re.IGNORECASE
            )
        return text
```

## 5. 使用示例

```python
# main.py - 使用安全配置的示例
from security.config_manager import SecureConfigManager
from security.sql_builder import SafeSQLBuilder
from security.log_filter import SensitiveDataFilter
import logging

# 设置安全日志
logger = logging.getLogger()
logger.addFilter(SensitiveDataFilter())

# 初始化安全配置管理器
config_manager = SecureConfigManager("config.yaml")

# 安全获取API密钥
api_key = config_manager.get_api_key("1")
if not api_key:
    logger.error("API密钥未配置")
    exit(1)

# 安全构建SQL查询
try:
    query, params = SafeSQLBuilder.build_select_query(
        table_name="tasks",
        columns=["id", "question", "context"],
        where_conditions={"status": "pending"},
        order_by="id",
        limit=100
    )
    print(f"生成的安全查询: {query}")
    print(f"参数: {params}")
except ValueError as e:
    logger.error(f"SQL查询构建失败: {e}")

# 安全的数据库配置
db_config = config_manager.get_database_config()
logger.info("数据库连接配置已加载")  # 密码会被自动过滤
```

## 6. requirements.txt 更新

```txt
# 现有依赖
pyyaml
aiohttp
pandas
openpyxl
psutil
mysql-connector-python
fastapi
uvicorn
pydantic

# 新增安全依赖
python-dotenv>=0.19.0
cryptography>=3.4.8
```

## 7. 部署时的安全配置

```bash
#!/bin/bash
# deploy.sh - 部署脚本示例

# 生成加密密钥
export ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# 设置环境变量
export API_KEY_MODEL_1="your-actual-api-key-1"
export API_KEY_MODEL_2="your-actual-api-key-2"
export DB_PASSWORD="your-secure-database-password"

# 设置文件权限 (确保.env文件只有所有者可读)
chmod 600 .env

# 启动应用
python AI-DataFlux.py --config config.yaml
```

这个安全实现方案提供了：

1. **环境变量管理**: API密钥和敏感配置从代码中分离
2. **加密存储**: 支持敏感数据的加密存储
3. **SQL安全**: 防止SQL注入的查询构建器
4. **日志过滤**: 自动过滤日志中的敏感信息
5. **部署安全**: 安全的部署和配置管理

实施这些改进将显著提升AI-DataFlux的安全性。