# Elasticsearch 替换指南

本文档说明如何将 SimpleMem 项目中的 LanceDB 替换为 Elasticsearch。

## 概述

`elasticsearch_store.py` 提供了与 `vector_store.py` 完全相同的接口，可以直接替换使用。主要优势：

1. **更强的全文搜索**：使用 Elasticsearch 原生的 BM25 算法
2. **混合搜索**：支持向量搜索 + 全文搜索的混合查询
3. **更好的扩展性**：支持分布式部署和集群
4. **企业级特性**：高可用、备份、监控等

## 安装依赖

首先安装 Elasticsearch 相关依赖：

```bash
pip install elasticsearch>=8.0.0 elasticsearch-dsl>=8.0.0
```

## 启动 Elasticsearch

### 使用 Docker（推荐）

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.12.0
```

### 验证 Elasticsearch 运行

```bash
curl http://localhost:9200
```

应该返回 Elasticsearch 的版本信息。

## 配置修改

### 1. 更新 config.py

在 `config.py` 中添加 Elasticsearch 配置：

```python
# Database backend selection
DATABASE_BACKEND = "elasticsearch"  # 改为 "elasticsearch"

# Elasticsearch Configuration
ELASTICSEARCH_HOSTS = "localhost:9200"
ELASTICSEARCH_USERNAME = None  # 如果需要认证
ELASTICSEARCH_PASSWORD = None  # 如果需要认证
ELASTICSEARCH_INDEX_NAME = "simplemem_memory_entries"
ELASTICSEARCH_VERIFY_CERTS = False  # 生产环境建议设为 True
```

### 2. 修改 main.py

在 `main.py` 中，根据配置选择使用哪个存储后端：

```python
# 在 SimpleMemSystem.__init__ 中
from database.vector_store import VectorStore
from database.elasticsearch_store import ElasticsearchStore

# 根据配置选择存储后端
db_backend = getattr(config, 'DATABASE_BACKEND', 'lancedb')

if db_backend == 'elasticsearch':
    self.vector_store = ElasticsearchStore(
        hosts=getattr(config, 'ELASTICSEARCH_HOSTS', 'localhost:9200'),
        username=getattr(config, 'ELASTICSEARCH_USERNAME', None),
        password=getattr(config, 'ELASTICSEARCH_PASSWORD', None),
        embedding_model=self.embedding_model,
        index_name=getattr(config, 'ELASTICSEARCH_INDEX_NAME', 'simplemem_memory_entries'),
        verify_certs=getattr(config, 'ELASTICSEARCH_VERIFY_CERTS', False)
    )
else:
    # 默认使用 LanceDB
    self.vector_store = VectorStore(
        db_path=db_path,
        embedding_model=self.embedding_model,
        table_name=table_name
    )
```

## 代码修改示例

### 方式 1：直接替换（简单）

在 `main.py` 中直接导入并使用：

```python
# 替换这行
# from database.vector_store import VectorStore

# 改为
from database.elasticsearch_store import ElasticsearchStore as VectorStore
```

### 方式 2：条件选择（推荐）

在 `main.py` 中添加条件判断：

```python
import config
from database.vector_store import VectorStore as LanceDBStore
from database.elasticsearch_store import ElasticsearchStore

# 根据配置选择
if getattr(config, 'DATABASE_BACKEND', 'lancedb') == 'elasticsearch':
    VectorStore = ElasticsearchStore
else:
    VectorStore = LanceDBStore

# 然后正常使用
self.vector_store = VectorStore(...)
```

## API 兼容性

`ElasticsearchStore` 实现了与 `VectorStore` 完全相同的接口：

- ✅ `add_entries(entries: List[MemoryEntry])` - 批量添加条目
- ✅ `semantic_search(query: str, top_k: int)` - 向量相似度搜索
- ✅ `keyword_search(keywords: List[str], top_k: int)` - 关键词搜索（使用 BM25）
- ✅ `structured_search(...)` - 结构化元数据过滤
- ✅ `get_all_entries()` - 获取所有条目
- ✅ `clear()` - 清空数据

**无需修改其他代码**，所有调用 `VectorStore` 的地方都可以直接使用 `ElasticsearchStore`。

## 性能对比

| 特性 | LanceDB | Elasticsearch |
|------|---------|---------------|
| 向量搜索性能 | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐⭐ 良好 |
| 全文搜索（BM25） | ⭐⭐ 简单实现 | ⭐⭐⭐⭐⭐ 原生支持 |
| 部署复杂度 | ⭐⭐⭐⭐⭐ 简单（嵌入式） | ⭐⭐⭐ 需要独立服务 |
| 扩展性 | ⭐⭐⭐ 单机 | ⭐⭐⭐⭐⭐ 分布式集群 |
| 资源占用 | ⭐⭐⭐⭐⭐ 低 | ⭐⭐⭐ 中等 |

## 注意事项

1. **版本要求**：Elasticsearch 8.0+（支持 KNN 搜索）
2. **内存需求**：Elasticsearch 需要更多内存，建议至少 2GB
3. **索引创建**：首次运行会自动创建索引和映射
4. **数据迁移**：如果需要从 LanceDB 迁移数据，需要编写迁移脚本

## 故障排查

### 连接失败

```python
# 检查 Elasticsearch 是否运行
curl http://localhost:9200

# 检查防火墙和端口
netstat -an | grep 9200
```

### 版本不兼容

确保 Elasticsearch 版本 >= 8.0：

```python
# 在代码中会自动检查版本
# 如果版本 < 8.0，会抛出异常
```

### 索引创建失败

检查 Elasticsearch 日志：

```bash
docker logs elasticsearch
```

## 参考

- [Elasticsearch KNN 搜索文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)
- [Elasticsearch Python 客户端](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.html)
- ragflow 项目中的 Elasticsearch 实现：`~/myWorkspace/ragflow-0.20.4/rag/utils/es_conn.py`
