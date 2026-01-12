# ElasticsearchStore 使用说明

## 快速开始

### 1. 安装依赖

```bash
pip install elasticsearch>=8.0.0 elasticsearch-dsl>=8.0.0
```

### 2. 启动 Elasticsearch

```bash
docker run -d --name elasticsearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.12.0
```

### 3. 在代码中使用

```python
from database.elasticsearch_store import ElasticsearchStore
from utils.embedding import EmbeddingModel

# 创建 ElasticsearchStore 实例
store = ElasticsearchStore(
    hosts="localhost:9200",
    embedding_model=EmbeddingModel(),
    index_name="my_memory_entries"
)

# 使用方式与 VectorStore 完全相同
store.add_entries(entries)
results = store.semantic_search("query", top_k=10)
```

## 主要改进

1. **更好的关键词搜索**：使用 Elasticsearch 原生 BM25，比 LanceDB 的简单匹配更准确
2. **混合搜索支持**：可以同时进行向量搜索和全文搜索
3. **企业级特性**：支持集群、高可用、备份等

## 与 VectorStore 的差异

- ✅ **接口完全兼容**：所有方法签名相同
- ✅ **无需修改调用代码**：可以直接替换
- ⚠️ **需要独立服务**：需要运行 Elasticsearch 服务
- ⚠️ **资源占用更高**：需要更多内存

## 配置示例

在 `config.py` 中：

```python
DATABASE_BACKEND = "elasticsearch"
ELASTICSEARCH_HOSTS = "localhost:9200"
ELASTICSEARCH_INDEX_NAME = "simplemem_memory_entries"
```
