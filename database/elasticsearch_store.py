"""
Elasticsearch Store - Elasticsearch-based implementation of VectorStore

This module provides an Elasticsearch-based implementation to replace LanceDB,
maintaining the same interface as VectorStore for seamless integration.

Paper Reference: Section 3.2 - Structured Indexing
Implements the three structured indexing dimensions using Elasticsearch:
- Semantic Layer: Dense vectors using Elasticsearch KNN search
- Lexical Layer: BM25 keyword matching (native Elasticsearch)
- Symbolic Layer: Metadata filtering (native Elasticsearch)
"""
from typing import List, Optional, Dict, Any
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch_dsl import Search, Q, Index
from elastic_transport import ConnectionTimeout
import numpy as np
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
import config
import os
import json
import time
import logging

logger = logging.getLogger(__name__)


class ElasticsearchStore:
    """
    Elasticsearch-based Vector Store - Replacement for LanceDB
    
    Paper Reference: Section 3.2 - Structured Indexing
    Implements M(m_k) with three structured layers using Elasticsearch:
    1. Semantic Layer: Dense embedding vectors for conceptual similarity (KNN search)
    2. Lexical Layer: Sparse keyword vectors for precise term matching (BM25)
    3. Symbolic Layer: Structured metadata for deterministic filtering
    """
    
    def __init__(
        self, 
        hosts: str = None,
        username: str = None,
        password: str = None,
        embedding_model: EmbeddingModel = None, 
        index_name: str = None,
        verify_certs: bool = False,
        timeout: int = 600
    ):
        """
        Initialize Elasticsearch connection
        
        Args:
            hosts: Elasticsearch hosts (comma-separated), e.g., "localhost:9200"
            username: Elasticsearch username (optional)
            password: Elasticsearch password (optional)
            embedding_model: Embedding model instance
            index_name: Index name for storing memory entries
            verify_certs: Whether to verify SSL certificates
            timeout: Connection timeout in seconds
        """
        # Get configuration from config or use defaults
        self.hosts = hosts or getattr(config, 'ELASTICSEARCH_HOSTS', 'localhost:9200')
        self.username = username or getattr(config, 'ELASTICSEARCH_USERNAME', None)
        self.password = password or getattr(config, 'ELASTICSEARCH_PASSWORD', None)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index_name = index_name or getattr(config, 'ELASTICSEARCH_INDEX_NAME', 'simplemem_memory_entries')
        self.verify_certs = verify_certs
        self.timeout = timeout
        
        # Vector field name based on dimension
        self.vector_field = f"q_{self.embedding_model.dimension}_vec"
        
        # Connect to Elasticsearch
        self._connect()
        
        # Initialize index
        self._init_index()
    
    def _connect(self):
        """Connect to Elasticsearch cluster"""
        try:
            hosts_list = self.hosts.split(",") if isinstance(self.hosts, str) else self.hosts
            
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)
            
            self.es = Elasticsearch(
                hosts_list,
                basic_auth=auth,
                verify_certs=self.verify_certs,
                timeout=self.timeout
            )
            
            # Check connection
            if not self.es.ping():
                raise Exception(f"Failed to connect to Elasticsearch at {self.hosts}")
            
            # Check version (require 8.0+ for KNN support)
            info = self.es.info()
            version = info.get("version", {}).get("number", "0.0.0")
            major_version = int(version.split(".")[0])
            if major_version < 8:
                raise Exception(f"Elasticsearch version must be >= 8.0, current: {version}")
            
            logger.info(f"Connected to Elasticsearch {self.hosts}, version: {version}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def _init_index(self):
        """Initialize Elasticsearch index with proper mapping"""
        try:
            # Check if index exists
            if self.es.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists")
                return
            
            # Create index with mapping
            mapping = self._get_index_mapping()
            
            self.es.indices.create(
                index=self.index_name,
                body=mapping
            )
            logger.info(f"Created Elasticsearch index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise
    
    def _get_index_mapping(self) -> dict:
        """
        Get Elasticsearch index mapping for memory entries
        
        Returns:
            Mapping configuration for Elasticsearch index
        """
        dimension = self.embedding_model.dimension
        
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "knn": True,  # Enable KNN search
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "entry_id": {
                        "type": "keyword"
                    },
                    "lossless_restatement": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "keywords": {
                        "type": "keyword"
                    },
                    "timestamp": {
                        "type": "keyword"
                    },
                    "location": {
                        "type": "keyword"
                    },
                    "persons": {
                        "type": "keyword"
                    },
                    "entities": {
                        "type": "keyword"
                    },
                    "topic": {
                        "type": "keyword"
                    },
                    # Dense vector field for semantic search
                    self.vector_field: {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
    
    def add_entries(self, entries: List[MemoryEntry]):
        """
        Batch add memory entries to Elasticsearch
        
        Args:
            entries: List of MemoryEntry objects to add
        """
        if not entries:
            return
        
        try:
            # Generate vectors
            restatements = [entry.lossless_restatement for entry in entries]
            vectors = self.embedding_model.encode_documents(restatements)
            
            # Prepare bulk operations
            operations = []
            for entry, vector in zip(entries, vectors):
                doc = {
                    "entry_id": entry.entry_id,
                    "lossless_restatement": entry.lossless_restatement,
                    "keywords": entry.keywords,
                    "timestamp": entry.timestamp or "",
                    "location": entry.location or "",
                    "persons": entry.persons,
                    "entities": entry.entities,
                    "topic": entry.topic or "",
                    self.vector_field: vector.tolist()
                }
                
                operations.append({
                    "index": {
                        "_index": self.index_name,
                        "_id": entry.entry_id
                    }
                })
                operations.append(doc)
            
            # Bulk insert
            if operations:
                response = self.es.bulk(
                    operations=operations,
                    refresh=True,
                    timeout="60s"
                )
                
                # Check for errors
                if response.get("errors"):
                    errors = []
                    for item in response["items"]:
                        if "error" in item.get("index", {}):
                            errors.append(item["index"]["error"])
                    if errors:
                        logger.warning(f"Some entries failed to index: {errors[:5]}")
                
                logger.info(f"Added {len(entries)} memory entries to Elasticsearch")
            
        except Exception as e:
            logger.error(f"Error adding entries to Elasticsearch: {e}")
            raise
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """
        Semantic Layer Search - Dense vector similarity using KNN
        
        Paper Reference: Section 3.1
        Retrieves based on v_k = E_dense(S_k) where S_k is the lossless restatement
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of MemoryEntry objects sorted by similarity
        """
        try:
            # Check if index is empty
            count = self.es.count(index=self.index_name)["count"]
            if count == 0:
                return []
            
            # Generate query vector
            query_vector = self.embedding_model.encode_single(query, is_query=True)
            query_vector_list = query_vector.tolist()
            
            # Build KNN search query
            search_body = {
                "knn": {
                    "field": self.vector_field,
                    "query_vector": query_vector_list,
                    "k": top_k,
                    "num_candidates": top_k * 2
                },
                "_source": [
                    "entry_id",
                    "lossless_restatement",
                    "keywords",
                    "timestamp",
                    "location",
                    "persons",
                    "entities",
                    "topic"
                ]
            }
            
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=search_body,
                size=top_k
            )
            
            # Convert results to MemoryEntry objects
            entries = []
            for hit in response["hits"]["hits"]:
                try:
                    source = hit["_source"]
                    entry = MemoryEntry(
                        entry_id=source["entry_id"],
                        lossless_restatement=source["lossless_restatement"],
                        keywords=source.get("keywords", []),
                        timestamp=source.get("timestamp") or None,
                        location=source.get("location") or None,
                        persons=source.get("persons", []),
                        entities=source.get("entities", []),
                        topic=source.get("topic")
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse search result: {e}")
                    continue
            
            return entries
            
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []
    
    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]:
        """
        Lexical Layer Search - BM25 keyword matching
        
        Paper Reference: Section 3.1
        Retrieves based on h_k = Sparse(S_k) for precise term and entity matching
        Uses Elasticsearch's native BM25 scoring
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            
        Returns:
            List of MemoryEntry objects sorted by BM25 score
        """
        try:
            # Check if index is empty
            count = self.es.count(index=self.index_name)["count"]
            if count == 0:
                return []
            
            if not keywords:
                return []
            
            # Build query: search in keywords field and lossless_restatement
            keyword_query = " OR ".join(keywords)
            
            search = Search(using=self.es, index=self.index_name)
            search = search.query(
                Q("bool", should=[
                    Q("terms", keywords=keywords, boost=2.0),  # Higher boost for keyword field
                    Q("query_string", query=keyword_query, fields=["lossless_restatement^1.0"])
                ])
            )
            search = search[:top_k]
            
            # Execute search
            response = search.execute()
            
            # Convert results to MemoryEntry objects
            entries = []
            for hit in response:
                try:
                    source = hit.to_dict()
                    entry = MemoryEntry(
                        entry_id=source["entry_id"],
                        lossless_restatement=source["lossless_restatement"],
                        keywords=source.get("keywords", []),
                        timestamp=source.get("timestamp") or None,
                        location=source.get("location") or None,
                        persons=source.get("persons", []),
                        entities=source.get("entities", []),
                        topic=source.get("topic")
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse search result: {e}")
                    continue
            
            return entries
            
        except Exception as e:
            logger.error(f"Error during keyword search: {e}")
            return []
    
    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Symbolic Layer Search - Metadata-based deterministic filtering
        
        Paper Reference: Section 3.1
        Retrieves based on R_k = {(key, val)} for structured constraints
        Enables precise filtering by time, entities, persons, and locations
        
        Args:
            persons: Filter by person names
            timestamp_range: Filter by time range (start, end)
            location: Filter by location
            entities: Filter by entities
            top_k: Maximum number of results to return (default: no limit)
            
        Returns:
            List of MemoryEntry objects matching the filters
        """
        try:
            # Check if index is empty
            count = self.es.count(index=self.index_name)["count"]
            if count == 0:
                return []
            
            # If no filters provided, return empty
            if not any([persons, timestamp_range, location, entities]):
                return []
            
            # Build filter query
            must_clauses = []
            
            if persons:
                must_clauses.append(Q("terms", persons=persons))
            
            if location:
                must_clauses.append(Q("term", location=location))
            
            if entities:
                must_clauses.append(Q("terms", entities=entities))
            
            if timestamp_range:
                start_time, end_time = timestamp_range
                must_clauses.append(
                    Q("range", timestamp={
                        "gte": start_time,
                        "lte": end_time
                    })
                )
            
            if not must_clauses:
                return []
            
            # Build search query
            search = Search(using=self.es, index=self.index_name)
            search = search.query(Q("bool", must=must_clauses))
            
            if top_k is not None:
                search = search[:top_k]
            
            # Execute search
            response = search.execute()
            
            # Convert results to MemoryEntry objects
            entries = []
            for hit in response:
                try:
                    source = hit.to_dict()
                    entry = MemoryEntry(
                        entry_id=source["entry_id"],
                        lossless_restatement=source["lossless_restatement"],
                        keywords=source.get("keywords", []),
                        timestamp=source.get("timestamp") or None,
                        location=source.get("location") or None,
                        persons=source.get("persons", []),
                        entities=source.get("entities", []),
                        topic=source.get("topic")
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse search result: {e}")
                    continue
            
            return entries
            
        except Exception as e:
            logger.error(f"Error during structured search: {e}")
            return []
    
    def get_all_entries(self) -> List[MemoryEntry]:
        """
        Get all memory entries from Elasticsearch
        
        Returns:
            List of all MemoryEntry objects
        """
        try:
            # Use scroll API for large datasets
            entries = []
            scroll_size = 1000
            
            response = self.es.search(
                index=self.index_name,
                body={"query": {"match_all": {}}},
                scroll="2m",
                size=scroll_size
            )
            
            scroll_id = response.get("_scroll_id")
            
            while len(response["hits"]["hits"]) > 0:
                for hit in response["hits"]["hits"]:
                    try:
                        source = hit["_source"]
                        entry = MemoryEntry(
                            entry_id=source["entry_id"],
                            lossless_restatement=source["lossless_restatement"],
                            keywords=source.get("keywords", []),
                            timestamp=source.get("timestamp") or None,
                            location=source.get("location") or None,
                            persons=source.get("persons", []),
                            entities=source.get("entities", []),
                            topic=source.get("topic")
                        )
                        entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Failed to parse entry: {e}")
                        continue
                
                # Get next batch
                if scroll_id:
                    response = self.es.scroll(
                        scroll_id=scroll_id,
                        scroll="2m"
                    )
                else:
                    break
            
            # Clear scroll
            if scroll_id:
                self.es.clear_scroll(scroll_id=scroll_id)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error getting all entries: {e}")
            return []
    
    def clear(self):
        """
        Clear all data from Elasticsearch index
        """
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logger.info(f"Deleted Elasticsearch index: {self.index_name}")
            
            # Recreate index
            self._init_index()
            logger.info("Database cleared and reinitialized")
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
