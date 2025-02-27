#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    embedding_service.py
# @Author:      Kuro
# @Time:        2/27/2025 2:44 PM


from FlagEmbedding import BGEM3FlagModel
from pymilvus import connections, utility


class LLMEmbeddings:
    def __init__(self, model):
        self.embedding_model = BGEM3FlagModel(model, use_fp16=True)

    def embed_documents(self, texts):
        embeddings = self.embedding_model.encode(texts, batch_size=1, max_length=512)['dense_vecs']
        return embeddings


class MilvusCollection:
    def __init__(self, host, port: int, collection_name: str, dimension: int, index_param: dict, batch_size: int, batch_size_insert: int):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_param = index_param
        self.batch_size = batch_size
        self.batch_size_insert = batch_size_insert

        self.collection = None

    def connect(self):
        connections.connect(host=self.host, port=self.port)

    def drop_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

    def create_collection(self, fields, field_name):
        self.collection.load()

    def insert(self, data):
        self.collection.insert(data)

    def upsert(self, data):
        self.collection.upsert(data)

    def delete(self, ids):
        self.collection.delete(ids)

    def flush(self):
        self.collection.flush()
