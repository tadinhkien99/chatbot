#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    query_service.py
# @Author:      Kuro
# @Time:        2/27/2025 2:55 PM

from typing import List

from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient

from api.services.embedding_service import LLMEmbeddings


class MilvusSearch:
    def __init__(self, model):
        self.embedding_model = LLMEmbeddings(model)
        self.client = MilvusClient(uri="http://localhost:19530")
        self.search_params = {"metric_type": "L2", "params": {"itopk_size": 16, "search_width": 16, "team_size": 8}}

    def search(self, collection_name, data: List[str], top_k: int = 10, filter="", output_fields=None):
        embedding = self.embedding_model.embed_documents(data)
        result = self.client.search(collection_name=collection_name, data=embedding, limit=top_k, filter=filter,
                                    output_fields=output_fields, search_params=self.search_params)
        return result

