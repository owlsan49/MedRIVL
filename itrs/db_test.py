# -*- coding: utf-8 -*-
# Copyright (c) 2022, Shang Luo
# All rights reserved.
# 
# Author: 罗尚
# Building Time: 2024/6/7
# Reference: None
# Description: None
import chromadb

client = chromadb.PersistentClient(path="test")

collection = client.get_or_create_collection(name="test", embedding_function=None)

collection.add(
    documents=["Article by john", "Article by Jack", "Article by Jill"],
    embeddings=[[1,2,3],[4,5,6],[7,8,9]],
    metadatas=[{"author": "john"}, {"author": "jack"}, {"author": "jill"}],
    ids=["1", "2", "3"])

result = collection.query(
    query_embeddings=[[1,2,3]],
    n_results=2,
)
print(result)

if __name__ == '__main__':
    ...
