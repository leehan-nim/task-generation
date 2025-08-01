import os

import requests
import openai
import pandas as pd
import re

from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    SearchField,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    SearchFieldDataType,
    AzureOpenAIVectorizerParameters,
    SearchIndexer,
    SearchIndexerDataSourceConnection,
)
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# ── 설정 및 클라이언트 초기화 ─────────────────────────────────────────

load_dotenv()

# openai 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENA_API_VERSION")
gpt_deployment_name = os.getenv("GPT_DEPLOYMENT_NAME")
embedding_deployment_name = os.getenv("EMBEDDING_DEPLOYMENT_NAME")

# ai search 설정
search_key = os.getenv("SEARCH_KEY")
search_endpoint = os.getenv("SEARCH_ENDPOINT")
index_name = os.getenv("INDEX_NAME")
indexer_name = os.getenv("INDEXER_NAME")
data_source_name = os.getenv("DATA_SOURCE_NAME")

# Azure Storage 설정
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# 인증 객체 생성
credential = AzureKeyCredential(search_key)

# 클라이언트 생성
index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
indexer_client = SearchIndexerClient(endpoint=search_endpoint, credential=credential)

# ── 핵심 기능 함수 ─────────────────────────────────────────

# Blob 문서 READ 함수
def read_blobs():
    documents = []
    for blob in container_client.list_blobs():
        blob_client = container_client.get_blob_client(blob)
        content = blob_client.download_blob().readall().decode("utf-8")
        documents.append({"id": blob.name, "content": content})
    return documents


# 임베딩 함수
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        input=texts,
        model=embedding_deployment_name,
    )
    return [d.embedding for d in response.data]


# 데이터 소스 등록 함수
def create_data_source():
    indexer_client = SearchIndexerClient(
        endpoint=search_endpoint, credential=AzureKeyCredential(search_key)
    )

    data_source = SearchIndexerDataSourceConnection(
        name=data_source_name,
        type="azureblob",
        connection_string=connection_string,
        container={"name": container_name},
        description="Blob data source for taskgen documents",
    )

    indexer_client.create_or_update_data_source_connection(data_source)


# 인덱스 생성 함수
def create_index():
    """

    검색할 문서를 저장·구조화

    """
    index_client = SearchIndexClient(
        endpoint=search_endpoint, credential=AzureKeyCredential(search_key)
    )

    vector_dimensions = 1536

    fields = [
        SimpleField(name="id", type="Edm.String", key=True),
        SearchableField(name="content", type="Edm.String"),
        SearchField(
            name="contentvector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            hidden=False,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name="taskgenHnswProfile",
        ),
    ]
    # 벡터 검색을 위한 설정
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="taskgenHnsw")],
        profiles=[
            VectorSearchProfile(
                name="taskgenHnswProfile",
                algorithm_configuration_name="taskgenHnsw",
                vectorizer_name="taskgenVectorizer",
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="taskgenVectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=openai.azure_endpoint,
                    deployment_name=embedding_deployment_name,
                    model_name=embedding_deployment_name,
                    api_key=openai.api_key,
                ),
            )
        ],
    )

    # 인덱스 생성
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

    # 인덱스가 존재하지 않으면 새로 생성
    if index_name not in [i.name for i in index_client.list_indexes()]:
        index_client.create_index(index)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

    print("인덱스 생성까지 완료")


# 파일명 전처리 함수
def sanitize_id(filename: str) -> str:
    # 확장자 제거
    base = os.path.splitext(filename)[0]
    return re.sub(r"[^a-zA-Z0-9_\-=]", "_", base)


# AI Search 인덱스 문서 업로드 함수
def upload_documents(documents):
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key),
    )
    contents = [doc["content"] for doc in documents]
    vectors = get_embeddings(contents)

    enriched_docs = []
    for doc, vector in zip(documents, vectors):
        safe_id = sanitize_id(doc["id"])  # 안전한 ID로 변환
        enriched_docs.append(
            {"id": safe_id, "content": doc["content"], "contentvector": vector}
        )
    result = search_client.upload_documents(documents=enriched_docs)
    print(result)


# 인덱서 생성 함수
def create_indexer(    
    search_endpoint, search_key, index_name, data_source_name, indexer_name
):
    """
    실제 데이터 소스를 읽어와 인덱스에 문서를 채워 넣는 ‘색인 실행기’
    """
    credential = AzureKeyCredential(search_key)
    indexer_client = SearchIndexerClient(
        endpoint=search_endpoint, credential=credential
    )

    indexer = SearchIndexer(
        name=indexer_name,
        data_source_name=data_source_name,
        target_index_name=index_name,
    )

    try:
        # 인덱서 생성
        indexer_client.create_or_update_indexer(indexer)
        print(f"✅ 인덱서 '{indexer_name}' 생성 또는 업데이트 완료")
    except Exception as e:
        print(f"❌ 인덱서 생성 실패: {e}")


# 인덱서 실행 함수
def run_indexer(search_endpoint, search_key, indexer_name):
    credential = AzureKeyCredential(search_key)
    indexer_client = SearchIndexerClient(
        endpoint=search_endpoint, credential=credential
    )

    indexer_client.run_indexer(indexer_name)
    print(f"🚀 인덱서 '{indexer_name}' 수동 실행됨")


# 벡터 기반 RAG 검색 함수 (Azure AI Search)
def search_similar_documents(query: str, k=3) -> list[str]:
    try:
        # 쿼리 벡터 얻기
        query_vector = get_embeddings([query])[0]

        # REST API 호출 설정
        url = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2023-07-01-Preview"
        headers = {"Content-Type": "application/json", "api-key": search_key}
        payload = {
            "vectors": [{"value": query_vector, "fields": "contentvector", "k": k}],
            "top": k,
            "select": "content",
        }

        # POST 요청
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            print("❌ REST 검색 실패:", response.status_code, response.text)
            return []

        # 결과 파싱
        result_json = response.json()
        contents = [doc["content"] for doc in result_json.get("value", [])]

        print("📄 검색 결과:", contents)
        return contents

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return []
    

# RAG기반 질문 응답 함수
def generate_rag_response(messages: list[dict]) -> str:
    try:
        # user의 질문 추출
        user_query = next((m["content"] for m in messages if m["role"] == "user"), None)

        if not user_query:
            return "❌ 사용자 질문이 없습니다."

        # 유사 문서 검색
        print("질문내용 포함 gpt에게 던지는 값", user_query)
        contexts = search_similar_documents(user_query)

        # 문서 내 내용이 없을 경우 안내
        if not contexts:
            return "❗ 관련된 문서를 찾을 수 없어 정확한 답변을 드리기 어렵습니다."

        context_text = "\n\n".join(contexts)

        print("📄 문서 context:", context_text[:200])

        # context를 system message에 추가
        enhanced_messages = [
            {
                "role": "system",
                "content": f"다음 문서를 참고하여 정확하게 답변하세요:\n\n{context_text}",
            },
            *messages,
        ]

        # GPT 호출
        response = openai.chat.completions.create(
            model=gpt_deployment_name,
            messages=enhanced_messages,
            temperature=0.7,
        )
        print("GPT 응답:", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ GPT 응답 생성 실패: {e}")
        return "❌ 답변 생성 중 오류가 발생했습니다."


# 유사도 기반 중복 제거 함수(유사 문제 생성 방지)
def remove_similar_questions_by_embedding(
    df: pd.DataFrame, threshold: float = 0.9
) -> pd.DataFrame:
    texts = df["문제내용"].tolist()
    embeddings = get_embeddings(texts)

    similarity_matrix = cosine_similarity(embeddings)
    seen = set()
    keep_indices = []

    for i in range(len(texts)):
        if i in seen:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] > threshold:
                seen.add(j)

    return df.iloc[keep_indices].reset_index(drop=True)
