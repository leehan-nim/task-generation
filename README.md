### 사용가이드    
1. https://nim-taskgen.azurewebsites.net/ 접속
2. RAG 데이터 업로드(*contents_software.txt 파일 다운 후, 업로드 기능 활용)  
3. '문서 분석' 기능 확인(업로드 문서 임베딩 > 인덱스 > 인덱서 > RAG 검색 자동화)
   * 인덱스 생성 소요 시간 : 약 40s  
5. 문제 유형, 난이도, 개수, 상세 유형 등 조건 선택 후 '문제 생성'  
6. 생성된 문제 확인  
7. 엑셀 다운로드 기능 확인  
8. 문제 풀기 기능 확인  

### RAG 사용할 원본 데이터  
task-generation/data/contents_software  

### 설치환경  
pip install streamlit  
pip install openai  
pip install python-dotenv  
pip install Python==3.12.3  
pip install streamlit==1.45.1  
pip install azure-storage-blob  
pip install --upgrade azure-search-documents  
pip install scikit-learn  

### 환경변수  
#### GPT API  
OPENAI_API_KEY = "EMar4isrsxWEypquPBauNuNPsOEoHBAJqvoQHj6RWlLfJG8Vu2KCJQQJ99BFACfhMk5XJ3w3AAABACOGhkVF"  
AZURE_ENDPOINT = "https://nim-openai-005.openai.azure.com/"  
OPENAI_API_TYPE = "azure"  
OPENA_API_VERSION = "2024-12-01-preview"  
GPT_DEPLOYMENT_NAME = "gpt-4o-mini"  

#### AI SEARCH API
SEARCH_KEY = "cQ5ieDeTzVEcUkt32PzduZ8l391LojW2Crygu1XEKFAzSeBe3HeE"  
SEARCH_ENDPOINT = "https://nim-search-005.search.windows.net"  
INDEX_NAME = "taskgetn-rag"  
VECTOR_DIMENSIONS = 1536  
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-small"  
INDEXER_NAME = "indexer"  
DATA_SOURCE_NAME = "taskgen-blob-datasource"  


#### BLOB API  
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=nimstorage001;AccountKey=N2nYamdbhb7fKJaUxi9j8VttatWx4RcSVE0tgkcq4sg9R07WrNuXC5qMONT+qKOrc/YJ6xTyzs93+AStrvNrCQ==;EndpointSuffix=core.windows.net"  
AZURE_CONTAINER_NAME="taskgen"  

