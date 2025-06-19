---
### 사용가이드    
---
1. https://nim-taskgen.azurewebsites.net/ 접속
2. RAG 데이터 업로드( /data/contents_software.txt  파일 다운 후, 업로드 기능 활용)  
3. '문서 분석' 기능 확인(업로드 문서 임베딩 > 인덱스 > 인덱서 > RAG 검색 자동화)
5. 문제 유형, 난이도, 개수, 상세 유형 등 조건 선택 후 '문제 생성'  
6. 생성된 문제 확인  
7. 엑셀 다운로드 기능 확인  
8. 문제 풀기 기능 확인  

---
### 설치환경  
---
<pre>pip install streamlit  
pip install openai  
pip install python-dotenv  
pip install Python==3.12.3  
pip install streamlit==1.45.1  
pip install azure-storage-blob  
pip install --upgrade azure-search-documents  
pip install scikit-learn  </pre>

---
### 배포 가이드  
---
#### 1. Azure Portal 내 'App Service' 리소스 생성  
#### 2. VSCode에서 "streamlit.sh"와 ".deployment" 파일 생성 후, 필요 환경 세팅
![image](https://github.com/user-attachments/assets/1e4cc0bd-64b5-4681-81f4-62b99cf812d2)  

streamlist.sh  
<pre>
pip install streamlit  
pip install openai  
pip install python-dotenv  
pip install Python==3.12.3  
pip install streamlit==1.45.1  
pip install azure-storage-blob
pip install --upgrade azure-search-documents
pip install scikit-learn

python -m streamlit run task_generator.py --server.port 8000 --server.address 0.0.0.0  
</pre>

.deployment  
<pre>
[config]  
SCM_DO_BUILD_DURING_DEPLOYMENT=false  
</pre>

#### 3. VSCode 내 'Azure App Service' Extension 설치
![image](https://github.com/user-attachments/assets/f559bd97-c44d-48ce-b45d-57b1e2d7afd1)

#### 4. 로컬 코드 APP Service에 배포하기  
(1) vscode > auzre extension > ms 로그인 및 인증 > "Sign in to Tenant" > 연결할 테넌트 선택
![image](https://github.com/user-attachments/assets/dbf0ae21-f27b-4b3a-b241-ba73a666f1d3)

(2) app services 리소스 선택 > 우클릭 > deploy to webapp  
![image](https://github.com/user-attachments/assets/26740060-0b59-4ed1-b7a9-a970941fc3e8)  
**재배포할 경우, 기존 코드 지우고 덮어씌어짐.

(3) 배포 완료  
![image](https://github.com/user-attachments/assets/4d56b459-d8bb-424e-9526-4ca65de089d6)

---
### 환경변수  
---
#### GPT API  
<pre>OPENAI_API_KEY = "EMar4isrsxWEypquPBauNuNPsOEoHBAJqvoQHj6RWlLfJG8Vu2KCJQQJ99BFACfhMk5XJ3w3AAABACOGhkVF"  
AZURE_ENDPOINT = "https://nim-openai-005.openai.azure.com/"  
OPENAI_API_TYPE = "azure"  
OPENA_API_VERSION = "2024-12-01-preview"  
GPT_DEPLOYMENT_NAME = "gpt-4o-mini"  </pre>

#### AI SEARCH API
<pre>SEARCH_KEY = "cQ5ieDeTzVEcUkt32PzduZ8l391LojW2Crygu1XEKFAzSeBe3HeE"  
SEARCH_ENDPOINT = "https://nim-search-005.search.windows.net"  
INDEX_NAME = "taskgetn-rag"  
VECTOR_DIMENSIONS = 1536  
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-small"  
INDEXER_NAME = "indexer"  
DATA_SOURCE_NAME = "taskgen-blob-datasource"  </pre>


#### BLOB API  
<pre>AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=nimstorage001;AccountKey=N2nYamdbhb7fKJaUxi9j8VttatWx4RcSVE0tgkcq4sg9R07WrNuXC5qMONT+qKOrc/YJ6xTyzs93+AStrvNrCQ==;EndpointSuffix=core.windows.net"  
AZURE_CONTAINER_NAME="taskgen"  </pre>

