### 설치환경  
pip install streamlit  
pip install openai  
pip install python-dotenv  
pip install Python==3.12.3  
pip install streamlit==1.45.1  
pip install azure-storage-blob  
pip install --upgrade azure-search-documents  
pip install scikit-learn  

### RAG 데이터  
task-generation/data/contents_software  

### 실행방법    
1. https://nim-taskgen.azurewebsites.net/ 접속
2. RAG 데이터 업로드(*contents_software.txt 파일 다운 후, 업로드 기능 활용)  
3. '문서 분석' 기능 확인(업로드 문서 임베딩 > 인덱스 > 인덱서 > RAG 검색)  
4. 문제 유형, 난이도, 개수, 상세 유형 등 조건 선택  
5. 생성된 문제 확인  
6. 엑셀 다운로드 기능 확인  
7. 문제 풀기 기능 확인  

