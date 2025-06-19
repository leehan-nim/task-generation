from datetime import datetime
import os
from dotenv import load_dotenv
import openai
import streamlit as st
import pandas as pd
import re
from azure.storage.blob import BlobServiceClient

from contents_embedding import (
    read_blobs,
    remove_similar_questions_by_embedding,
    generate_rag_response,
    create_data_source,
    create_index,
    upload_documents,
    create_indexer,
    run_indexer,
)

# css 불러오기
with open("static/style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 환경변수 불러오기
load_dotenv()

# OpenAI library 초기화
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENA_API_VERSION")
gpt_deployment_name = os.getenv("GPT_DEPLOYMENT_NAME")
embedding_deployment_name = os.getenv("EMBEDDING_DEPLOYMENT_NAME")

# search 관련 설정
gpt_deployment_name = os.getenv("GPT_DEPLOYMENT_NAME")
embedding_deployment_name = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
search_key = os.getenv("SEARCH_KEY")
search_endpoint = os.getenv("SEARCH_ENDPOINT")
index_name = os.getenv("INDEX_NAME")
vector_dimensions = os.getenv("VECTOR_DIMENSIONS")
indexer_name = "indexer"
data_source_name = "taskgen-blob-datasource"

# Azure Storage 설정
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)


# gpt-4o-mini 모델 호출
def get_openai_response(messages):
    """
    Azure OpenAI API를 호출하여 응답을 가져오는 함수
    """
    try:
        response = openai.chat.completions.create(
            model=gpt_deployment_name, messages=messages, temperature=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# 텍스트 청크 함수
def chunk_text(text, chunk_size=7500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# Azure Blob 업로드 함수
def upload_chunk_to_blob(blob_service_client, container, base_filename, chunks):
    container_client = blob_service_client.get_container_client(container)
    uploaded_files = []
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    for i, chunk in enumerate(chunks, 1):
        blob_name = f"{base_filename}_chunk_{i}_{timestamp}.txt"
        container_client.upload_blob(name=blob_name, data=chunk, overwrite=True)
        uploaded_files.append(blob_name)

    return uploaded_files


# 응답 파서 함수(엑셀 다운로드용)
def parse_response(response):
    split_pattern = r"- (" + "|".join(fields) + r")\s*[:\-]?\s*"
    parts = re.split(split_pattern, response, flags=re.DOTALL)

    parsed = {field: "" for field in fields}
    i = 1
    while i < len(parts) - 1:
        field = parts[i].strip()
        value = parts[i + 1].strip()
        if field in parsed:
            parsed[field] = value
        i += 2
    return parsed


# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------
# 1. 문제 유형별 시스템 프롬프트 정의
# --------------------------
PROMPT_TEMPLATES = {
    "알고리즘": {
        "system_message": """당신은 전문적인 알고리즘 문제 출제자입니다.

        다음 형식으로 문제를 생성하세요.
        단, 입력과 출력 설명에 '-'(하이픈) 문자는 포함하지 마세요.
        - 문제제목:
        - 문제내용:
        - 입력조건:
        - 입력예시:
        - 출력조건:
        - 출력예시:
        - 답안:
        """,
        "fields": [
            "문제제목",
            "문제내용",
            "입력조건",
            "입력예시",
            "출력조건",
            "출력예시",
            "답안",
        ],
    },
    "정보처리기사": {
        "system_message": """당신은 정보처리기사 자격증 시험 출제 전문가입니다.

        다음 형식으로 객관식 문제를 생성하세요:
        - 문제제목:
        - 문제내용:
        - 보기:
        - 답안:
        """,
        "fields": ["문제제목", "문제내용", "보기", "답안"],
    },
}
st.title("Task Generator")

### UI: 2분할 구성
col1, col2 = st.columns([1, 2])  # 왼쪽 1, 오른쪽 2 비율

with col1:

    # 파일 업로드 영역
    st.subheader("📄 문서 업로드")

    uploaded_file = st.file_uploader(
        "코지가 학습할 문서를 선택해주세요!!", type=["pdf", "txt"]
    )

    # 파일 업로드 후 인덱싱 및 인덱서 생성
    if uploaded_file is not None:

        # -----------------------------------
        # 2. parser & chunk
        # -----------------------------------
        file_contents = uploaded_file.read().decode("utf-8")
        file_name = os.path.splitext(uploaded_file.name)[0]

        chunk_size = st.slider(
            "청크 크기 (문자 수)", min_value=200, max_value=7500, value=800, step=100
        )
        overlap = st.slider(
            "오버랩 (중복 문자 수)", min_value=0, max_value=500, value=100, step=50
        )
        
        if st.button("업로드", use_container_width=True):

            try:
                chunks = chunk_text(
                    file_contents, chunk_size=chunk_size, overlap=overlap
                )
                blob_service_client = BlobServiceClient.from_connection_string(
                    connection_string
                )
                uploaded_files = upload_chunk_to_blob(
                    blob_service_client, container_name, file_name, chunks
                )
                st.success(
                    f"✅ `{uploaded_file.name}` 파일이 Blob Container에 업로드되었습니다."
                )
            except Exception as e:
                st.error(f"❌ 업로드 실패: {str(e)}")

        if st.button("문서 분석", use_container_width=True):
            with st.spinner("문서를 분석하고 AI 검색 준비 중입니다..."):
                try:
                    create_data_source()
                    create_index()
                    docs = read_blobs()

                    upload_documents(docs)

                    create_indexer(
                        search_endpoint=search_endpoint,
                        search_key=search_key,
                        index_name=index_name,
                        data_source_name="taskgen-blob-datasource",
                        indexer_name="taskgen-indexer",
                    )

                    run_indexer(
                        search_endpoint=search_endpoint,
                        search_key=search_key,
                        indexer_name="taskgen-indexer",
                    )

                    st.success(
                        "✅ 문서 분석 완료! 이제 이 문서를 기반으로 문제를 생성할 수 있어요!"
                    )
                except Exception as e:
                    st.error(f"❌ 문서 처리 중 오류 발생: {str(e)}")

    ######### 구분선 추가 #########
    st.divider()
    ##############################

    # -----------------------------------
    # 2. Streamlit 입력 영역
    # -----------------------------------
    st.subheader("🚀 문제 생성")

    # 문제 유형 선택 드롭다운 추가
    problem_type = st.selectbox("문제 유형을 선택하세요:", ("정보처리기사", "알고리즘"))

    # 문제 난이도 선택 드롭다운 추가
    problem_level = st.selectbox(
        "문제 난이도를 선택하세요:", ("랜덤", "초급", "중급", "고급")
    )

    # 문제 개수 입력 받기
    problem_count = st.number_input(
        "생성할 문제 개수를 입력하세요:", min_value=0, max_value=100, value=2
    )

    detail = ""
    if problem_type == "정보처리기사":
        # 추가 내용 입력
        detail = st.text_input("(선택) 자신이 부족한 영역을 얘기해 보세요:")

    # -----------------------------------
    # 3. 문제 생성 버튼 클릭 시
    # -----------------------------------

    # 문제 생성 버튼
    if st.button("문제 생성"):

        # 프롬프트 설정 불러오기
        selected_prompt = PROMPT_TEMPLATES.get(problem_type)
        system_message = selected_prompt["system_message"]
        fields = selected_prompt["fields"]

        # 유저 입력 문장 구성
        if detail:
            user_input = f"주제: {detail}과 관련된 문제를 생성해 주세요."

        else:
            user_input = "문제 생성"

        st.session_state.messages.append({"role": "user", "content": user_input})

        assistant_responses = []

        with st.spinner("응답을 기다리는 중..."):
            for i in range(problem_count):
                # 문제마다 다른 메세지 생성
                dynamic_user_message = {
                    "role": "user",
                    "content": f"{i+1}번째 문제입니다. '{problem_type}' 유형의 '{problem_level}' 난이도로 '{detail}' 문제를 생성해 주세요. 문제 설명과 답안을 포함해 주세요.",
                }

                # system message 새로 구성
                message = [
                    {"role": "system", "content": system_message},
                    dynamic_user_message,
                ]

                # 유형에 따라 응답 함수 분기
                if problem_type == "알고리즘":
                    response = get_openai_response(message)
                elif problem_type == "정보처리기사":
                    response = generate_rag_response(message)
                else:
                    response = "❌ 지원되지 않는 문제 유형입니다."

                assistant_responses.append(response)
                st.chat_message("assistant").write(response)

            # 유형에 따라 응답 함수 분기
            parsed_data = [parse_response(r) for r in assistant_responses]
            df = pd.DataFrame(parsed_data)

            if problem_type == "알고리즘":
                # 데이터 정제
                df = df.applymap(
                    lambda x: (
                        x.replace("```python", "").replace("```", "")
                        if isinstance(x, str)
                        else x
                    )
                )

            df = remove_similar_questions_by_embedding(df, threshold=0.9)
            st.session_state["generated_df"] = df

# -----------------------------------
# 4. 문제 생성 결과 엑셀 다운로드
# -----------------------------------

with col2:
    st.subheader("📋 생성된 문제")

    if "generated_df" in st.session_state:

        df_clean = st.session_state["generated_df"]

        csv = st.session_state["generated_df"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "엑셀 다운로드",
            data=csv,
            file_name="generated_problems.csv",
            mime="text/csv",
        )
        st.dataframe(df_clean)
    else:
        st.info("왼쪽에서 문제를 생성하면 결과가 표시됩니다.")
        st.stop()  # 생성된 문제 없으면 아래 코드 실행 안 되도록 차단

    st.divider()

    # -----------------------------------
    # 5. 생성된 문제 풀기
    # -----------------------------------

    # 상태 초기화
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0

    if "user_answers" not in st.session_state or len(
        st.session_state.user_answers
    ) != len(df_clean):
        st.session_state.user_answers = [""] * len(df_clean)

    # 현재 문제 index
    idx = st.session_state.current_question
    row = df_clean.iloc[idx]

    st.subheader(f"문제 {idx + 1} / {len(df_clean)}: {row['문제제목']}")
    st.write(row["문제내용"])

    if "보기" in row and pd.notna(row["보기"]):
        cleaned_input = "\n".join(
            [line.strip() for line in str(row["보기"]).splitlines()]
        )
        st.markdown(f"\n```\n{cleaned_input}\n```")
    # 입력/출력 예시 (공백 제거)
    if "입력예시" in row and pd.notna(row["입력예시"]):
        cleaned_input = "\n".join(
            [line.strip() for line in str(row["입력예시"]).splitlines()]
        )
        st.markdown(f"**입력 예시:**\n```\n{cleaned_input}\n```")

    if "출력예시" in row and pd.notna(row["출력예시"]):
        cleaned_output = "\n".join(
            [line.strip() for line in str(row["출력예시"]).splitlines()]
        )
        st.markdown(f"**출력 예시:**\n```\n{cleaned_output}\n```")

    # 정답 입력 필드
    st.session_state.user_answers[idx] = st.text_area(
        label="✏️ 정답 입력:",
        key=f"user_answer_{idx}",
        value=st.session_state.user_answers[idx],
        height=100,
    )

    col1, col2, col3, _ = st.columns([1, 1, 1, 6])

    with col1:
        # 채점 버튼
        check = st.button("채점하기", use_container_width=True)

    # 페이지 이동
    with col2:
        if st.button("⬅️ 이전", disabled=(idx == 0), use_container_width=True):
            st.session_state.current_question = max(0, idx - 1)
    with col3:

        if st.button(
            "다음 ➡️", disabled=(idx == len(df_clean) - 1), use_container_width=True
        ):
            st.session_state.current_question = min(len(df_clean) - 1, idx + 1)

    if check:
        correct = str(row.get("답안", "")).strip()
        submitted = st.session_state.user_answers[idx].strip()

        if not correct:
            st.warning("정답 데이터가 없습니다. 채점할 수 없습니다.")
        elif submitted == correct:
            st.success("정답입니다! ✅")
        else:
            st.error("오답입니다 ❌")
            st.text(f"정답: {correct}")
