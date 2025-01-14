import os
import re
import stanza
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from langchain_openai import ChatOpenAI
import logging
import warnings

# DeprecationWarning을 무시하도록 설정
warnings.simplefilter("ignore", category=DeprecationWarning)

### Dependency parsing ###

# Suppress all logs, including INFO, WARNING, and download logs
logging.basicConfig(level=logging.CRITICAL)

# Global variable for processed DataFrame
global_df = pd.DataFrame()

def init_pipeline():
    """
    Stanza 파이프라인 초기화 함수.
    """
    global nlp
    nlp = stanza.Pipeline(
        lang='en',
        processors='tokenize,pos,lemma,depparse',
        use_gpu=True,
        batch_size=512,
        logging_level='ERROR'  # Suppress unnecessary logs from Stanza
    )

def process_text(text, base_s_id):
    """
    Stanza로 단일 텍스트를 처리하고 결과를 데이터프레임으로 반환합니다.

    Args:
        text (str): 입력 텍스트
        base_s_id (int): 문장 ID의 시작 값
    Returns:
        pd.DataFrame: 의존 파싱 결과 데이터프레임
    """
    global nlp
    data = []
    doc = nlp(text)
    for i, sentence in enumerate(doc.sentences):
        for word in sentence.words:
            data.append({
                "s_id": base_s_id + i,
                "id": word.id,
                "text": word.text,
                "lemma": word.lemma,
                "upos": word.upos,
                "xpos": word.xpos,
                "head": word.head,
                "deprel": word.deprel,
                "start_char": word.start_char,
                "end_char": word.end_char
            })
    return pd.DataFrame(data)

def parallel_processing(texts, num_processes, base_s_id):
    """
    병렬 처리를 통해 텍스트 청크를 Stanza로 처리합니다.

    Args:
        texts (list): 텍스트 리스트
        num_processes (int): 병렬 처리 프로세스 수
        base_s_id (int): 문장 ID의 시작 값
    Returns:
        pd.DataFrame: 병렬 처리된 데이터프레임
    """
    with Pool(num_processes, initializer=init_pipeline) as pool:
        results = pool.starmap(
            process_text,
            [(text, base_s_id + idx) for idx, text in enumerate(texts)]
        )
    return pd.concat(results, ignore_index=True)

def load_data_in_chunks(file_path, chunk_size, max_lines=None):
    """
    파일에서 텍스트 데이터를 청크 단위로 읽어옵니다.

    Args:
        file_path (str): 파일 경로
        chunk_size (int): 청크 크기
        max_lines (int, optional): 최대 읽을 줄 수
    Yields:
        list: 텍스트 청크
    """
    with open(file_path, "r", encoding="utf-8") as file:
        buffer = []
        total_lines = 0
        for line in file:
            line = line.strip()
            if line:
                buffer.append(line)
                total_lines += 1
                if max_lines and total_lines >= max_lines:
                    yield buffer
                    return
                if len(buffer) >= chunk_size:
                    yield buffer
                    buffer = []
        if buffer:
            yield buffer

def dp(file_path, max_lines=100000):
    """
    Dependency Parsing을 수행하고 진행 상황을 표시합니다.

    Args:
        file_path (str): 입력 텍스트 파일 경로
        max_lines (int, optional): 최대 처리할 문장 수 (기본값: 100000)
    """
    global global_df

    print("=== Dependency Parsing (DP) ===")
    print(f"File path: {file_path}")
    print(f"Sentence limit: {max_lines if max_lines else '제한 없음'}")

    chunk_size = 1
    num_processes = 8
    base_s_id = 1

    final_results = []
    chunks = list(load_data_in_chunks(file_path, chunk_size, max_lines=max_lines))
    total_chunks = len(chunks)

    # tqdm progress bar 초기화
    with tqdm(total=total_chunks, desc="", leave=True) as pbar:
        for idx, texts in enumerate(chunks):
            result_df = parallel_processing(texts, num_processes, base_s_id + idx * chunk_size)
            final_results.append(result_df)
            pbar.update(1)  # 매 청크 처리 후 업데이트

    global_df = pd.concat(final_results, ignore_index=True)
    total_sentences = global_df["s_id"].nunique()
    print("\nProcessing finished successfully.")
    print(f"\nTotal sentences processed: {total_sentences}")

    return global_df  # global_df를 반환


def preview(s_id):
    """
    처리된 문장의 결과를 출력합니다.

    Args:
        s_id (int): 확인할 문장의 s_id
    """
    global global_df

    if global_df.empty:
        print("No data available. Please run process() first.")
        return

    preview_df = global_df[global_df["s_id"] == s_id]

    if preview_df.empty:
        print(f"No sentence found for sentence ID {s_id}.")
    else:
        print(f"\n=== Dependency parsing results for Sentence ID {s_id} === \n")
        print(preview_df)

### langchain ###

def load_list_from_file(file_name):
    """
    지정된 파일에서 줄바꿈된 텍스트 리스트를 로드하는 함수.
    .txt 파일 형식과 줄바꿈된 텍스트 형식을 검사.

    Args:
        file_name (str): 로드할 파일 이름.

    Returns:
        list: 파일에서 불러온 텍스트 리스트.
    """
    # 파일 확장자 확인
    if not file_name.endswith(".txt"):
        raise ValueError(f"'{file_name}' is invalid. Only '.txt' files are allowed.")

    # 파일 존재 확인
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"'{file_name}' is not found in the current directory.")

    # 파일 로드
    with open(file_name, "r", encoding="utf-8") as file:
        loaded_list = [line.strip() for line in file if line.strip()]

    # 파일 내용 검사
    if not loaded_list:
        raise ValueError(f"'{file_name}' is empty or not in a line-separated text format.")

    print(f"Loaded {len(loaded_list)} items successfully from '{file_name}'.")
    return loaded_list


def extract_file_name(natural_language_query):
    """
    자연어 명령에서 파일 이름을 추출하는 함수.

    Args:
        natural_language_query (str): 사용자 입력 자연어 명령.

    Returns:
        str: 추출된 파일 이름 (없을 경우 None 반환).
    """
    match = re.search(r"(\S+\.txt)", natural_language_query)
    if match:
        return match.group(1)
    return None


def query_data(df, query, api_key, output_file=None):
    """
    자연어 질의를 입력받아 조건문을 생성하고 데이터를 필터링한 후 문장을 반환하는 함수.

    Args:
        df (pd.DataFrame): 데이터프레임
        query (str): 자연어 질의
        api_key (str): OpenAI API 키
        output_file (str): 결과를 저장할 파일 경로
    """
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key, max_tokens=500)
    prompt = f"""
              Convert the following natural language query into a Python Pandas DataFrame condition.
              The condition must include the DataFrame name 'df' and be in an executable form.

              - Respond only with Python code, and do not include any explanations.
              - The generated condition must be executable in Pandas.
              - Write the code in a single line whenever possible.
              - If there is a phrase with {{file_name}}', use 'file_list' as the variable.

              Natural language query: {query}

              The DataFrame has the following columns:
                  - s_id: Sentence ID (unique identifier for each sentence).
                  - id: Word ID within a sentence (starting from 1).
                  - text: Surface form of the word (e.g., 'read').
                  - lemma: Base form of the word (e.g., 'reads' → 'read').
                  - upos: Universal part-of-speech tag (e.g., 'NOUN', 'VERB').
                  - xpos: Language-specific part-of-speech tag (e.g., 'NN', 'VBZ').
                  - head: ID of the head word this word depends on.
                  - deprel: Dependency relation label (e.g., 'nsubj', 'obj', 'root').
                  - start_char, end_char: Character offsets for the word's position in the original text.

              Condition:
              """

    try:
        # 파일 이름 추출
        file_name = extract_file_name(query)
        file_list = []

        if file_name:
            try:
                # 파일이 지정된 경우 로드
                file_list = load_list_from_file(file_name)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error loading file: {e}")
                return None

        # GPT-4로부터 조건문 생성
        response = llm(prompt)
        condition = response.content.strip()  # 응답 내용 추출

        # 디버깅: 생성된 조건문 출력
        print(f"Generated Condition Code: {condition}")

        # 조건문 평가 (eval 사용)
        try:
            # 조건문 실행 결과
            context = {"df": df, "file_list": file_list}  # 필요한 변수 포함
            result = eval(condition, context)

            # 조건문 결과 처리
            if isinstance(result, pd.Series):  # Boolean Series 또는 일반 Series
                if result.dtype == "bool":  # Boolean Series
                    filtered_ids = result.index[result]  # True인 s_id만 추출
                else:  # 일반 Series (e.g., groupby().size())
                    filtered_ids = result.index[result > 0]  # 조건 만족하는 s_id 추출
            elif isinstance(result, pd.DataFrame):  # DataFrame
                filtered_ids = result["s_id"].unique()
            else:
                print("The condition result is invalid.")
                return None

            # 조건에 맞는 s_id로 데이터프레임 필터링
            filtered_df = df[df["s_id"].isin(filtered_ids)]

            # 조건에 맞는 데이터가 없는 경우 처리
            if filtered_df.empty:
                print("No data matches the condition.")
                return None

        except Exception as eval_error:
            print(f"Condition Evaluation Error: {eval_error}")
            print(f"Condition Attempted to Execute: {condition}")
            return None

        # 문장 조합 함수
        def combine_text(group):
            """
            그룹화된 데이터에서 단어를 조합해 문장을 생성합니다.

            Args:
                group (pd.DataFrame): s_id로 그룹화된 데이터프레임
            Returns:
                str: 조합된 문장
            """
            words = group["text"].tolist()
            upos = group["upos"].tolist()
            if upos[-1] == "PUNCT":  # 문장 끝에 구두점이 있으면 붙이기
                return " ".join(words[:-1]) + words[-1]
            else:
                return " ".join(words)

        # 필요한 열만 선택하여 그룹화
        sentences = (
            filtered_df.groupby("s_id", group_keys=False)
            .apply(combine_text)
            .reset_index(name="sentence")
        )

        # 결과 반환
        return sentences

    except Exception as e:
        print(f"Error message: {e}")
        return None


def query_data_batch(df, query, api_key, batch_size=1000, output_file=None):
    """
    데이터를 s_id 기준으로 배치 사이즈로 나누어 API를 반복 호출하여 처리합니다.
    """
    total_rows = len(df)
    all_sentences = []  # 모든 결과를 모을 리스트

    # s_id 기준으로 데이터를 배치 처리
    unique_s_ids = df["s_id"].unique()  # 고유한 s_id 값들

    # 데이터를 s_id 기준으로 나누어 처리
    for start in range(0, len(unique_s_ids), batch_size):
        end = min(start + batch_size, len(unique_s_ids))
        batch_s_ids = unique_s_ids[start:end]

        # batch_s_ids에 해당하는 데이터만 필터링
        batch_df = df[df["s_id"].isin(batch_s_ids)]

        # query_data 호출
        sentences = query_data(batch_df, query, api_key, output_file)

        if sentences is not None:
            all_sentences.append(sentences)
        else:
            print(f"Batch from s_id {batch_s_ids[0]} to s_id {batch_s_ids[-1]} did not return any results.")
            continue

    # 모든 배치 결과 합치기
    final_sentences = pd.concat(all_sentences, ignore_index=True)

    # 결과 반환
    return final_sentences



def user_query_interface(df, api_key):
    """
    사용자 채팅 기반 데이터 질의 인터페이스 함수.
    Args:
        df (pd.DataFrame): 데이터프레임
        api_key (str): OpenAI API 키
    """
    print("=== Chat Interface ===")

    include_conditions = []  # 포함 조건 리스트
    exclude_conditions = []  # 제외 조건 리스트

    # 포함 조건 입력
    print("\nProvide inclusion condition. Press Enter to finish your input.")
    while True:
        include_query = input("Inclusion: ").strip()
        if include_query == "":  # 빈 입력으로 종료
            break
        include_conditions.append(include_query)

    # 제외 조건 입력
    print("\nProvide exclusion condition. Press Enter to finish your input.")
    while True:
        exclude_query = input("Exclusion: ").strip()
        if exclude_query == "":  # 빈 입력으로 종료
            break
        exclude_conditions.append(exclude_query)

    print("\nProcessing condition filtering...")

    try:
        # 포함 조건 처리
        filtered_df = df.copy()
        for include_query in include_conditions:
            sentences = query_data_batch(filtered_df, include_query, api_key=api_key, batch_size=1000)  # 배치 처리
            if sentences is None:
                print("No data matches the inclusion conditions.")
                return
            filtered_df = filtered_df[filtered_df["s_id"].isin(sentences["s_id"])]  # 결과 반영

        # 제외 조건 처리
        for exclude_query in exclude_conditions:
            sentences = query_data_batch(filtered_df, exclude_query, api_key=api_key, batch_size=1000)  # 배치 처리
            if sentences is not None:  # 제외 조건에 맞는 데이터가 있을 경우
                filtered_df = filtered_df[~filtered_df["s_id"].isin(sentences["s_id"])]  # 제외 조건 반영

        # 결과 조합
        final_sentences = (
            filtered_df.groupby("s_id", group_keys=False)
            .apply(lambda group: " ".join(group["text"].tolist()))
            .reset_index(name="sentence")
        )

        total_sentences = len(final_sentences)
        print(f"\nTotal filtered sentences: {total_sentences}")

        # 미리보기 최대 10개
        print("\nFiltered sentences preview (up to 10):")
        preview_sentences = final_sentences.head(10)
        for _, row in preview_sentences.iterrows():
            print(f"{row['s_id']}, {row['sentence']}")

        # 결과 저장
        save_response = input("\nDo you want to save the results? (yes/no): ").strip().lower()
        if save_response in ["yes", "y"]:
            while True:
                save_path = input("Enter the file name to save (e.g., result.txt/csv): ").strip()
                if save_path.endswith(".csv"):
                    final_sentences.to_csv(save_path, index=False)
                    print(f"The results have been saved to '{save_path}'.")
                    break
                elif save_path.endswith(".txt"):
                    with open(save_path, "w", encoding="utf-8") as f:
                        for _, row in final_sentences.iterrows():
                            f.write(f"{row['s_id']}, {row['sentence']}\n")
                    print(f"The results have been saved to '{save_path}'.")
                    break
                else:
                    print("Invalid file format. The file name must end with .csv or .txt.")
        elif save_response in ["no", "n"]:
            print("\nThe results have not been saved.")
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    except Exception as e:
        print(f"Error: {e}")


def chat(api_key):
    """
    채팅 인터페이스를 실행하는 함수.

    Args:
        api_key (str): OpenAI API 키
    """

    if global_df.empty:
        print("데이터가 없습니다. 먼저 process()를 실행하여 데이터를 로드하세요.")
        return

    user_query_interface(global_df, api_key)
