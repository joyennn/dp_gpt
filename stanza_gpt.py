import os
import re
import stanza
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from langchain_openai import ChatOpenAI
import logging

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
        batch_size=1024,
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

    chunk_size = 100
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


def preview(df, s_id):
    """
    처리된 문장의 결과를 출력합니다.

    Args:
        df (pd.DataFrame): 데이터프레임
        s_id (int): 확인할 문장의 s_id
    """
    if df.empty:
        print("No data available. Please run dp() first.")
        return

    preview_df = df[df["s_id"] == s_id]

    if preview_df.empty:
        print(f"No sentence found for sentence ID {s_id}.")
    else:
        print(f"\n=== Dependency parsing results for Sentence ID {s_id} === \n")
        print(preview_df)

import os
import re
import pandas as pd
from langchain_openai import ChatOpenAI
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # UserWarning 경고 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)  # DeprecationWarning 경고 무시


def query_data(query, api_key):
    """
    자연어 질의를 입력받아 조건문을 생성하고 데이터를 필터링한 후 문장을 반환하는 함수.

    Args:
        df (pd.DataFrame): 데이터프레임
        query (str): 자연어 질의
        api_key (str): OpenAI API 키

    Returns:
        str: 최종적으로 생성된 필터링 코드 (Pandas 실행 가능 코드)
    """

    llm = ChatOpenAI(temperature=0, openai_api_key=api_key, max_tokens=500)

    prompt = f"""
              Convert the following natural language query into a Python Pandas DataFrame condition.
              - The condition should be executable in Pandas.
              - Include 'df' as the name of Pandas DataFrame.
              - Respond only with Python code, and do not include any explanations.
              - Write the code in a single line whenever possible.
              - Ensure that the generated code strictly follows the given pattern.


              The 'df' has the following columns:
                  - s_id: Sentence ID (unique identifier for each sentence) - int
                  - id: Word ID within a sentence (starting from 1) - int
                  - text: Surface form of the word (e.g., 'read') - str
                  - lemma: Base form of the word (e.g., 'reads' → 'read') - str
                  - upos: Universal part-of-speech tag (e.g., 'NOUN', 'VERB') - str
                  - xpos: Language-specific part-of-speech tag (e.g., 'NN', 'VBZ') - str
                  - head: ID of the head word this word depends on - int
                  - deprel: Dependency relation label (e.g., 'nsubj', 'obj', 'root') - str
                  - start_char, end_char: Character offsets for the word's position in the original text - int


              Code generation patterns:
              (The column name is 'key' and its value is 'value'.)
              - When the key has a value which belongs to the .txt file: `df[df['key'].isin(open('file.txt').read().splitlines())]`
              - When comparing values between two columns : df[df['key1'] == 'value1', 'key2'].isin(df[df['key3'] == 'value3', 'key4'])


              Natural language query: {query}


              Condition: """


#     prompt = f"""
#               Convert the following natural language query into a Python Pandas DataFrame condition.
#               - The condition should be executable in Pandas.
#               - Include 'df' as the name of Pandas DataFrame.
#               - Respond only with Python code, and do not include any explanations.
#               - Write the code in a single line whenever possible.

#               Code generation guide:
#               (The column name is 'key' and its value is 'value'.)
#               - when the key has a value: df[df['key'] == 'value']
#               - when key has a list of values: df[df['key'].isin(['value1', 'value2', ...])]
#               - when the number of values of the key is identical or more(fewer) than n : df[df['key'] == 'value'].shape[0] >= n (>, <, =>, =<, ==, !=)
#               - when the key2 has a value2 when key1 has a value1: df[df['key1'] == 'value1']['key2'] == 'value2']
#               - when the key has a value which belongs to the .txt file: df[df['key'].isin(open('file.txt').read().splitlines())]
#               - compound conditions (AND): df[(df['key1'] == 'value1') & (df['key2'] == 'value2')]
#               - compound conditions (OR): df[(df['key1'] == 'value1') | (df['key2'] == 'value2')]
#               - when one condition depends on another column's values: df[df['key1'] == 'value1']['key2'].values == df[df['key3'] == 'value3']['key4']].values


#               The 'df' has the following columns:
#                   - s_id: Sentence ID (unique identifier for each sentence).
#                   - id: Word ID within a sentence (starting from 1).
#                   - text: Surface form of the word (e.g., 'read').
#                   - lemma: Base form of the word (e.g., 'reads' → 'read').
#                   - upos: Universal part-of-speech tag (e.g., 'NOUN', 'VERB').
#                   - xpos: Language-specific part-of-speech tag (e.g., 'NN', 'VBZ').
#                   - head: ID of the head word this word depends on.
#                   - deprel: Dependency relation label (e.g., 'nsubj', 'obj', 'root').
#                   - start_char, end_char: Character offsets for the word's position in the original text.

#               Natural language query: {query}

#               Condition:
# """

    try:
        # GPT-4로부터 조건문 생성
        response = llm(prompt)
        condition = response.content.strip()  # 응답 내용 추출

        # 생성된 조건 코드 반환
        return condition

    except Exception as e:
        print(f"Error: {e}")
        return None



# 전역 변수로 조건 코드 리스트 저장


def chat(api_key):
    """
    채팅 인터페이스를 실행하는 함수.
    자연어로 조건을 생성하고 필터링 조건을 리스트에 저장합니다.

    Args:
        api_key (str): OpenAI API 키
    """

    global chat_api_key, inclusion_codes, exclusion_codes
    chat_api_key = api_key  # API 키를 전역으로 설정

    # if global_df.empty:
    #     print("데이터가 없습니다. 먼저 dp()를 실행하여 데이터를 로드하세요.")
    #     return

    print("=== Chat Interface ===")

    # 조건을 저장할 리스트 초기화
    inclusion_codes = []
    exclusion_codes = []

    # 조건을 입력받는 루프
    while True:
        include_conditions = []  # 포함 조건 리스트
        exclude_conditions = []  # 제외 조건 리스트

        # 포함 조건 입력
        print("\nProvide inclusion condition. Press Enter to finish your input.")
        while True:
            include_query = input("Inclusion: ").strip()
            if include_query == "":  # 빈 입력으로 종료
                break
            if include_query.lower() == "exit":  # exit 입력 시 종료
                print("Exiting chat...")
                return [], []  # ✅ 빈 리스트 반환 (예외 방지)
            include_conditions.append(include_query)

        # 제외 조건 입력
        print("\nProvide exclusion condition. Press Enter to finish your input.")
        while True:
            exclude_query = input("Exclusion: ").strip()
            if exclude_query == "":  # 빈 입력으로 종료
                break
            if exclude_query.lower() == "exit":  # exit 입력 시 종료
                print("Exiting chat...")
                return [], []  # ✅ 빈 리스트 반환 (예외 방지)
            exclude_conditions.append(exclude_query)

        print("\nGenerating condition codes...")

        # inclusion 조건에 대해 생성된 코드 출력 및 저장
        print("\n-Generated Code for Inclusion Conditions:")
        files_loaded = False  # 파일 로드 여부 체크

        # 포함 조건에 대해 생성된 코드 출력 및 저장
        for include_query in include_conditions:
            include_code = query_data(include_query, api_key)  # 생성된 Pandas 코드
            print(f"{include_code}")  # 생성된 코드 출력
            inclusion_codes.append(include_code)  # 코드 리스트에 저장

        # exclusion 조건에 대해 생성된 코드 출력 및 저장
        print("\n-Generated Code for Exclusion Conditions:")
        for exclude_query in exclude_conditions:
            exclude_code = query_data(exclude_query, api_key)  # 생성된 Pandas 코드
            print(f"\n{exclude_code}")  # 생성된 코드 출력
            exclusion_codes.append(exclude_code)  # 코드 리스트에 저장

        # 사용자에게 필터링 진행 여부 확인
        user_choice = input("\nDo you want to apply these conditions and filter the data? (yes/no): ").strip().lower()

        if user_choice == "exit":  # exit 입력 시 종료
            print("Exiting chat...")
            return [], []  # ✅ 빈 리스트 반환

        if user_choice in ["yes", "y"]:
            print(inclusion_codes)
            print(exclusion_codes)
            print("\nThe codes have been generated and saved. You can use them later in the extract() function.")
            print("The filter conditions are saved and ready to be applied.")
            return inclusion_codes, exclusion_codes  # ✅ 정상적으로 리스트 반환
        elif user_choice in ["no", "n"]:
            # 이전에 생성된 코드들 초기화
            inclusion_codes = []
            exclusion_codes = []
            include_conditions = []  # 포함 조건 리스트 초기화
            exclude_conditions = []  # 제외 조건 리스트 초기화
            print("\nPlease provide new inclusion and exclusion conditions.")
            continue  # 다시 루프 실행
        else:
            print("Invalid input. Please enter 'yes', 'no', or 'exit'.")

    # ✅ 기본적으로 빈 리스트 반환 (예외 발생 방지)
    return inclusion_codes, exclusion_codes

import warnings
import numpy as np

# 경고를 무시하도록 설정
warnings.filterwarnings("ignore", category=UserWarning)  # UserWarning 경고 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)  # DeprecationWarning 경고 무시

# 파일을 불러오고 필터링 조건에 사용
def extract(df, inclusion_codes, exclusion_codes):
    """
    inclusion_codes와 exclusion_codes에서 생성된 조건을 이용하여 데이터를 필터링합니다.
    필터링된 데이터를 s_id 기준으로 저장합니다.

    Args:
        df (pd.DataFrame): 분석된 데이터프레임

    Returns:
        pd.DataFrame: 필터링된 데이터프레임
    """
    # global inclusion_codes, exclusion_codes

    sentences = []
    filtered_df = df.copy()  # df를 복사하여 시작


    # 포함 조건 적용
    for include_code in inclusion_codes:
        try:
          if 'df[(' in include_code or 'df[df' in include_code:
              include_code = include_code.replace('df[(', 'df.loc[(')
              include_code = include_code.replace('df[df', 'df.loc[df')
              filtered_df = filtered_df.groupby('s_id').apply(
                            lambda group: [eval(include_code.replace('df', 'group'))]
                            ).reset_index()




          else:
            filtered_df = filtered_df.groupby('s_id').apply(
                          lambda group: [eval(include_code.replace('df', 'group'))]
                          if len(eval(include_code.replace('df', 'group'))) > 0
                          else False
                          ).reset_index()

        except Exception as e:
            print(f"Error applying inclusion code: {include_code} \n{e}")



    # 제외 조건 적용
    for exclude_code in exclusion_codes:
        try:
          if 'df[(' or 'df[df' in exclude_code:
            exclude_code = exclude_code.replace('df[(', 'df.loc[(')
            exclude_code = exclude_code.replace('df[df', 'df.loc[df')
            filtered_df = filtered_df.groupby('s_id').apply(
                          lambda group: [~eval(exclude_code.replace('df', 'group'))]
                          ).reset_index()

          else:
            filtered_df = filtered_df.groupby('s_id').apply(
                          lambda group: [~eval(exclude_code.replace('df', 'group'))]
                          ).reset_index()

        except Exception as e:
            print(f"Error applying inclusion code: {include_code} \n{e}")


    unique_s_id = filtered_df['s_id'].unique()
    final_df = df[df['s_id'].isin(unique_s_id)]


    def combine_text(group):
        """
        그룹화된 데이터에서 단어를 조합해 문장을 생성합니다.

        Args:
            group (pd.DataFrame): s_id로 그룹화된 데이터프레임
        Returns:
            str: 조합된 문장
        """
        words = group["text"].tolist()  # 각 행의 'text'를 리스트로 변환
        upos = group["upos"].tolist()  # 각 행의 'upos'를 리스트로 변환

        # 문장 끝에 구두점이 있으면 붙여서 문장을 반환
        if upos[-1] == "PUNCT":  # 마지막 단어가 구두점이면
            return " ".join(words[:-1]) + " " + words[-1]
        else:
            return " ".join(words)  # 구두점이 없으면 그냥 단어들을 합침


    sentence = (
    final_df.groupby("s_id", group_keys=False)  # 's_id' 기준으로 그룹화
    .apply(combine_text)  # 그룹별로 문장 생성
    .reset_index(drop=True)  # 인덱스 초기화 (기존 인덱스 열 제거)
    )

    # Series의 이름을 'sentence'로 지정
    sentence.name = 'sentence'


    # 필터링된 s_id의 개수 출력
    num_s_ids = filtered_df['s_id'].nunique()
    print(f"Total number of filtered sentences: {num_s_ids}")



    # 상위 10개의 문장 번호와 문장 미리보기 출력
    print("\n========== Preview (Top 10 sentences)==========\n")
    preview_sentences = final_df.groupby("s_id").apply(combine_text).head(10).reset_index()  # 's_id' 복원
    preview_sentences = preview_sentences.rename(columns={0: 'sentence'})  # apply() 결과에서 'sentence'라는 열로 이름 변경

    for index, row in preview_sentences.iterrows():  # iterrows()로 DataFrame을 순회
        print(f"{row['s_id']}: {row['sentence']}")


        # 파일로 저장
    while True:
        file_name = input("\nEnter the file name to save (with extension .txt or .csv), or 'exit' to quit: ").strip()

        # "exit" 입력 시 종료
        if file_name.lower() == "exit":
            print("Exiting the program.")
            break

        # 파일 확장자에 맞춰 파일 저장
        try:
            sentence_df = sentence.to_frame()

            if file_name.endswith(".txt"):
                with open(file_name, "w", encoding="utf-8") as file:
                    for index, row in sentence_df.iterrows():
                        file.write(f"{index}, {row['sentence']}\n")
                print(f"\nResults saved to {file_name}")
                break


            elif file_name.endswith(".csv"):
                sentence.to_csv(file_name, index=False)
                print(f"\nResults saved to {file_name}")
                break


            else:
                print("Invalid file extension. Please use .txt or .csv.")

        except Exception as e:
            print(f"Error saving the file: {e}")

