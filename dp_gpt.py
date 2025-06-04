import stanza
import pandas as pd
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import sys
import warnings
import csv



##### file size #####

def size(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return len(lines)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return 0



##### dependency parsing #####

# def dp(filename, *args):
#     if len(args) == 0:
#         start = 1
#         end = 100000
#     elif len(args) == 1:
#         start = 1
#         end = args[0]
#     elif len(args) == 2:
#         start = args[0]
#         end = args[1]
#     else:
#         raise ValueError("The dp() function accepts a maximum of three arguments: filename, start, and end.")

#     with open(filename, 'r', encoding='utf-8') as f:
#         lines = f.read().splitlines()

#     if end is None:
#         selected_lines = lines[start - 1:]
#     else:
#         selected_lines = lines[start - 1 : end]

#     all_tokens = []

#     stanza.download('en', verbose=False)
#     nlp = stanza.Pipeline(lang='en',
#                           processors='tokenize,pos,lemma,depparse',
#                           use_gpu=True,
#                           verbose=False)

#     for idx, sentence in enumerate(tqdm(selected_lines, desc="Dependency Parsing"), start=start):
#         doc = nlp(sentence)
#         doc_dict = doc.to_dict()
#         for sentence_data in doc_dict:
#             for token in sentence_data:
#                 new_token = {"s_id": idx, **token}
#                 all_tokens.append(new_token)

#     total_sentences = len(selected_lines)
#     print(f"\n\nA total of {total_sentences} sentences have been processed.")
#     print(f"Processed sentences from {start} to {start + total_sentences - 1}.")

#     df = pd.DataFrame(all_tokens)
#     df = df.drop(columns=['feats', 'misc'], errors='ignore')
#     df = df.dropna(subset=['head'])
#     df['head'] = df['head'].astype(int)

#     return df


def dp(input_data, *args):
    stanza.download('en', verbose=False)
    nlp = stanza.Pipeline(lang='en',
                          processors='tokenize,pos,lemma,depparse',
                          use_gpu=True,
                          verbose=False)

    # (1) 단일 문장 처리
    if not input_data.endswith(".txt"):
        sentence = input_data
        doc = nlp(sentence)
        doc_dict = doc.to_dict()

        tokens = []
        for sentence_data in doc_dict:
            for token in sentence_data:
                tokens.append(token)

        df = pd.DataFrame(tokens)
        df = df.drop(columns=['feats', 'misc'], errors='ignore')
        df = df.dropna(subset=['head'])
        df['head'] = df['head'].astype(int)

        print("\n[Dependency Parsing Result for Input Sentence:]\n")
        # print(df)
        return df

    # (2) 파일 처리
    filename = input_data
    if len(args) == 0:
        start = 1
        end = 100000
    elif len(args) == 1:
        start = 1
        end = args[0]
    elif len(args) == 2:
        start = args[0]
        end = args[1]
    else:
        raise ValueError("The dp() function accepts a maximum of three arguments: filename, start, and end.")

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    if end is None:
        selected_lines = lines[start - 1:]
    else:
        selected_lines = lines[start - 1 : end]

    all_tokens = []
    for idx, sentence in enumerate(tqdm(selected_lines, desc="Dependency Parsing"), start=start):
        doc = nlp(sentence)
        doc_dict = doc.to_dict()
        for sentence_data in doc_dict:
            for token in sentence_data:
                new_token = {"s_id": idx, **token}
                all_tokens.append(new_token)

    total_sentences = len(selected_lines)
    print(f"\n\nA total of {total_sentences} sentences have been processed.")
    print(f"Processed sentences from {start} to {start + total_sentences - 1}.")

    df = pd.DataFrame(all_tokens)
    df = df.drop(columns=['feats', 'misc'], errors='ignore')
    df = df.dropna(subset=['head'])
    df['head'] = df['head'].astype(int)

    return df



##### preview #####

def preview(df, s_id):
    total_sentences = df["s_id"].nunique()
    min_sentence = df["s_id"].min()
    max_sentence = df["s_id"].max()

    print(f"Total number of sentences: {total_sentences}")
    print(f"Sentence range: sentence {min_sentence} to {max_sentence}\n\n")
    print(f"<Preview of sentence ID {s_id}>\n\n")

    filtered_df = df[df["s_id"] == s_id]
    print(filtered_df)



##### chat #####

warnings.filterwarnings("ignore", message=".*ChatOpenAI.*deprecated.*")
warnings.filterwarnings("ignore", message=".*BaseChatModel.__call__.*deprecated.*")

def condition_loop():
    inclusion_conditions = []
    exclusion_conditions = []

    while True:
        print("Inclusion conditions (press Enter to finish):")
        cond = input().strip()
        if cond == "":
            break
        if cond == "exit":
            return None, None
        if cond == "reset":
            print("\n(((Resetting conditions.)))\n")
            inclusion_conditions = []
            exclusion_conditions = []
            return condition_loop()
        inclusion_conditions.append(cond)


    while True:
        print("Exclusion conditions (press Enter to finish):")
        cond = input().strip()
        if cond == "":
            break
        if cond == "exit":
            return None, None
        if cond == "reset":
            print("\n(((Resetting conditions.)))\n")
            inclusion_conditions = []
            exclusion_conditions = []
            return condition_loop()
        exclusion_conditions.append(cond)

    return inclusion_conditions, exclusion_conditions


def condition_to_code(conditions, api_key):
    if not conditions: 
        return ""
    query = "\n".join([f"- {cond}" for cond in conditions])
    prompt = f"""
              I have a pandas DataFrame named "df" that contains dependency-parsed sentences.
              Each row represents a token with the following columns:
              - s_id: sentence index
              - id: token's position in the sentence
              - text: token text
              - lemma: token lemma
              - upos: universal POS tag
              - xpos: language-specific POS tag
              - deprel: dependency relation
              - start_char: start index of token in the sentence
              - end_char: end index of token in the sentence

              Convert the following natural language query exactly into a single line of plain Python code that serves as a condition to filter "df".
              Do not add any extra conditions or modify any names or identifiers in any way.
              If the query references a .txt file (e.g., "file.txt"), you must call the load_list function with that exact file name (i.e. load_list("file.txt")) and then use the isin() method to check membership against the returned list.
              Do not wrap any identifier in load_list() unless it explicitly contains a .txt extension.
              When query is like "[COLUMN1] (when [COLUMN2] is [VALUE2]) is the same as [COLUMN3] (when [COLUMN4] is [VALUE4])", the code should be like "df[df['[COLUMN2]'] == '[VALUE2]']['[COLUMN1]'].isin(df[df['[COLUMN4]'] == '[VALUE4]']['[COLUMN3]'])".
              Use the exact COLUMN and VALUE names as suggested in the query, especially between the square brackets "[ ]".
              Output should be only the plain, executable code without any markdown formatting, such as triple backticks or language tags.
              {query}
    """

    #llm = ChatOpenAI(temperature=0, openai_api_key=api_key, max_tokens=300)
    llm = ChatOpenAI(
    model_name="gpt-4o",  
    temperature=0,
    openai_api_key=api_key
    )
    messages = [
        SystemMessage(content="You are a helpful assistant that generates error-free Python code."),
        HumanMessage(content=prompt)
    ]
    response = llm(messages)
    code = response.content.strip()

    if code.startswith("```"):
        code = code.strip("`")
        lines = code.splitlines()
        if lines and lines[0].strip().lower() == "python":
            code = "\n".join(lines[1:]).strip()

    return code


def chat(api_key):
        print("\n[[[Start chatting]]]\n")
        inclusions, exclusions = condition_loop()

        if inclusions is None or exclusions is None:
            print("\n(((Exiting the chat.)))")
            return None, None

        print(("\n------------------------------"))
        print("\n[[[Requested conditions]]]")
        print("\nInclusion conditions:")
        for cond in inclusions:
            print(f"-{cond}")
        print("\nExclusion conditions:")
        for cond in exclusions:
            print(f"-{cond}")

        print(("\n------------------------------"))
        print("\n[[[Generated code]]]")
        inclusion_codes = []
        exclusion_codes = []
        print("\nCode for inclusion conditions:")
        for cond in inclusions:
            inclusion_code = condition_to_code(cond, api_key)
            inclusion_codes.append(inclusion_code)
            print(f"-{inclusion_code}")
        print("\nCode for exclusion conditions:")
        for cond in exclusions:
            exclusion_code = condition_to_code(cond, api_key)
            exclusion_codes.append(exclusion_code)
            print(f"-{exclusion_code}")

        while True:
            print(("\n------------------------------"))
            user_input = input("\nDo you want to proceed with these conditions? (yes/no): ").strip().lower()
            if user_input == "exit":
                print("\n(((Exiting the chat.)))")
                return None, None
            elif user_input == "reset":
                print("\n(((Resetting conditions.)))\n")
                return chat(api_key)
            elif user_input in ["yes", "y"]:
                print("\n(((Code saved.)))")
                return inclusion_codes, exclusion_codes
            elif user_input in ["no", "n"]:
                print("\n==============================")
                print("(((Restarting condition input.)))\n")
                return chat(api_key)
            else:
                print("Invalid input. Please enter 'yes', 'no', 'exit', or 'reset'.\n")



##### apply #####

def apply(df, inclusion, exclusion):
    def load_list(file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Error reading {file_name}: {e}")

    # 1. Inclusion 조건: s_id 그룹별 평가
    valid_s_ids = set()  # inclusion 조건을 모두 만족하는 s_id 저장용 집합
    grouped = df.groupby("s_id")  # s_id 기준으로 그룹화
    for s_id, group in grouped:
        satisfies_all = True  # 현재 그룹이 모든 inclusion 조건을 만족하는지 추적하는 변수
        for inclusion_code in inclusion:
            eval_globals = {"pd": pd}
            if ".txt" in inclusion_code:
                eval_globals["load_list"] = load_list
            try:
                # 현재 그룹(group)에 대해 inclusion 조건 평가
                result = eval(inclusion_code, {**eval_globals, "df": group})
            except Exception as e:
                raise ValueError(f"Error evaluating inclusion code '{inclusion_code}' for s_id {s_id}: {e}")
            # 평가 결과가 Boolean Series여야 함
            if isinstance(result, pd.Series) and result.dtype == bool:
                # 그룹 내 어느 행이라도 조건을 만족하면 True로 간주
                if not result.any():
                    satisfies_all = False  # 하나라도 만족하는 행이 없으면 해당 그룹은 포함 대상 아님
                    break
            else:
                raise ValueError("Inclusion code must return a Boolean Series.")
        if satisfies_all:
            valid_s_ids.add(s_id)  # 모든 inclusion 조건을 만족하면 해당 s_id를 저장

    # inclusion 조건을 만족하는 그룹의 데이터만 추출
    inclusion_filtered_df = df[df["s_id"].isin(valid_s_ids)]

    # 2. Exclusion 조건: s_id 그룹별 평가
    final_s_ids = set()  # exclusion 조건에서 걸러지지 않은 s_id 저장용 집합
    grouped_inclusion = inclusion_filtered_df.groupby("s_id")
    for s_id, group in grouped_inclusion:
        exclude_group = False  # 현재 그룹이 배제 조건에 걸리는지 여부
        for exclusion_code in exclusion:
            eval_globals = {"pd": pd}
            if ".txt" in exclusion_code:
                eval_globals["load_list"] = load_list
            try:
                # 현재 그룹(group)에 대해 exclusion 조건 평가
                result = eval(exclusion_code, {**eval_globals, "df": group})
            except Exception as e:
                raise ValueError(f"Error evaluating exclusion code '{exclusion_code}' for s_id {s_id}: {e}")
            if isinstance(result, pd.Series) and result.dtype == bool:
                # 그룹 내 어느 행이라도 조건을 만족하면 그룹 전체를 배제
                if result.any():
                    exclude_group = True
                    break
            else:
                raise ValueError("Exclusion code must return a Boolean Series.")
        if not exclude_group:
            final_s_ids.add(s_id)  # 배제 조건에 걸리지 않은 그룹의 s_id 저장

    # 최종적으로 inclusion과 exclusion 조건을 모두 반영한 결과 생성
    final_df = df[df["s_id"].isin(final_s_ids)]
    selected_sids = final_df['s_id'].unique()
    selected_tokens = df[df['s_id'].isin(selected_sids)]

    final_sentences = []
    for s_id, group in selected_tokens.groupby('s_id'):
        sorted_group = group.sort_values(by='id')
        sentence = ""
        for _, token in sorted_group.iterrows():
            if token['upos'].lower() == 'punct':
                sentence = sentence.rstrip() + token['text']
            else:
                if sentence:
                    sentence += " " + token['text']
                else:
                    sentence = token['text']
        final_sentences.append((s_id, sentence))

    total_count = len(set([s_id for s_id, _ in final_sentences]))
    print(f"\nTotal filtered sentences: {total_count}\n")
    if total_count > 10:
        print("(Displaying the first 10 results)")
    for s_id, sent in final_sentences[:10]:
        print(f"{s_id}, {sent}")

    while True:
        file_name = input("\nSave the result (with extension .txt or .csv): ").strip()
        if file_name.lower() == 'exit':
            print("(((Exiting without saving)))")
            return  
        elif file_name.lower().endswith('.csv'):
            with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["s_id", "sentence"]) 
                for s_id, sent in final_sentences:
                    writer.writerow([s_id, sent])
            print(f"Results saved to {file_name}")
            break 
        elif file_name.lower().endswith('.txt'):
            with open(file_name, 'w', encoding='utf-8') as f:
                for s_id, sent in final_sentences:
                    f.write(f"{s_id}, {sent}\n")
            print(f"Results saved to {file_name}")
            break 
        else:
            print("***Invalid file name: please enter a file name ending with '.csv' or '.txt', or type 'exit' to quit.")
