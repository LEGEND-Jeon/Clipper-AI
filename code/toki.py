import openai
import tiktoken
import time

MAX_TOKENS = 20000  # gpt-4o 기준 최대 토큰 수

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def subtitles_token_count(sub_list, model="gpt-4o"):
    texts_only = [item[2] for item in sub_list]
    merged = "\n".join(texts_only)
    return count_tokens(merged, model)

def split_subtitles_by_token_limit(sub_list, model="gpt-4o", max_tokens=MAX_TOKENS):
    chunks = []
    current_chunk = []
    current_tokens = 0

    encoding = tiktoken.encoding_for_model(model)

    for sub in sub_list:
        sub_text = sub[2]
        sub_tokens = len(encoding.encode(sub_text))
        if current_tokens + sub_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = [sub]
            current_tokens = sub_tokens
        else:
            current_chunk.append(sub)
            current_tokens += sub_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def get_cut_indices_from_token_limit(sub_list, model="gpt-4o", max_tokens=MAX_TOKENS):
    cut_indices = [0]
    current_tokens = 0
    encoding = tiktoken.encoding_for_model(model)

    for i, sub in enumerate(sub_list):
        sub_text = sub[2]
        sub_tokens = len(encoding.encode(sub_text))
        if current_tokens + sub_tokens > max_tokens:
            cut_indices.append(i)
            current_tokens = sub_tokens
        else:
            current_tokens += sub_tokens
    return cut_indices

def split_by_cut_indices(subtitles, cut_indices):
    cut_indices = sorted(cut_indices)
    segments = []
    prev_index = 0

    for cut in cut_indices:
        segments.append(subtitles[prev_index:cut])
        prev_index = cut

    segments.append(subtitles[prev_index:])  # 마지막 조각
    return segments

def toti(subtitles):
    total_tokens = subtitles_token_count(subtitles)

    if total_tokens > MAX_TOKENS:
        cut_indices = get_cut_indices_from_token_limit(subtitles)

        subtitle_chunks = split_by_cut_indices(subtitles, cut_indices)
        # for i, chunk in enumerate(subtitle_chunks):
        #     print(f"Chunk {i+1} 자막 개수: {len(chunk)} / 토큰 수: {subtitles_token_count(chunk)}")

        return cut_indices  # 자막 자를 위치 반환
    else:
        return 0

