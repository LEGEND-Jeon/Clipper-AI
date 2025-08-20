from openai import OpenAI

#local ver
from whisper_local import whisper_local_transcribe
def whisperlocal(audio_path):
    result = whisper_local_transcribe(audio_path)
    return result

def gptapi(title, data):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are the video editor for a vlog YouTuber, responsible for cutting scenes."
            },
            {
                "role": "user",
                "content":f""":{data}

이것은 {title}에 대한 브이로그입니다.  
당신은 유튜브 브이로그 전문 편집자이며, 이 스크립트를 기반으로 영상 하이라이트만 남겨 자연스럽고 몰입감 있게 편집해야 합니다.

편집 목표:
- 최종 영상 길이가 원본의 최대 25% 이하, 가능하면 10% 이하가 되도록 분량을 과감히 줄이세요.
- 메인 스토리라인과 관련된 장면만 추출하고, 그 외 장면은 모두 삭제하세요.
- 10초~1분 이상의 연속된 대화 단위로 컷을 자르되, 10초 이하 클립은 포함하지 마세요. (30초 이상을 권장)

반드시 포함할 장면 (우선순위 순서):
1. 여행의 시작, 클라이맥스, 마무리 등 기승전결 구조
2. 감정이 담긴 장면 (웃음, 감탄, 놀라움, 감동 등) — 단 너무 짧게 자르면 흐름이 끊기니 주의
3. 질문과 답변이 자연스럽게 이어지는 대화
4. 장소나 상황에 대한 생생한 설명
5. 다음 계획이나 여정의 전환점이 되는 대사

제거할 장면:
- 메인 주제와 관련 없는 반복적 대화, 사적인 잡담, 지루한 설명
- 10초 이하의 짧은 반응 또는 문맥이 단절된 리액션
- 메인 주제와 관련 없는 내용

편집 기준:
- 편집본은 최대한 짧고 간결하며 주요 내용만 들어가는 것을 최우선으로 합니다. 최종 영상 길이가 원본의 최대 25% 이하, 가능하면 10% 이하가 되도록 분량을 과감히 줄이세요.
- 대화 단위로 묶어 자르며, 중간 생략 없이 자연스럽게 이어지도록 하세요.
- 스토리의 흐름이 매끄럽게 연결되도록, 전후 맥락이 끊기지 않게 하세요.

엔딩 조건:
- “이번 영상은 여기까지입니다”, “오늘 하루 끝!” 등 자연스러운 마무리 멘트가 등장하면 그 지점까지 포함하고 영상을 종료하세요.
- 따로 엔딩이 없는 경우, 마지막 주요 장면이나 감정의 여운이 남는 대사로 자연스럽게 끝내도록 하세요.

출력 형식: [['시작 시간', '끝 시간'], ['시작 시간', '끝 시간'], ...]  
※ 리스트 외 문장은 출력하지 마세요.
※ 시간 형식은 반드시 'MM:SS.mmm' 또는 'HH:MM:SS.mmm'으로 출력하고, 숫자(초)로 변환하지 마세요.
                        """
            }
        ],
        temperature=0.4,
        top_p=1
    )
    return  response

def gptapi_edit(title, data, before, req): #before: 이전 컷편집 리스트, req : nlp 요구 사항
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are the video editor for a vlog YouTuber, responsible for cutting scenes."
            },
            {
                "role": "user",
                "content":f"""
                        {data} 다음은 {title}에 대한 브이로그 대본입니다.
                        이건 이미 당신이 {before}로 편집한 적 있고 이번엔 수정을 할 겁니다.
                        사용자가 원하는 수정 사항은 {req} 입니다.
                        이전 편집본에서 다음 수정 사항을 반영하여 수정 편집본을 제작해주세요.
                        결과는 무조건 ['대화 시작 시간', '대화 종료 시간'] 형식의 리스트만 반환하세요.
                        """
            }
        ],
        temperature=0.4,
        top_p=1
    )
    return response

# GPT-api 호출
def gptapi_shorts(title, data,cv_time): #cv_time 
    time = 60 - cv_time
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are the vlog video editor for a shorts YouTuber, responsible for cutting scenes."
            },
            {
                "role": "user",
                "content":f"""
                        {data} 다음은 {title}에 대한 브이로그 대본입니다.
                        숏폼 유튜브 스타일로 빠르고 몰입감 있게 편집하세요. 총 영상 길이는 {time}초 이하여야 합니다. 강한 임팩트와 빠른 전개를 유지하며 재미와 감정을 강조하세요.
                        짧지만 핵심적인 장면과 대화만을 남기되, 다음 규칙들을 지키세요.
                        1. 핵심정보와 장소 정보는 유지하세요.
                        2. 대화와 장면이 이어지지만 짧고 강렬하게 편집해 지루하지 않게 하세요.
                        3. 빠른 템포를 유지하며 텐션이 높은 장면을 중심으로 구성하세요.
                        4. 감탄사와 리액션을 강조하세요.
                        5. 자연스러운 시작과 마무리를 맺으세요.
                        6. 흐름을 자연스럽게 하고 맥락이 끊기지 않도록 유의하세요.
                        7. 시작 → 주요 경험 → 클라이맥스(하이라이트) → 마무리까지 자연스럽게 이어지도록 하세요.
                        8. 결과는 무조건 ['대화 시작 시간', '대화 종료 시간'] 형식의 리스트만 반환하세요.
                        """
            }
        ],
        temperature=0.4,
        top_p=1
    )
    return response

def gptapi_shorts_edit(title, data, before, req,cv_time):
    time = 60 - cv_time
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are the vlog video editor for a shorts YouTuber, responsible for cutting scenes."
            },
            {
                 "role": "user",
                "content":f"""
                        {data} 다음은 {title}에 대한 브이로그 대본입니다.
                        이건 이미 당신이 {before}로 편집한 적 있고 이번엔 수정을 할 겁니다.
                        사용자가 원하는 수정 사항은 {req} 입니다.
                        이전 편집본에서 다음 수정 사항을 반영하여 수정 편집본을 제작해주세요.
                        총 영상 길이는 {time}초 이하여야 합니다.
                        결과는 무조건 ['대화 시작 시간', '대화 종료 시간'] 형식의 리스트만 반환하세요.
                        """
            }
        ],
        temperature=0.4,
        top_p=1
    )
    return response
