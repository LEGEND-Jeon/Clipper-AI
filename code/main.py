from moviepy.editor import VideoFileClip
import shutil
from IPython.display import HTML
import os
import subprocess
from typing import List
import prompt
from send import send

import re
import ast

from whisper_util import whisper_local_transcribe, get_silence_intervals
from QVHighlight import extract_highlights_and_intro
from intro_test import mk_bgm
import time

import json

import os
import ast
import toki


def  extract_content_list(content):
    try:
        code_block = re.search(r"```(?:json|python)?\s*(.*?)```", content, re.DOTALL)
        if code_block:
            content_clean = code_block.group(1).strip()
        else:
            content_clean = content.strip()

        result = json.loads(content_clean)
        return result

    except Exception as e:
        return content

def delete(result_path):
    # 디렉터리인지 확인
    if not os.path.isdir(result_path):
        print(f"{result_path} 디렉토리가 아닙니다.")
        return

    for filename in os.listdir(result_path):
        # ✋ "intro"로 시작하는 파일은 건너뛰기
        if filename.lower().startswith("intro"):
            continue
        # 삭제 대상 확장자
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".txt")):
            file_path = os.path.join(result_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"❌ 삭제 실패: {file_path}, 이유: {e}")

def time_str_to_seconds(time_str):
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"잘못된 시간 형식: {time_str}")

def seconds_to_hhmmss(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}"

def cut_video(tmp_path, input_video_path, time_intervals, merge_file):
    output_prefix = "cut_video"
    os.makedirs(tmp_path, exist_ok=True)

    with open(merge_file, "w") as merge_f:
        download_links = []

        for idx, (start_time, end_time) in enumerate(time_intervals):
            output_file = f"{tmp_path}/{output_prefix}_{idx + 1}.mp4"

            ffmpeg_command = [
                'ffmpeg',
                '-i', input_video_path,
                '-ss', seconds_to_hhmmss(time_str_to_seconds(start_time)),
                '-to', seconds_to_hhmmss(time_str_to_seconds(end_time))
,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                output_file
            ]

            try:
                subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                merge_f.write(f"file '{output_file}'\n")
                download_links.append(create_download_link(output_file))  # Replace with your link function
            except subprocess.CalledProcessError as e:
                print(f"Error processing interval {start_time}-{end_time}: {e.stderr.decode()}")

        return download_links


def create_download_link(filename):
    return HTML(f'<a href="{filename}" download>{filename}</a>')


def save_to_srt(formatted_list: List[List[str]], srt_file_path: str):
    def time_to_srt_format(min_sec: str) -> str:
        minutes_str, sec_part = min_sec.split(":")

        if "." in sec_part:
            sec_str, ms_str = sec_part.split(".")
            seconds = int(sec_str)
            ms_str = (ms_str + "000")[:3]
            milliseconds = int(ms_str)
        else:
            seconds = int(sec_part)
            milliseconds = 0

        return f"00:{int(minutes_str):02}:{seconds:02},{milliseconds:03}"

    with open(srt_file_path, "w", encoding="utf-8") as srt_file:
        for idx, (start_min_sec, end_min_sec, text) in enumerate(formatted_list, start=1):
            start_time = time_to_srt_format(start_min_sec)
            end_time = time_to_srt_format(end_min_sec)

            srt_file.write(f"{idx}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text.strip()}\n\n")

def convert_srt_to_ass_with_style(srt_path: str, ass_path: str, font: str, font_size: int, primary_color: str,
                                  back_color: str, outline: int, border_style: int):
    ass_content = f"""[Script Info]
Title: Custom ASS Subtitle
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, OutlineColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{primary_color},{back_color},&HFF000000,-1,0,0,0,100,100,0,0,{border_style},{outline},0,2,10,10,5,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def format_time(srt_time):
        h, m, s = srt_time.split(":")
        s, ms = s.split(".")
        return f"{int(h)}:{m}:{s}.{ms[:3]}"

    with open(srt_path, "r", encoding="utf-8") as srt_file:
        with open(ass_path, "w", encoding="utf-8") as ass_file:
            ass_file.write(ass_content)
            lines = srt_file.readlines()
            idx = 0
            while idx < len(lines):
                line = lines[idx].strip()
                if line.isdigit():  # 자막 번호 (무시)
                    idx += 1
                    continue
                if "-->" in line:
                    start_time, end_time = line.split(" --> ")
                    start_time = format_time(start_time.replace(",", "."))
                    end_time = format_time(end_time.replace(",", "."))

                    idx += 1
                    if idx < len(lines):
                        text = lines[idx].strip()
                        ass_file.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,,0,,{text}\n")
                idx += 1
              
def add_subtitles_to_video(video_path: str, ass_file_path: str, output_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(ass_file_path):
        raise FileNotFoundError(f"ASS file not found: {ass_file_path}")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"ass={ass_file_path}",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while adding subtitles:\n{e.stderr}")

    return output_path

def process_subtitles_and_video(formatted_list: List[List[str]], input_video: str, srt_file: str, ass_file: str,
                                output_video: str):

    save_to_srt(formatted_list, srt_file)

    convert_srt_to_ass_with_style(
        srt_file, ass_file,
        font=" NanumBarunpen",
        font_size=16,
        primary_color="&H00FFFFFF",
        back_color="&H80000000",
        outline=0,
        border_style=4
    )

    return add_subtitles_to_video(input_video, ass_file, output_video)

def timestamp_to_seconds(ts: str) -> float:
    # ":"로 분리하고, 마지막 부분을 점(.)을 기준으로 초와 밀리초로 나눈다
    parts = ts.split(":")
    minutes = int(parts[0])
    seconds, millis = parts[1].split(".") if '.' in parts[1] else (parts[1], "000")
    
    # 초와 밀리초를 합쳐서 초 단위로 변환
    return minutes * 60 + int(seconds) + float(f"0.{millis}")

def seconds_to_timestamp(sec: float) -> str:
    # 초 단위로 된 시간을 분과 초, 밀리초로 나눈다
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    millis = int((sec - int(sec)) * 1000)
    
    return f"{minutes:02}:{seconds:02}.{millis:03}"

def last_srt(video):
    renamed_file = video[:-4] + ".mp3"
    try:
        video_clip = VideoFileClip(video)
        video_clip.audio.write_audiofile(renamed_file)
    except Exception as e:
        print(f"오디오 추출 중 오류 발생: {e}")

    transcription_list = whisper_local_transcribe(renamed_file)
    print('transcription_list', transcription_list)
    cleaned_data = [
        [start, end, re.sub(r'\s*\[\(.*?\)\]', '', text)] for start, end, text in transcription_list
    ]
    openai_script = [entry for entry in cleaned_data if entry[-1] != '']
    print('last_srt', openai_script)
    return openai_script
# SRT 파일을 뒤로 밀기
def shift_srt_by_intro(srt_path: str, output_path: str, intro_path: str):
    # 인트로 길이 계산
    try:
        intro_clip = VideoFileClip(intro_path)
        intro_duration = intro_clip.duration  # 초 단위 (예: 4.3)
        print(f"인트로 길이: {intro_duration:.3f}초")
    except Exception as e:
        print(f"인트로 길이 측정 오류: {e}")
        return

    # 기존 SRT 파일 열기
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 타임스탬프 패턴: MM:SS.mmm
    timestamp_pattern = re.compile(r"(\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}\.\d{3})")
    shifted_lines = []

    for line in lines:
        match = timestamp_pattern.search(line)
        if match:
            start, end = match.groups()
            new_start = seconds_to_timestamp(timestamp_to_seconds(start) + intro_duration)
            new_end = seconds_to_timestamp(timestamp_to_seconds(end) + intro_duration)
            shifted_lines.append(f"{new_start} --> {new_end}\n")
        else:
            shifted_lines.append(line)

    # 새 SRT 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(shifted_lines)
        
def extract_times(content):
    content = content.strip()
    if content.startswith("```python"):
        content = content[len("```python"):].strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    time_matches = re.findall(r"[\"'](\d{2,3}:\d{2}\.\d{3})[\"']", content)

    if len(time_matches) % 2 != 0:
        time_matches = time_matches[:-1]

    intervals = [
        [time_matches[i], time_matches[i + 1]]
        for i in range(0, len(time_matches), 2)
    ]

    def to_seconds(ts: str) -> float:
        minutes, seconds = ts.split(":")
        return int(minutes) * 60 + float(seconds)

    merged = []
    for start, end in intervals:
        if merged:
            prev_start, prev_end = merged[-1]
            gap = to_seconds(start) - to_seconds(prev_end)
            # Merge if gap in (0, 10]
            if gap <= 10:
                merged[-1][1] = end
                continue
        merged.append([start, end])

    return merged

def extract_number(filename):
    # 파일명에서 숫자만 추출
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')
def concat_videos_reencode(file_list_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True)
def reencode_files_in_place(total_merge):
    with open(total_merge, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines:
        if not line.strip().startswith("file "):
            continue

        # 원본 파일 경로 추출
        original_path = line.strip().split("file ")[1].strip().strip("'")

        # 임시 파일 경로
        tmp_path = original_path.replace(".mp4", "_reencoded.mp4")

        # 재인코딩 수행 (원본 → tmp)
        subprocess.run([
            "ffmpeg", "-y", "-i", original_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-r", "30",
            tmp_path
        ], check=True)

        # 재인코딩한 걸 원본 이름으로 덮어쓰기
        os.replace(tmp_path, original_path)  # 안전하게 rename

import tempfile

def reencode_to_uniform(input_path, output_path, target_size="1280x720", target_fps="30"):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale={target_size},fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-ac", "2", "-ar", "44100",
        output_path
    ], check=True)
def merge_videos_reencode_force(intro_path, outro_path, result_path):
    os.makedirs(result_path, exist_ok=True)

    # 스펙 맞춘 임시 파일 만들기
    fixed_intro = os.path.join(result_path, "fixed_intro.mp4")
    fixed_outro = os.path.join(result_path, "fixed_outro.mp4")

    reencode_to_uniform(intro_path, fixed_intro)
    reencode_to_uniform(outro_path, fixed_outro)

    output_path = os.path.join(result_path, "output_final.mp4")

    command = [
        "ffmpeg", "-y",
        "-i", fixed_intro,
        "-i", fixed_outro,
        "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    result.check_returncode()

    print(f"✅ 병합 완료: {output_path}")

def concat_videos(total_merge, result_path):
    output_path = result_path
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", total_merge,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path
    ], check=True)
import os
import shutil

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # 파일 또는 심볼릭 링크 삭제
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 하위 폴더 삭제
        except Exception as e:
            print(f"삭제 실패: {file_path}. 오류: {e}")


from pathlib import Path
from typing import List

FFMPEG = "ffmpeg"  # 시스템 PATH에 ffmpeg가 있다면 그대로 OK

def ensure_mp4(video_paths: List[str],
               overwrite: bool = False,
               reencode_mp4: bool = False,
               common_filters: str | None = None) -> List[Path]:
    """
    video_paths 내 영상들을 모두 .mp4 로 변환/검증하고 변환된 경로 리스트 반환.

    Parameters
    ----------
    video_paths : List[str]
        변환 대상 영상 경로들
    overwrite : bool
        True면 동일 이름 .mp4 가 있을 때 덮어씀(-y), False면 건너뜀
    reencode_mp4 : bool
        기존 mp4도 다시 인코딩해 규격을 맞출지 여부
    common_filters : str | None
        scale/fps 등 공통 비디오 필터(e.g. "scale=1280:720,fps=30")

    Returns
    -------
    List[Path]
        변환된(또는 유지된) mp4 파일 경로 리스트
    """
    converted_paths: List[Path] = []

    for src in map(Path, video_paths):
        if not src.exists():
            print(f"[WARN] {src} → 파일 없음, 건너뜀")
            continue

        # 목표 경로 (.mp4)
        dst = src.with_suffix(".mp4")

        # 변환이 필요한 경우?
        need_convert = (src.suffix.lower() != ".mp4") or reencode_mp4

        if not need_convert and dst.exists():
            # 이미 MP4이고 재인코딩 필요없는 경우
            converted_paths.append(dst)
            continue

        # ffmpeg 명령어 구성
        cmd = [
            FFMPEG,
            "-y" if overwrite else "-n",
            "-i", str(src),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-movflags", "+faststart"
        ]
        if common_filters:
            cmd += ["-vf", common_filters]
        cmd.append(str(dst))

        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            converted_paths.append(dst)
            print(f"[OK] {src.name} → {dst.name}")
        except subprocess.CalledProcessError as e:
            print(f"[ERR] {src.name} 변환 실패\n{e.stderr.decode(errors='ignore')[:300]} ...")

    return converted_paths

def ai(reqId, ttype, title, topic, ccolor, cconcept,email, subtitle):
    _start = time.perf_counter()
    intro_path = []
    tmp_intro_path =[]
    user_directory = f'/home/ubuntu/reallyfinal/uploads_files'
    user_input_path = f'{user_directory}'
    video_files = [f for f in os.listdir(user_input_path) if os.path.isfile(os.path.join(user_input_path, f))]
    video_files = sorted(video_files, key=extract_number)
    video_paths = [os.path.join(user_input_path, file) for file in video_files if
                   file.lower().endswith(('.mp4', '.mov'))]
    converted = ensure_mp4(
        video_paths,
        overwrite=True,             # 같은 이름 mp4가 있으면 덮어쓰기
        reencode_mp4=False,         # 기존 mp4는 그대로 두기
        common_filters="scale=1280:720,fps=30"   # 해상도·FPS 통일(옵션)
    )
    

    num = 0
    total_len = len(video_paths)
    for video in video_paths:
        os.makedirs(os.path.join(user_directory, 'result'), exist_ok=True)
        result_path = f'{user_directory}/result'
        tmp_video = video
        original_file = video

        renamed_file = original_file[:-4] + ".mp3"
        try:
            video_clip = VideoFileClip(tmp_video)
            video_clip.audio.write_audiofile(renamed_file)
        except Exception as e:
            print(f"오디오 추출 중 오류 발생: {e}")

        # nlp
        transcription_list = whisper_local_transcribe(renamed_file)
        cleaned_data = [
            [start, end, re.sub(r'\s*\[\(.*?\)\]', '', text)] for start, end, text in transcription_list
        ]
        openai_script = [entry for entry in cleaned_data if entry[-1] != '']
        silence_list_str = get_silence_intervals(video, cleaned_data)

        # gpt api 토큰 수 넘는 확인
        possicut = toki.toti(openai_script)
        if possicut == 0:
            if ttype == 1:
                gpt_response_result = extract_content_list(prompt.gptapi(title, openai_script))
                gpt_response_result = gpt_response_result.choices[0].message.content
                gpt_response_result = extract_times(gpt_response_result)
            if ttype == 2:
                gpt_response_result = extract_content_list(prompt.gptapi_edit(title, openai_script))
        else:
            gpt_response_result = []
            for i in range(len(possicut)):
                if i == 0:
                    if len(possicut) > 1:
                        script = openai_script[:possicut[1]]
                    else:
                        continue
                elif i == len(possicut) - 1:
                    script = openai_script[possicut[i] + 1:]
                else:
                    script = openai_script[possicut[i] + 1:possicut[i + 1]]

                # 컷편집 #if 문 1. 기본 2. 기본 수정 3. 숏츠 4. 숏츠 수정
                if ttype == 1:
                    tmp_response_result = prompt.gptapi(title,script)
                    content = tmp_response_result.choices[0].message.content

                    #parsed_result = merge_subtitles(gpt_response_list)
                    parsed_result = extract_times(content)
                    gpt_response_result += parsed_result
                if ttype == 2:
                    tmp_response_result = extract_content_list(prompt.gptapi_edit(title, script))  # ,before, req)
                    gpt_response_result += tmp_response_result
                if ttype == 3:
                    tmp_response_result = extract_content_list(prompt.gptapi_shorts(title, script, 5))  # cv_time)고치기!
                    gpt_response_result += tmp_response_result
                if ttype == 4:
                    tmp_response_result = extract_content_list(
                        prompt.gptapi_shorts_edit(title, script))  # before, req,cv_time):
                    gpt_response_result += tmp_response_result

        gpt_response_list = gpt_response_result

        def time_to_seconds(ts):
            m, s = ts.split(":")
            return int(m) * 60 + float(s)

        gpt_response_list.sort(key=lambda x: time_to_seconds(x[0]))

        # cv
        final_result, intro_segments = extract_highlights_and_intro(video, result_path, silence_list_str, num)
        tmp_intro_path += intro_segments
        if gpt_response_list:
            formatted_list = gpt_response_list

            download_links = cut_video(result_path, video, formatted_list, f"{result_path}/files_to_merge.txt")

            output_file_path = f"{user_directory}/z_{num}.mp4"
            #num += 1

            total_merge = f"{user_directory}/total_merge.txt"

            with open(total_merge, "a") as f:
                f.write(f"file '{output_file_path}'\n")

            input_file = f"{result_path}/files_to_merge.txt"

            final_output_file_path = output_file_path  # +'_final.mp4'
            fin_srt_file = f'{result_path}/fin_srt.srt'
            command = f"ffmpeg -f concat -safe 0 -i {input_file} -c copy {final_output_file_path}"
            subprocess.run(command, shell=True, check=True)
            delete(result_path)
        else:
            continue
        num += 1
        if num == total_len:
            print('len',num, total_len)
            final_merge = f"{user_directory}/total_merge.txt"
            
            total_merge = "/home/ubuntu/reallyfinal/uploads_files/total_merge.txt"
            merge_result_path = f"{user_directory}/result/output_{num}.mp4"

            reencode_files_in_place(total_merge)
            concat_videos(total_merge, merge_result_path)
            #concat_videos(total_merge, merge_result_path)
            # shell=False 버전 (권장)
            #subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", total_merge, "-c", "copy", f"{result_path}/final.mp4"], check=True)
            # command = f"ffmpeg -f concat -safe 0 -i {total_merge} -c copy {result_path}/final.mp4"
            # subprocess.run(command, shell=True, check=True)
            # os.system(
            #     f"ffmpeg -f concat -safe 0 -i {total_merge} -r 30 -c:v libx264 -c:a aac -movflags +faststart {result_path}/final.mp4")
            #num = 0
            mp4_file_path = merge_result_path #f'{result_path}/final.mp4'
            
            # 자막 생성
            if subtitle==1: 
                srt_script = last_srt(mp4_file_path)
                input_video = mp4_file_path
                srt_file = f"{user_directory}/result/script.srt"
                ass_file = f"{user_directory}/result/script.ass"
                output_video = f"{user_directory}/result/fin_output_{num}.mp4"

                video_srt = process_subtitles_and_video(srt_script, input_video, srt_file, ass_file, output_video)
            
            # intro
            intro_path = mk_bgm(tmp_intro_path, result_path, title, ccolor, cconcept)

            #intro_path = '/home/ubuntu/reallyfinal/uploads_files/result/intro_with_text_and_bgm.mp4'
            #output_video = f"{user_directory}/result/fin_output_{num}.mp4"
            
            videos_to_merge = f"{result_path}/videos_to_merge.txt"
            with open(videos_to_merge, "w") as f:
                f.write(f"file '{intro_path}'\n")
                f.write(f"file '{output_video}'\n")
                
            dst_folder = '/home/ubuntu/reallyfinal/uploads_files/result/fin'
            os.makedirs(dst_folder, exist_ok=True)
            output_path = f'{dst_folder}/output_final.mp4'
            
            merge_videos_reencode_force(
                intro_path,
                output_video,
                output_path
            )

            send(dst_folder, reqId, email)
            _end = time.perf_counter()
            elapsed_sec = _end - _start
            elapsed_min = elapsed_sec / 60
            print(f"Total execution time: {elapsed_min:.2f} minutes")
            return
            clear_folder('/home/ubuntu/reallyfinal/uploads_files')

            import request
            request.Request(reqId, final_result, gpt_response_list)

