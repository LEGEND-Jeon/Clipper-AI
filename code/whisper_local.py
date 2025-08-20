import whisper
from moviepy.editor import VideoFileClip
import subprocess
import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

def extract_audio(video_path):
    audio_path = video_path[:-4] + ".mp3"
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"오디오 추출 중 오류 발생: {e}")
        return None
    return audio_path

def whisper_local_transcribe(audio_path,
                             ns_thresh=0.6,
                             lp_thresh=-1.4,
                             cr_thresh=2.6):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cpu")

    model = whisper.load_model("large-v3", device=device)

    result = model.transcribe(
        audio_path,
        language='ko',
        verbose=True,
        temperature=0,
        condition_on_previous_text=False,
        initial_prompt="이 영상은 여행 브이로그입니다. 자연스러운 대화체입니다.",
        no_speech_threshold=0.7,
        logprob_threshold=-1.4,
        compression_ratio_threshold=2.6,
        beam_size=5
    )

    def format_time(seconds):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{minutes:02}:{sec:02}.{ms:03}"

    transcription_list = []
    buffer = ""
    start_time = None

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    log_filename = f"segments_threshold_ns{ns_thresh}_lp{lp_thresh}_cr{cr_thresh}.txt"
    final_filename = f"final_transcript_ns{ns_thresh}_lp{lp_thresh}_cr{cr_thresh}.txt"

    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write("start\tend\tno_speech_prob\tavg_logprob\tcompression_ratio\ttext\tkeep\n")

        for segment in result['segments']:
            no_speech_prob = segment.get('no_speech_prob', 0)
            avg_logprob = segment.get('avg_logprob', -10)
            compression_ratio = segment.get('compression_ratio', 0)
            start = format_time(segment['start'])
            end = format_time(segment['end'])
            text = segment['text'].strip()

            keep = (
                no_speech_prob < ns_thresh and
                avg_logprob > lp_thresh and
                compression_ratio < cr_thresh
            )

            log_file.write(f"{start}\t{end}\t{no_speech_prob:.2f}\t{avg_logprob:.2f}\t{compression_ratio:.2f}\t{text}\t{keep}\n")

            if keep:
                if buffer:
                    buffer += " " + text
                else:
                    buffer = text

                if len(buffer) > 20 or buffer.endswith((".", "!", "?")):
                    transcription_list.append([start, end, buffer.strip()])
                    buffer = ""

        if buffer:
            transcription_list.append([start, end, buffer.strip()])

    with open(final_filename, "w", encoding="utf-8") as out_file:
        for start, end, text in transcription_list:
            out_file.write(f"[{start} - {end}] {text}\n")
    return transcription_list

def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
         video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return format_time(int(float(result.stdout.strip())))

def get_excluded_times(video_path, exclude_list):
    total_start = "00:00"
    total_end = get_video_duration(video_path)
    result = []
    current_start = total_start

    for start, end, _ in exclude_list:
        if current_start < start:
            result.append([current_start, format_time(seconds_to_int(start) - 1)])
        current_start = format_time(seconds_to_int(end) + 1)

    if current_start < total_end:
        result.append([current_start, total_end])

    return result

def format_time(seconds):
    return f"{seconds // 60:02}:{seconds % 60:02}"


def seconds_to_int(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds
