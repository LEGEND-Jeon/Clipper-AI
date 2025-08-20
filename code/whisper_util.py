from moviepy.editor import VideoFileClip
import re
import subprocess
import torch
import sys
import numpy as np
import torch.nn.functional as F
from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx

import subprocess
import json

def get_video_duration(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout
        data = json.loads(output)
        duration = float(data["format"]["duration"])
        return duration
    except Exception as e:
        print("ffprobe 실패:", e)
        return 0.0

from whisper_local import whisper_local_transcribe

def whisperapi(audio_path):
    result = whisper_local_transcribe(audio_path)
    return result
def time_str_to_seconds(t):
    minutes_part, sec_part = t.split(':')
    minutes = int(minutes_part)
    if '.' in sec_part:
        sec_str, ms_str = sec_part.split('.')
        seconds = int(sec_str)
        ms = int((ms_str + "000")[:3])
    else:
        seconds = int(sec_part)
        ms = 0
    total_ms = (minutes * 60 + seconds) * 1000 + ms
    return total_ms

def seconds_to_time_str(ms):
    total_seconds, milli = divmod(int(ms), 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}.{milli:03}"

def format_time(seconds):
    return f"{seconds // 60:02}:{seconds % 60:02}"

def get_silence_intervals(video_path, transcription_list):
    total_duration = get_video_duration(video_path)
    result = []
    current_start = 0.0

    for start_str, end_str, _ in transcription_list:
        start = time_str_to_seconds(start_str)
        end = time_str_to_seconds(end_str)

        if current_start < start:
            result.append([seconds_to_time_str(current_start), seconds_to_time_str(start)])
        current_start = end

    if current_start < total_duration:
        result.append([seconds_to_time_str(current_start), seconds_to_time_str(total_duration)])
    return result

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end, score in intervals[1:]:
        last_start, last_end, last_score = merged[-1]
        if start <= last_end:
            merged[-1] = [last_start, max(last_end, end), max(last_score, score)]
        else:
            merged.append([start, end, score])
    return merged

def process_video_highlights_from_silence(video_path, checkpoint_path, clip_model_name_or_path):
    checkpoint_path = "/final/run_on_video/moment_detr_ckpt/model_best.ckpt"
    clip_model_name_or_path = "ViT-B/32"
    renamed_file = video_path[:-4] + ".mp3"
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(renamed_file)
    except Exception as e:
        print(f"오디오 추출 오류: {e}")
        return

    transcription_list = whisperapi(renamed_file)
    cleaned_data = [
        [start, end, re.sub(r'\s*\[\(.*?\)\]', '', text)] for start, end, text in transcription_list
    ]
    cleaned_data = [entry for entry in cleaned_data if entry[-1] != '']
    silence_intervals = get_silence_intervals(video_path, transcription_list)

    queries = [
        {"query": "A woman is standing to take picture."},
        {"query": "A woman smiling at the camera."},
        {"query": "A group of women smiling for a photo."},
        {"query": "A woman is posing to take picture."},
        {"query": "A man is standing to take picture."},
        {"query": "A man is posing to take picture."},
        {"query": "A man is traveling."},
        {"query": "A woman is traveling."},
        {"query": "A man is visiting landmarks, mountain, beach."},
        {"query": "A woman is visiting landmarks, mountain, beach."},
        {"query": "People are eating food."},
        {"query": "A cute animal is moving or looking at the camera."},
        {"query": "A fluffy animal appears in the scene and draws attention."},
        {"query": "A person is singing or playing an instrument on the street."},
        {"query": "A performer is entertaining people in a public place."}
    ]
    query_text_list = [q["query"] for q in queries]

    clip_len = 1
    highlight_threshold = 0.98
    num_predictions = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = ClipFeatureExtractor(
        framerate=1.0, size=224, centercrop=True,
        model_name_or_path=clip_model_name_or_path, device=device
    )
    model = build_inference_model(checkpoint_path).to(device)
    video_feats_dict = feature_extractor.encode_video_in_chunks(video_path, chunk_sec=10)
    full_video_feats = video_feats_dict["features"].to(device)
    full_video_feats = F.normalize(full_video_feats, dim=-1, eps=1e-5)
    video_duration = len(full_video_feats) * clip_len

    all_highlights, highlights_above_threshold = [], []

    for query_text in query_text_list:
        query_feats = feature_extractor.encode_text([query_text])
        query_feats, query_mask = pad_sequences_1d(query_feats, dtype=torch.float32, device=device)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        query_feats = query_feats.unsqueeze(0)
        query_mask = query_mask.view(1, -1)

        for start_time, end_time in silence_intervals:
            interval_duration = end_time - start_time
            start_frame = int(start_time // clip_len)
            end_frame = int(end_time // clip_len)
            if end_frame <= start_frame:
                continue

            segment_feats = full_video_feats[start_frame:end_frame]
            if segment_feats.shape[0] == 0:
                continue

            L_vid = segment_feats.shape[0]
            L_txt = query_feats.shape[1]
            video_mask = torch.ones(1, L_vid).to(device)
            query_mask = torch.ones(1, L_txt).to(device)
            mask = torch.cat([video_mask, query_mask], dim=1).bool()

            if mask.shape[1] != L_vid + L_txt:
                continue

            tef_st = torch.linspace(start_time, end_time, steps=L_vid, device=device) / video_duration
            tef_ed = tef_st + (interval_duration / video_duration)
            tef = torch.stack([tef_st, tef_ed], dim=1)
            segment_feats = torch.cat([segment_feats, tef], dim=1)
            segment_feats = segment_feats.unsqueeze(0)

            model_inputs = {
                "src_vid": segment_feats,
                "src_vid_mask": video_mask,
                "src_txt": query_feats,
                "src_txt_mask": query_mask
            }

            with torch.no_grad():
                outputs_list = [model(**model_inputs)["pred_logits"].cpu().numpy() for _ in range(num_predictions)]
                avg_scores = np.mean(outputs_list, axis=0)
                scores = torch.tensor(avg_scores).to(device)
                scores = F.softmax(scores, -1)[..., 0]

                pred_spans = model(**model_inputs)["pred_spans"].cpu()

                for spans, score in zip(pred_spans, scores.cpu()):
                    for (st, ed), s in zip(span_cxw_to_xx(spans).tolist(), score.tolist()):
                        abs_st = round(start_time + st * interval_duration, 2)
                        abs_ed = round(start_time + ed * interval_duration, 2)
                        all_highlights.append([abs_st, abs_ed, round(s, 4)])
                        if s >= highlight_threshold:
                            highlights_above_threshold.append([abs_st, abs_ed, round(s, 4)])

    final_moment = merge_intervals(highlights_above_threshold)
    top_10 = sorted(all_highlights, key=lambda x: x[2], reverse=True)[:10]
    title_video = [[max(0, round((st + ed) / 2 - 0.5, 2)), round((st + ed) / 2 + 0.5, 2)] for st, ed, _ in top_10]

    return all_highlights, final_moment, title_video

