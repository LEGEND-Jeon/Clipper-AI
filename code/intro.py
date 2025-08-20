from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips

def mk_bgm(tmp_intro_path, result_path, title, ccolor, concept):
    top3_paths = tmp_intro_path[:4]
    video_clips = [VideoFileClip(path[0]) for path in top3_paths]
    final_clip = concatenate_videoclips(video_clips, method="compose")
    output_path = f'{result_path}/no_bgm_intro.mp4'
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    video_dir = output_path
    video = VideoFileClip(video_dir)  
    if concept =='moody' :
        bgm_select = "/home/ubuntu/reallyfinal/bgm/vlog_moody.mp3"
        ffont ="KCC-Ahnchangho"
    elif concept =='Energetic' :
        bgm_select = "/home/ubuntu/reallyfinal/bgm/vlog_Energetic.mp3"
        ffont = "SB Aggro"
    elif concept == 'fun' :
        bgm_select = "/home/ubuntu/reallyfinal/bgm/vlog_fun.mp3"
        ffont ="Ownglyph PDH"
    elif concept =='hip' :
        bgm_select = "/home/ubuntu/reallyfinal/bgm/vlog_hip.mp3"
        ffont = "Gmarket Sans"
    elif concept =='chill' :
        bgm_select = "/home/ubuntu/reallyfinal/bgm/vlog_chill.mp3"
        ffont="Ownglyph PDH"
    else :
        bgm_select = "/home/ubuntu/reallyfinal/bgm/vlog_peaceful.mp3"
        ffont = "SB Aggro"

    #text = TextClip(title, fontsize=200, color=ccolor, font=ffont,  method='caption')
    text = TextClip(title,
                fontsize=160,
                color=ccolor,
                font=ffont,
                #method='caption',
                method='label',
                size=(1500, 1500),
                align='center')
    #text = text.set_position("center").set_duration(video.duration)
    text = text.set_duration(video.duration)
    text = text.set_position(("center", "center"))
    
    final_video = CompositeVideoClip([video, text.set_start(0)])

    bgm = AudioFileClip(bgm_select).subclip(0, final_video.duration)
    final_video = final_video.set_audio(bgm)

    fin_path = f"{result_path}/intro_with_text_and_bgm.mp4" 
    # 최종 영상 출력
    final_video.write_videofile(fin_path, codec="libx264", audio_codec="aac")
    
    return fin_path
