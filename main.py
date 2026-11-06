import json
import logging
import multiprocessing
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import wave
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ==============================================================================
# 【底層防護】修正打包 EXE 或關閉終端時的 Bad file descriptor 崩潰問題
# ==============================================================================
system_log_queue: "queue.Queue[str]" = queue.Queue()


class GUIWriter:
    def __init__(self) -> None:
        try:
            self.null_file = open(os.devnull, "w", encoding="utf-8")
        except Exception:
            self.null_file = None

    def write(self, data: str) -> None:
        if data and data.strip():
            system_log_queue.put(data.strip())

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        if self.null_file is not None:
            return self.null_file.fileno()
        return 1


if getattr(sys, "frozen", False):
    sys_writer = GUIWriter()
    sys.stdout = sys_writer
    sys.stderr = sys_writer


# ==============================================================================
# 第三方模組匯入
# ==============================================================================
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit

# ==============================================================================
# 全域路徑與設定
# ==============================================================================
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = BASE_DIR / "templates"
FFMPEG_DIR = BASE_DIR / "ffmpeg" / "bin"
YT_DLP_PATH = BASE_DIR / "yt-dlp.exe"

if FFMPEG_DIR.exists():
    os.environ["PATH"] += os.pathsep + str(FFMPEG_DIR)
os.environ["PATH"] += os.pathsep + str(BASE_DIR)


def find_ffmpeg() -> Optional[str]:
    """
    尋找可用的 FFmpeg 執行檔，並保證其目錄已加入當前進程的 PATH 環境變數。
    若發現 imageio_ffmpeg 附帶的執行檔但未命名為 ffmpeg.exe，自動建立 ffmpeg.exe 副本。
    """
    candidates: List[str] = []
    if FFMPEG_DIR.exists():
        exe = FFMPEG_DIR / "ffmpeg.exe"
        if exe.exists():
            candidates.append(str(exe))

    # 系統 PATH
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)

    # imageio_ffmpeg 備用
    try:
        import imageio_ffmpeg

        img_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if img_exe and Path(img_exe).exists():
            img_p = Path(img_exe)
            std_exe = img_p.parent / "ffmpeg.exe"
            if not std_exe.exists():
                try:
                    shutil.copyfile(img_p, std_exe)
                except Exception:
                    pass
            if std_exe.exists():
                candidates.append(str(std_exe))
            else:
                candidates.append(str(img_p))
    except Exception:
        pass

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            cand_dir = str(Path(candidate).parent.resolve())
            path_str = os.environ.get("PATH", "")
            if cand_dir not in path_str:
                os.environ["PATH"] = cand_dir + os.pathsep + path_str
            return candidate
    return None


# 初始化時立即尋找並注入 PATH
find_ffmpeg()

SONGS_DIR = BASE_DIR / "ktv_songs"
TEMP_BASE_DIR = BASE_DIR / "temp_processing"
SONGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = BASE_DIR / "voice_changer_config.json"
EFFECTS_PATH = BASE_DIR / "voice_changer_effects.json"
USER_EFFECT_MARK = "★ "
LIBRARY_DIR = BASE_DIR / "voice_library"

AUDIO_INPUT_TYPES = [
    ("音訊/視訊檔", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma *.mp4 *.mkv"),
    ("WAV 音檔", "*.wav"),
    ("MP3 音檔", "*.mp3"),
    ("MP4 影片", "*.mp4"),
    ("所有檔案", "*.*"),
]
AUDIO_OUTPUT_TYPES = [
    ("WAV 音檔", "*.wav"),
    ("MP3 音檔", "*.mp3"),
    ("M4A 音檔", "*.m4a"),
    ("FLAC 音檔", "*.flac"),
    ("所有檔案", "*.*"),
]

SAMPLE_RATE = 48000
DEFAULT_BLOCK_SIZE = 512
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 2

ANALYZE_MAX_SECONDS = 20.0
ANALYZE_FRAME = 2048
ANALYZE_HOP = 512
ANALYZE_FMIN = 60.0
ANALYZE_FMAX = 400.0
ANALYZE_SILENCE_RMS = 0.008

# ==============================================================================
# audio-separator 獨立子進程處理函式（記憶體回收隔離）
# ==============================================================================
def _run_audio_separator_process(
    input_path: str,
    output_dir: str,
    model_name: str = "UVR-MDX-NET-Inst_HQ_3.onnx",
    result_file: Optional[str] = None,
) -> None:
    """
    使用 audio-separator 在獨立進程中執行人聲/伴奏分離。
    結束後作業系統會自動強制釋放 PyTorch / ONNX 模型所佔用的記憶體。
    """
    import traceback

    # 關鍵：子進程啟動時強制確保 ffmpeg.exe 注入 PATH 最前面
    ff_exe = find_ffmpeg()

    try:
        if not shutil.which("ffmpeg") and not ff_exe:
            err = (
                "系統找不到 FFmpeg 執行檔 (ffmpeg.exe)！\n"
                "audio-separator 內部初始化必須呼叫 ffmpeg。\n"
                "解決方式（任選一種）：\n"
                "1. 在終端機執行: pip install imageio-ffmpeg (最簡單免安裝)\n"
                "2. 或下載 ffmpeg.exe，將其放入專案資料夾下的 ffmpeg\\bin\\ 中"
            )
            print(f"❌ {err}")
            if result_file:
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump({"success": False, "error": err}, f, ensure_ascii=False)
            sys.exit(1)

        try:
            from audio_separator.separator import Separator
        except ImportError as ie:
            err = (
                "找不到 audio-separator 模組，請先安裝:\n"
                "  pip install audio-separator[cpu]\n"
                "或如果有 NVIDIA 顯卡請安裝:\n"
                "  pip install audio-separator[gpu]\n"
                f"詳細錯誤: {ie}"
            )
            print(f"❌ {err}")
            if result_file:
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump({"success": False, "error": err}, f, ensure_ascii=False)
            sys.exit(1)

        separator = Separator(
            output_dir=output_dir,
            output_format="wav",
        )
        try:
            separator.load_model(model_filename=model_name)
        except Exception as me:
            print(f"載入指定模型 {model_name} 失敗 ({me})，嘗試載入預設模型...")
            try:
                separator.load_model()
            except Exception as me2:
                err = f"模型載入失敗，請確認網路連線可下載模型權重或檢查磁碟空間: {me2}"
                print(f"❌ {err}")
                if result_file:
                    with open(result_file, "w", encoding="utf-8") as f:
                        json.dump({"success": False, "error": err}, f, ensure_ascii=False)
                sys.exit(1)

        output_files = separator.separate(input_path)
        if result_file:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump({"success": True, "files": output_files}, f, ensure_ascii=False)
    except Exception as e:
        full_tb = traceback.format_exc()
        print(f"❌ audio-separator 執行發生未預期錯誤:\n{full_tb}")
        if result_file:
            try:
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump({"success": False, "error": f"{str(e)}\n{full_tb}"}, f, ensure_ascii=False)
            except Exception:
                pass
        sys.exit(1)


# ==============================================================================
# Flask + SocketIO KTV 伺服器
# ==============================================================================
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.config["SECRET_KEY"] = "ktv_secret"
socketio = SocketIO(app, cors_allowed_origins="*")


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


LOCAL_IP = get_local_ip()
PORT = 5000


def broadcast_log(msg: str) -> None:
    print(msg)
    try:
        socketio.emit("admin_log", {"msg": msg})
    except Exception:
        pass


@app.route("/player")
def page_player():
    return render_template("player.html")


@app.route("/remote")
def page_remote():
    return render_template("remote.html")


@app.route("/admin")
def page_admin():
    return render_template("admin.html")


@app.route("/combo")
def page_combo():
    return render_template("combo.html")


@app.route("/")
def page_index():
    return render_template("remote.html")


@app.route("/songs/<path:filename>")
def serve_song(filename: str):
    return send_from_directory(str(SONGS_DIR), filename)


@app.route("/api/list")
def get_song_list():
    songs = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith(".mp4")]
    return json.dumps(songs)


playlist_queue: List[str] = []


@socketio.on("add_to_queue")
def handle_add_queue(data):
    filename = data["filename"]
    playlist_queue.append(filename)
    emit("update_queue", playlist_queue, broadcast=True)
    if len(playlist_queue) == 1:
        emit("play_video", {"filename": filename, "title": filename}, broadcast=True)


@socketio.on("song_ended")
def handle_song_ended():
    if len(playlist_queue) > 0:
        playlist_queue.pop(0)
        emit("update_queue", playlist_queue, broadcast=True)
        if len(playlist_queue) > 0:
            next_song = playlist_queue[0]
            emit("play_video", {"filename": next_song, "title": next_song}, broadcast=True)
        else:
            emit("stop_video", broadcast=True)


@socketio.on("control")
def handle_control(action):
    if action == "cut":
        handle_song_ended()
    else:
        emit("command", action, broadcast=True)


@socketio.on("control_effect")
def handle_effect(data):
    emit("apply_effect", data, broadcast=True)


@socketio.on("change_track")
def handle_track(mode):
    emit("set_audio", mode, broadcast=True)


is_ktv_processing = False


@socketio.on("start_download")
def handle_start_download(data):
    global is_ktv_processing
    if is_ktv_processing:
        broadcast_log("⚠️ 系統正在處理其他歌曲，請稍候。")
        return

    url = data.get("url")
    title = data.get("title")

    def run_process():
        global is_ktv_processing
        is_ktv_processing = True
        socketio.emit("task_status", {"status": "busy"})
        processor = KTVProcessor(log_cb=broadcast_log)
        success = processor.process_song(url, title)
        if success:
            socketio.emit("refresh_list")
        is_ktv_processing = False
        socketio.emit("task_status", {"status": "idle"})

    broadcast_log("=== 開始新任務 ===")
    threading.Thread(target=run_process, daemon=True).start()


@socketio.on("update_ytdlp")
def handle_update_ytdlp():
    def run_update():
        socketio.emit("task_status", {"status": "busy"})
        broadcast_log("開始更新 yt-dlp 核心...")
        try:
            cmd = ["yt-dlp", "-U"]
            if YT_DLP_PATH.exists():
                cmd = [str(YT_DLP_PATH), "-U"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            broadcast_log(result.stdout)
            if result.stderr:
                broadcast_log(result.stderr)
            broadcast_log("✅ yt-dlp 更新程序結束。")
        except Exception as e:
            broadcast_log(f"❌ 更新失敗: {e}")
        finally:
            socketio.emit("task_status", {"status": "idle"})

    threading.Thread(target=run_update, daemon=True).start()


def run_server_thread():
    try:
        print("🚀 正在啟動 Flask KTV 伺服器...")
        from flask import cli

        cli.show_server_banner = lambda *args, **kwargs: None
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        socketio.run(app, host="0.0.0.0", port=PORT, debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        import traceback

        print(f"❌ 伺服器啟動失敗: {e}")
        print(traceback.format_exc())


# ==============================================================================
# 使用 MoviePy 進行影片與雙聲道 (L:原曲 R:伴奏) 合成
# ==============================================================================
def merge_dual_audio_video_moviepy(
    video_path: str,
    orig_audio_path: str,
    acc_audio_path: str,
    output_path: str,
    log_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    使用 moviepy 套件合成影片與 L/R 雙聲道（Left: 原曲，Right: 伴奏）
    若使用者尚未安裝 moviepy，回傳 False 供呼叫端降級使用 FFmpeg。
    """
    try:
        try:
            from moviepy.editor import VideoFileClip, ColorClip
            try:
                from moviepy.audio.AudioClip import AudioArrayClip
            except ImportError:
                from moviepy.editor import AudioArrayClip
        except ImportError:
            from moviepy import VideoFileClip
            try:
                from moviepy import ColorClip
            except ImportError:
                ColorClip = None
            try:
                from moviepy.audio.AudioClip import AudioArrayClip
            except ImportError:
                from moviepy.audio.io.AudioArrayClip import AudioArrayClip

        if log_cb:
            log_cb("🎵 正在使用 moviepy 載入音訊並構建立體聲 (L:原曲 R:伴奏)...")

        # 讀取左聲道 (原曲) 與右聲道 (伴奏)
        left_audio = load_audio_mono(Path(orig_audio_path), SAMPLE_RATE)
        right_audio = load_audio_mono(Path(acc_audio_path), SAMPLE_RATE)

        min_len = min(len(left_audio), len(right_audio))
        if min_len == 0:
            raise RuntimeError("原曲或伴奏音訊內容為空。")

        left = left_audio[:min_len]
        right = right_audio[:min_len]

        # 堆疊為立體聲 NumPy Array: (N, 2)，L: 原曲, R: 伴奏
        stereo_array = np.column_stack((left, right))
        stereo_clip = AudioArrayClip(stereo_array, fps=SAMPLE_RATE)
        total_duration = min_len / SAMPLE_RATE

        # 載入視訊軌
        video_clip = None
        has_video = False
        try:
            video_clip = VideoFileClip(video_path)
            if video_clip.duration and video_clip.size and video_clip.size[0] > 0:
                has_video = True
        except Exception as ve:
            if log_cb:
                log_cb(f"⚠️ 載入影片視訊軌失敗 ({ve})，嘗試生成預設背景...")

        # 若檔案本身為純音訊或無法解析視訊軌，則產生黑底視訊避免失敗
        if not has_video or video_clip is None:
            if ColorClip is not None:
                video_clip = ColorClip(size=(1280, 720), color=(10, 10, 10), duration=total_duration)
            else:
                raise RuntimeError("來源檔案不包含視訊軌，且無法建立背景畫面。")

        # 設定音訊至影片
        if hasattr(video_clip, "with_audio"):
            final_clip = video_clip.with_audio(stereo_clip)
        else:
            final_clip = video_clip.set_audio(stereo_clip)

        if hasattr(final_clip, "with_duration"):
            final_clip = final_clip.with_duration(total_duration)
        elif hasattr(final_clip, "set_duration"):
            final_clip = final_clip.set_duration(total_duration)

        if log_cb:
            log_cb("🎬 正在使用 moviepy 匯出最終 KTV 雙聲道影片...")

        temp_audio_file = os.path.join(os.path.dirname(output_path), "temp-audio.m4a")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=temp_audio_file,
            remove_temp=True,
            logger=None,
        )

        try:
            video_clip.close()
            stereo_clip.close()
            final_clip.close()
        except Exception:
            pass

        return True

    except ImportError:
        if log_cb:
            log_cb("💡 尚未安裝 moviepy 套件 (pip install moviepy)，將自動切換為 FFmpeg 核心合成。")
        return False


# ==============================================================================
# KTV 處理器 (下載、AI 分離與雙聲道合成)
# ==============================================================================
class KTVProcessor:
    def __init__(self, log_cb: Callable[[str], None]) -> None:
        self.log = log_cb

    def sanitize_filename(self, name: str) -> str:
        return "".join([c for c in name if c not in r'\/:*?"<>|'])

    def process_song(self, url: str, manual_title: str) -> bool:
        job_temp_dir: Optional[str] = None
        try:
            safe_title = self.sanitize_filename(manual_title)
            self.log(f"目標歌曲：{safe_title}")

            job_id = str(int(time.time()))
            job_temp_dir = str(TEMP_BASE_DIR / job_id)
            os.makedirs(job_temp_dir, exist_ok=True)

            temp_input = os.path.join(job_temp_dir, "input.mp4")
            temp_output = os.path.join(job_temp_dir, "output.mp4")

            self.log("步驟 1/4: 下載影片...")
            ff_exe = find_ffmpeg()
            ff_dir_for_ytdlp = str(Path(ff_exe).parent) if ff_exe else str(FFMPEG_DIR)
            cmd_dl = [
                "yt-dlp",
                "--ffmpeg-location",
                ff_dir_for_ytdlp,
                "--force-overwrites",
                "--no-playlist",
                "--merge-output-format",
                "mp4",
                "-f",
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
                "-o",
                temp_input,
                url,
            ]
            if YT_DLP_PATH.exists():
                cmd_dl[0] = str(YT_DLP_PATH)

            dl_res = subprocess.run(
                cmd_dl,
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if dl_res.returncode != 0:
                err_detail = dl_res.stderr.strip() if dl_res.stderr else f"Exit code {dl_res.returncode}"
                raise RuntimeError(f"yt-dlp 下載失敗: {err_detail}")

            self.log("步驟 2/4: AI 去人聲 (audio-separator)... (抽取音訊並分離中)")
            # 優先取得真正包含視訊的檔案（避免誤抓到未合併的 .m4a 純音訊）
            actual_video = None
            if os.path.exists(temp_input) and not temp_input.lower().endswith((".m4a", ".wav", ".mp3")):
                actual_video = temp_input
            else:
                # 掃描資料夾中包含視訊副檔名且檔案體積最大者
                video_candidates = [
                    os.path.join(job_temp_dir, f)
                    for f in os.listdir(job_temp_dir)
                    if f.lower().endswith((".mp4", ".mkv", ".webm", ".avi", ".mov"))
                    and not f.startswith("output")
                ]
                if video_candidates:
                    video_candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
                    actual_video = video_candidates[0]

            if not actual_video or not os.path.exists(actual_video):
                # 若完全找不到視訊副檔名，檢查是否有其他 input 開頭的非音訊檔
                for f in os.listdir(job_temp_dir):
                    if f.startswith("input") and not f.lower().endswith((".wav", ".json")):
                        actual_video = os.path.join(job_temp_dir, f)
                        break

            if not actual_video or not os.path.exists(actual_video):
                raise RuntimeError(f"找不到下載的影片檔案 (預期路徑: {temp_input})")

            temp_wav = os.path.join(job_temp_dir, "input_audio.wav")
            ffmpeg_cmd = find_ffmpeg() or "ffmpeg"
            cmd_extract = f'"{ffmpeg_cmd}" -y -i "{actual_video}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{temp_wav}"'
            extract_res = subprocess.run(
                cmd_extract,
                shell=True,
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if extract_res.returncode != 0 or not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                err_detail = extract_res.stderr.strip() if extract_res.stderr else f"Exit code {extract_res.returncode}"
                self.log(f"⚠️ FFmpeg 抽取音訊警告: {err_detail}，嘗試直接傳入原始檔案。")
                target_audio_for_sep = actual_video
            else:
                target_audio_for_sep = temp_wav

            result_json_path = os.path.join(job_temp_dir, "result.json")
            p = multiprocessing.Process(
                target=_run_audio_separator_process,
                args=(target_audio_for_sep, job_temp_dir, "UVR-MDX-NET-Inst_HQ_3.onnx", result_json_path),
            )
            p.start()
            p.join()

            if p.exitcode != 0:
                detail_err = "未知錯誤 (請檢查日誌或是否安裝 audio-separator)"
                if os.path.exists(result_json_path):
                    try:
                        with open(result_json_path, "r", encoding="utf-8") as f:
                            res = json.load(f)
                            if not res.get("success") and res.get("error"):
                                detail_err = res.get("error")
                    except Exception:
                        pass
                raise RuntimeError(f"audio-separator 分離失敗 (Exit code: {p.exitcode}):\n{detail_err}")

            # 解析 audio-separator 輸出的檔案
            output_files = []
            if os.path.exists(result_json_path):
                try:
                    with open(result_json_path, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    if res.get("success"):
                        output_files = res.get("files", [])
                    else:
                        raise RuntimeError(f"audio-separator 內部錯誤: {res.get('error')}")
                except Exception as e:
                    if not output_files and "audio-separator 內部錯誤" in str(e):
                        raise

            all_wavs = [os.path.join(job_temp_dir, f) for f in os.listdir(job_temp_dir) if f.lower().endswith(".wav")]
            candidates = output_files if output_files else all_wavs

            voc_path = None
            acc_path = None

            for f in candidates:
                full_f = os.path.join(job_temp_dir, f) if not os.path.isabs(f) else f
                if not os.path.exists(full_f):
                    continue
                name_lower = os.path.basename(full_f).lower()
                if "vocal" in name_lower:
                    voc_path = full_f
                elif "inst" in name_lower or "accompaniment" in name_lower:
                    acc_path = full_f

            if (not voc_path or not acc_path) and len(all_wavs) >= 2:
                for w in all_wavs:
                    if "vocal" in os.path.basename(w).lower():
                        voc_path = w
                    else:
                        acc_path = w
                if not voc_path:
                    voc_path = all_wavs[0]
                    acc_path = all_wavs[1]

            if not voc_path or not acc_path:
                raise RuntimeError("audio-separator 分離失敗，找不到輸出的人聲或伴奏音軌檔。")

            self.log("步驟 3/4: 合成 L/R 雙聲道 (L:原曲 R:伴奏)...")
            moviepy_success = False
            try:
                moviepy_success = merge_dual_audio_video_moviepy(
                    video_path=actual_video,
                    orig_audio_path=target_audio_for_sep,
                    acc_audio_path=acc_path,
                    output_path=temp_output,
                    log_cb=self.log,
                )
            except Exception as m_err:
                self.log(f"⚠️ moviepy 合成發生錯誤 ({m_err})，嘗試切換為 FFmpeg 備用合成...")

            if not moviepy_success:
                ffmpeg_cmd = find_ffmpeg() or "ffmpeg"
                # 使用 -map 0:v? 容錯（即使輸入檔案缺乏 video stream 也不會崩潰）
                cmd_ffmpeg = (
                    f'"{ffmpeg_cmd}" -y -i "{actual_video}" -i "{target_audio_for_sep}" -i "{acc_path}" '
                    '-filter_complex "[1:a]pan=mono|c0=0.5*FL+0.5*FR[L];[2:a]pan=mono|c0=0.5*FL+0.5*FR[R];[L][R]join=inputs=2:channel_layout=stereo[a]" '
                    f'-map 0:v? -map "[a]" -c:v copy -c:a aac "{temp_output}"'
                )

                merge_res = subprocess.run(
                    cmd_ffmpeg,
                    shell=True,
                    capture_output=True,
                    text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if merge_res.returncode != 0:
                    err_detail = merge_res.stderr.strip() if merge_res.stderr else f"Exit code {merge_res.returncode}"
                    raise RuntimeError(f"FFmpeg 合成雙聲道失敗: {err_detail}")

            self.log(f"步驟 4/4: 儲存為 {safe_title}.mp4")
            final = SONGS_DIR / f"{safe_title}.mp4"
            if final.exists():
                final = SONGS_DIR / f"{safe_title}_{job_id}.mp4"

            shutil.move(temp_output, str(final))
            self.log(f"✅ 製作完成！已存入歌庫: {final.name}")
            return True

        except subprocess.CalledProcessError as e:
            self.log(f"❌ 執行失敗 (Code {e.returncode})")
            return False
        except Exception as e:
            self.log(f"❌ 錯誤: {e}")
            return False
        finally:
            if job_temp_dir and os.path.exists(job_temp_dir):
                shutil.rmtree(job_temp_dir, ignore_errors=True)


# ==============================================================================
# 變聲核心與演算法
# ==============================================================================
@dataclass
class EffectSettings:
    pitch_ratio: float = 1.0
    robot_mix: float = 0.0
    echo_mix: float = 0.0
    echo_feedback: float = 0.2
    distortion_drive: float = 0.0
    compressor_amount: float = 0.0
    highpass_amount: float = 0.0
    lowpass_amount: float = 0.0
    presence_amount: float = 0.0
    air_amount: float = 0.0
    wet_mix: float = 1.0
    output_gain: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EffectSettings":
        fields = {
            key: float(data[key])  # type: ignore[arg-type]
            for key in cls.__dataclass_fields__
            if key in data and data[key] is not None
        }
        return cls(**fields)


PRESETS: Dict[str, EffectSettings] = {
    "自然": EffectSettings(),
    "機器人": EffectSettings(pitch_ratio=1.05, robot_mix=0.75, distortion_drive=0.12, wet_mix=1.0),
    "惡魔": EffectSettings(pitch_ratio=0.72, distortion_drive=0.3, lowpass_amount=0.45, wet_mix=1.0),
    "松鼠": EffectSettings(pitch_ratio=1.42, lowpass_amount=0.1, wet_mix=1.0),
    "廣播": EffectSettings(pitch_ratio=1.0, distortion_drive=0.08, lowpass_amount=0.22, wet_mix=0.9),
    "回音": EffectSettings(pitch_ratio=1.0, echo_mix=0.35, echo_feedback=0.35, wet_mix=0.95),
    "女聲-自然": EffectSettings(
        pitch_ratio=1.28,
        compressor_amount=0.28,
        highpass_amount=0.34,
        presence_amount=0.26,
        air_amount=0.12,
        lowpass_amount=0.04,
        wet_mix=1.0,
        output_gain=0.98,
    ),
    "女聲-明亮": EffectSettings(
        pitch_ratio=1.36,
        compressor_amount=0.35,
        highpass_amount=0.44,
        presence_amount=0.36,
        air_amount=0.18,
        lowpass_amount=0.0,
        wet_mix=1.0,
        output_gain=0.96,
    ),
    "女聲-柔和": EffectSettings(
        pitch_ratio=1.24,
        compressor_amount=0.42,
        highpass_amount=0.30,
        presence_amount=0.18,
        air_amount=0.08,
        lowpass_amount=0.08,
        wet_mix=0.92,
        output_gain=1.0,
    ),
    "女聲-可愛": EffectSettings(
        pitch_ratio=1.44,
        compressor_amount=0.30,
        highpass_amount=0.52,
        presence_amount=0.46,
        air_amount=0.20,
        lowpass_amount=0.0,
        wet_mix=1.0,
        output_gain=0.94,
    ),
    "極端測試": EffectSettings(
        pitch_ratio=1.75,
        compressor_amount=0.45,
        highpass_amount=0.70,
        presence_amount=0.65,
        air_amount=0.30,
        lowpass_amount=0.0,
        wet_mix=1.0,
        output_gain=0.88,
    ),
}


class SharedSettings:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings = EffectSettings()

    def update(self, **kwargs: float) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._settings, key, value)

    def replace(self, settings: EffectSettings) -> None:
        with self._lock:
            self._settings = EffectSettings(**settings.__dict__)

    def snapshot(self) -> EffectSettings:
        with self._lock:
            return EffectSettings(**self._settings.__dict__)


class DualWindowPitchShifter:
    def __init__(self, sample_rate: int, grain_ms: float = 70.0) -> None:
        self.sample_rate = sample_rate
        self.grain_size = max(128, int(sample_rate * grain_ms / 1000.0))
        self.buffer_size = self.grain_size * 8
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.phase = 0.0
        self.base_delay = self.grain_size * 2.5

    def _read_linear(self, pos: float) -> float:
        pos %= self.buffer_size
        idx0 = int(pos)
        idx1 = (idx0 + 1) % self.buffer_size
        frac = pos - idx0
        return float((1.0 - frac) * self.buffer[idx0] + frac * self.buffer[idx1])

    def process(self, samples: np.ndarray, pitch_ratio: float) -> np.ndarray:
        if abs(pitch_ratio - 1.0) < 1e-3:
            for sample in samples:
                self.buffer[self.write_pos] = sample
                self.write_pos = (self.write_pos + 1) % self.buffer_size
            return samples.copy()

        out = np.zeros_like(samples)
        phase_step = pitch_ratio - 1.0
        half_grain = self.grain_size / 2.0

        for i, sample in enumerate(samples):
            self.buffer[self.write_pos] = sample
            phase1 = self.phase % self.grain_size
            phase2 = (phase1 + half_grain) % self.grain_size
            norm1 = phase1 / self.grain_size
            norm2 = phase2 / self.grain_size
            win1 = 1.0 - abs(2.0 * norm1 - 1.0)
            win2 = 1.0 - abs(2.0 * norm2 - 1.0)

            read1 = self.write_pos - self.base_delay + phase1
            read2 = self.write_pos - self.base_delay + phase2
            mixed = self._read_linear(read1) * win1 + self._read_linear(read2) * win2
            out[i] = mixed / max(win1 + win2, 1e-5)

            self.phase = (self.phase + phase_step) % self.grain_size
            self.write_pos = (self.write_pos + 1) % self.buffer_size

        return out


class VoiceProcessor:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.pitch = DualWindowPitchShifter(sample_rate)
        self.echo_buffer = np.zeros(sample_rate * 2, dtype=np.float32)
        self.echo_pos = 0
        self.highpass_state = 0.0
        self.highpass_input_state = 0.0
        self.lowpass_state = 0.0
        self.presence_state = 0.0
        self.air_state = 0.0
        self.robot_phase = 0.0

    def _apply_robot(self, audio: np.ndarray, mix: float) -> np.ndarray:
        if mix <= 0.0:
            return audio
        carrier = 70.0
        phase_inc = 2.0 * np.pi * carrier / self.sample_rate
        phases = self.robot_phase + phase_inc * np.arange(audio.size, dtype=np.float32)
        modulated = audio * np.sign(np.sin(phases))
        self.robot_phase = float((phases[-1] + phase_inc) % (2.0 * np.pi))
        return audio * (1.0 - mix) + modulated * mix

    def _apply_distortion(self, audio: np.ndarray, drive: float) -> np.ndarray:
        if drive <= 0.0:
            return audio
        gain = 1.0 + drive * 25.0
        return np.tanh(audio * gain) / np.tanh(gain)

    def _apply_compression(self, audio: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0.0:
            return audio
        threshold = 0.72 - amount * 0.28
        ratio = 1.0 + amount * 4.0
        out = np.empty_like(audio)
        for i, sample in enumerate(audio):
            sign = 1.0 if sample >= 0.0 else -1.0
            level = abs(float(sample))
            if level > threshold:
                level = threshold + (level - threshold) / ratio
            out[i] = sign * level
        makeup = 1.0 + amount * 0.25
        return np.clip(out * makeup, -1.0, 1.0)

    def _apply_highpass(self, audio: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0.0:
            return audio
        cutoff = 110.0 + amount * 260.0
        rc = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / self.sample_rate
        alpha = rc / (rc + dt)
        out = np.empty_like(audio)
        y_prev = self.highpass_state
        x_prev = self.highpass_input_state
        for i, sample in enumerate(audio):
            y_prev = alpha * (y_prev + sample - x_prev)
            x_prev = sample
            out[i] = y_prev
        self.highpass_state = float(y_prev)
        self.highpass_input_state = float(x_prev)
        return out

    def _apply_lowpass(self, audio: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0.0:
            return audio
        cutoff = 8000.0 - amount * 6500.0
        alpha = min(0.45, 2.0 * np.pi * cutoff / self.sample_rate)
        out = np.empty_like(audio)
        state = self.lowpass_state
        for i, sample in enumerate(audio):
            state += alpha * (sample - state)
            out[i] = state
        self.lowpass_state = float(state)
        return out

    def _apply_presence(self, audio: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0.0:
            return audio
        alpha = 0.75
        out = np.empty_like(audio)
        state = self.presence_state
        for i, sample in enumerate(audio):
            state = alpha * state + (1.0 - alpha) * sample
            out[i] = sample + (sample - state) * amount * 1.8
        self.presence_state = float(state)
        return np.clip(out, -1.0, 1.0)

    def _apply_air(self, audio: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0.0:
            return audio
        alpha = 0.92
        out = np.empty_like(audio)
        state = self.air_state
        boost = amount * 1.2
        for i, sample in enumerate(audio):
            state = alpha * state + (1.0 - alpha) * sample
            airy = sample - state
            out[i] = sample + airy * boost
        self.air_state = float(state)
        return np.clip(out, -1.0, 1.0)

    def _apply_echo(self, audio: np.ndarray, mix: float, feedback: float) -> np.ndarray:
        if mix <= 0.0:
            return audio
        delay_samples = int(0.18 * self.sample_rate)
        out = np.empty_like(audio)
        for i, sample in enumerate(audio):
            read_pos = (self.echo_pos - delay_samples) % self.echo_buffer.size
            delayed = self.echo_buffer[read_pos]
            wet = sample + delayed * mix
            self.echo_buffer[self.echo_pos] = sample + delayed * feedback
            self.echo_pos = (self.echo_pos + 1) % self.echo_buffer.size
            out[i] = wet
        return out

    def process(self, samples: np.ndarray, settings: EffectSettings) -> np.ndarray:
        dry = samples.astype(np.float32, copy=True)
        wet = self.pitch.process(dry, settings.pitch_ratio)
        wet = self._apply_robot(wet, settings.robot_mix)
        wet = self._apply_distortion(wet, settings.distortion_drive)
        wet = self._apply_compression(wet, settings.compressor_amount)
        wet = self._apply_highpass(wet, settings.highpass_amount)
        wet = self._apply_presence(wet, settings.presence_amount)
        wet = self._apply_air(wet, settings.air_amount)
        wet = self._apply_lowpass(wet, settings.lowpass_amount)
        wet = self._apply_echo(wet, settings.echo_mix, settings.echo_feedback)
        mixed = dry * (1.0 - settings.wet_mix) + wet * settings.wet_mix
        mixed *= settings.output_gain
        return np.clip(mixed, -1.0, 1.0)




def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    out_size = max(1, int(round(audio.size * dst_rate / src_rate)))
    x_old = np.arange(audio.size, dtype=np.float64)
    x_new = np.linspace(0.0, audio.size - 1.0, out_size, dtype=np.float64)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _read_wav_stdlib(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        data = values.astype(np.float32) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"不支援的 WAV 位元深度: {width * 8} bit")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), rate


def _write_wav_stdlib(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def load_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 解碼失敗: {proc.stderr.decode('utf-8', 'ignore').strip()}")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    if path.suffix.lower() != ".wav":
        raise RuntimeError("找不到 ffmpeg，離線取樣目前只能讀取 WAV 檔。")

    data, rate = _read_wav_stdlib(path)
    return _resample(data, rate, sample_rate)


def save_audio_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    if path.suffix.lower() == ".wav":
        _write_wav_stdlib(path, audio, sample_rate)
        return

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg，目前只能輸出 WAV 檔。")

    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-i",
        "pipe:0",
        str(path),
    ]
    proc = subprocess.run(
        cmd,
        input=np.clip(audio, -1.0, 1.0).astype("<f4").tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 編碼失敗: {proc.stderr.decode('utf-8', 'ignore').strip()}")


def render_file(
    input_path: Path,
    output_path: Path,
    settings: EffectSettings,
    progress: Optional[Callable[[float], None]] = None,
) -> float:
    audio = load_audio_mono(input_path, SAMPLE_RATE)
    total = audio.size
    if total == 0:
        raise RuntimeError("輸入音檔沒有聲音資料。")

    processor = VoiceProcessor(SAMPLE_RATE)
    rendered = np.empty(total, dtype=np.float32)

    for start in range(0, total, DEFAULT_BLOCK_SIZE):
        chunk = audio[start : start + DEFAULT_BLOCK_SIZE]
        if chunk.size < DEFAULT_BLOCK_SIZE:
            chunk = np.pad(chunk, (0, DEFAULT_BLOCK_SIZE - chunk.size))

        processed = processor.process(chunk, settings)
        end = min(start + processed.size, total)
        rendered[start:end] = processed[: end - start]

        if progress is not None:
            progress(end / total)

    save_audio_mono(output_path, rendered, SAMPLE_RATE)
    return total / SAMPLE_RATE


# ==============================================================================
# 聲音庫與自訂效果儲存
# ==============================================================================
class VoiceLibrary:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.audio_dir = root_dir / "audio"
        self.index_path = root_dir / "index.json"
        self._entries: List[Dict[str, object]] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            self._entries = []
            try:
                self.audio_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                return
            if not self.index_path.exists():
                return
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                return
            raw = data.get("entries", []) if isinstance(data, dict) else []
            self._entries = [item for item in raw if isinstance(item, dict) and item.get("id")]
            self._entries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)

    def entries(self) -> List[Dict[str, object]]:
        with self._lock:
            return [dict(entry) for entry in self._entries]

    def find(self, entry_id: str) -> Optional[Dict[str, object]]:
        for entry in self.entries():
            if str(entry.get("id")) == entry_id:
                return entry
        return None

    def audio_path(self, entry: Dict[str, object]) -> Path:
        return self.root_dir / str(entry.get("file", ""))

    def _new_id(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def _save_locked(self) -> None:
        try:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            payload = {"entries": self._entries}
            self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"無法寫入聲音庫索引 {self.index_path}: {exc}") from exc

    def _register(
        self,
        entry_id: str,
        name: str,
        source: str,
        preset: str,
        duration: float,
        settings: EffectSettings,
        features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        entry: Dict[str, object] = {
            "id": entry_id,
            "name": name,
            "source": source,
            "preset": preset,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration": round(duration, 2),
            "file": f"audio/{entry_id}.wav",
            "settings": settings.to_dict(),
        }
        if features:
            entry["features"] = features
        with self._lock:
            self._entries.insert(0, entry)
            self._save_locked()
        return dict(entry)

    def add_wav(
        self,
        wav_path: Path,
        name: str,
        source: str,
        preset: str,
        duration: float,
        settings: EffectSettings,
    ) -> Dict[str, object]:
        entry_id = self._new_id()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wav_path, self.audio_dir / f"{entry_id}.wav")
        return self._register(entry_id, name, source, preset, duration, settings)

    def import_audio(self, path: Path, name: str, preset: str, settings: EffectSettings) -> Dict[str, object]:
        audio = load_audio_mono(path, SAMPLE_RATE)
        if audio.size == 0:
            raise RuntimeError("輸入音檔沒有聲音資料。")

        entry_id = self._new_id()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        _write_wav_stdlib(self.audio_dir / f"{entry_id}.wav", audio, SAMPLE_RATE)
        return self._register(entry_id, name, path.name, preset, audio.size / SAMPLE_RATE, settings)

    def render_into_library(
        self,
        source: Path,
        name: str,
        preset: str,
        settings: EffectSettings,
        features: Optional[Dict[str, float]] = None,
        progress: Optional[Callable[[float], None]] = None,
    ) -> Tuple[Dict[str, object], float]:
        entry_id = self._new_id()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        target = self.audio_dir / f"{entry_id}.wav"
        try:
            duration = render_file(source, target, settings, progress=progress)
        except Exception:
            if target.is_file():
                target.unlink()
            raise
        entry = self._register(entry_id, name, source.name, preset, duration, settings, features)
        return entry, duration

    def rename(self, entry_id: str, name: str) -> None:
        with self._lock:
            for entry in self._entries:
                if str(entry.get("id")) == entry_id:
                    entry["name"] = name
                    break
            self._save_locked()

    def delete(self, entry_id: str) -> None:
        with self._lock:
            target = next((entry for entry in self._entries if str(entry.get("id")) == entry_id), None)
            if target is None:
                return
            self._entries = [entry for entry in self._entries if str(entry.get("id")) != entry_id]
            audio = self.audio_path(target)
            self._save_locked()
        if audio.is_file():
            try:
                audio.unlink()
            except OSError:
                pass


def _strip_effect_mark(label: str) -> str:
    return label[len(USER_EFFECT_MARK) :] if label.startswith(USER_EFFECT_MARK) else label


class EffectStore:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self._effects: List[Dict[str, object]] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            self._effects = []
            if not self.index_path.exists():
                return
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                return
            raw = data.get("effects", []) if isinstance(data, dict) else []
            self._effects = [
                item
                for item in raw
                if isinstance(item, dict) and item.get("id") and item.get("name") and isinstance(item.get("settings"), dict)
            ]

    def entries(self) -> List[Dict[str, object]]:
        with self._lock:
            return [dict(item) for item in self._effects]

    def names(self) -> List[str]:
        return [str(item["name"]) for item in self.entries()]

    def find_by_name(self, name: str) -> Optional[Dict[str, object]]:
        for item in self.entries():
            if str(item["name"]) == name:
                return item
        return None

    def _save_locked(self) -> None:
        try:
            payload = {"effects": self._effects}
            self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"無法寫入自訂效果 {self.index_path}: {exc}") from exc

    def _reject_name_locked(self, name: str, ignore_id: str = "") -> None:
        if name in PRESETS:
            raise ValueError(f"「{name}」是內建預設名稱，請換一個名稱。")
        for item in self._effects:
            if str(item.get("id")) != ignore_id and str(item.get("name")) == name:
                raise ValueError(f"已經有同名的效果「{name}」。")

    def add(self, name: str, settings: EffectSettings, source: str = "") -> Dict[str, object]:
        effect: Dict[str, object] = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source": source,
            "settings": settings.to_dict(),
        }
        with self._lock:
            self._reject_name_locked(name)
            self._effects.insert(0, effect)
            self._save_locked()
        return dict(effect)

    def update_settings(self, effect_id: str, settings: EffectSettings) -> None:
        with self._lock:
            for item in self._effects:
                if str(item.get("id")) == effect_id:
                    item["settings"] = settings.to_dict()
                    break
            self._save_locked()

    def rename(self, effect_id: str, name: str) -> None:
        with self._lock:
            self._reject_name_locked(name, ignore_id=effect_id)
            for item in self._effects:
                if str(item.get("id")) == effect_id:
                    item["name"] = name
                    break
            self._save_locked()

    def delete(self, effect_id: str) -> None:
        with self._lock:
            self._effects = [item for item in self._effects if str(item.get("id")) != effect_id]
            self._save_locked()


class Soundboard:
    def __init__(self, library: VoiceLibrary, index_path: Path) -> None:
        self.library = library
        self.index_path = index_path
        self._items: List[Dict[str, object]] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            self._items = []
            if not self.index_path.exists():
                return
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                return
            raw = data.get("items", []) if isinstance(data, dict) else []
            self._items = [item for item in raw if isinstance(item, dict) and item.get("library_id")]

    def items(self) -> List[Dict[str, object]]:
        with self._lock:
            snapshot = [dict(item) for item in self._items]

        resolved: List[Dict[str, object]] = []
        for item in snapshot:
            entry = self.library.find(str(item.get("library_id", "")))
            if entry is None:
                continue
            resolved.append(
                {
                    "library_id": str(entry.get("id", "")),
                    "name": str(entry.get("name", "")),
                    "path": str(self.library.audio_path(entry)),
                    "loop": bool(item.get("loop", False)),
                }
            )
        return resolved

    def library_ids(self) -> List[str]:
        with self._lock:
            return [str(item.get("library_id", "")) for item in self._items]

    def index_of(self, library_id: str) -> Optional[int]:
        for index, library_key in enumerate(self.library_ids()):
            if library_key == library_id:
                return index
        return None

    def _save_locked(self) -> None:
        try:
            payload = {"items": self._items}
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"無法寫入音效選單 {self.index_path}: {exc}") from exc

    def add(self, library_id: str, loop: bool = False) -> bool:
        with self._lock:
            if any(str(item.get("library_id", "")) == library_id for item in self._items):
                return False
            self._items.append({"library_id": library_id, "loop": loop})
            self._save_locked()
        return True

    def remove_at(self, index: int) -> None:
        with self._lock:
            if 0 <= index < len(self._items):
                self._items.pop(index)
                self._save_locked()

    def set_loop(self, index: int, loop: bool) -> None:
        with self._lock:
            if 0 <= index < len(self._items):
                self._items[index]["loop"] = loop
                self._save_locked()

    def clear(self) -> None:
        with self._lock:
            self._items = []
            self._save_locked()

    def prune(self) -> bool:
        with self._lock:
            valid = {str(entry.get("id", "")) for entry in self.library.entries()}
            kept = [item for item in self._items if str(item.get("library_id", "")) in valid]
            if len(kept) == len(self._items):
                return False
            self._items = kept
            self._save_locked()
        return True


# ==============================================================================
# 音訊特徵分析與推薦
# ==============================================================================
def _guess_gender(f0: float) -> str:
    if f0 <= 0.0:
        return "未知"
    if f0 >= 165.0:
        return "偏女性"
    if f0 >= 130.0:
        return "中性"
    return "偏男性"


def _spectral_centroid(audio: np.ndarray) -> float:
    if audio.size < ANALYZE_FRAME:
        return 0.0

    window = np.hanning(ANALYZE_FRAME).astype(np.float32)
    accumulator = np.zeros(ANALYZE_FRAME // 2 + 1, dtype=np.float64)
    frames = 0
    for start in range(0, audio.size - ANALYZE_FRAME + 1, ANALYZE_FRAME):
        spectrum = np.abs(np.fft.rfft(audio[start : start + ANALYZE_FRAME] * window))
        accumulator += spectrum
        frames += 1

    if frames == 0:
        return 0.0

    mean_spectrum = accumulator / frames
    freqs = np.fft.rfftfreq(ANALYZE_FRAME, 1.0 / SAMPLE_RATE)
    total = float(mean_spectrum.sum())
    if total <= 1e-9:
        return 0.0
    return float((mean_spectrum * freqs).sum() / total)


@dataclass
class AudioFeatures:
    duration: float = 0.0
    analyzed_seconds: float = 0.0
    f0_median: float = 0.0
    f0_low: float = 0.0
    f0_high: float = 0.0
    f0_stability: float = 0.0
    brightness: float = 0.0
    rms: float = 0.0
    silence_ratio: float = 0.0
    gender_guess: str = "未知"

    def to_dict(self) -> Dict[str, float]:
        return {
            "duration": round(self.duration, 2),
            "f0_median": round(self.f0_median, 1),
            "f0_low": round(self.f0_low, 1),
            "f0_high": round(self.f0_high, 1),
            "f0_stability": round(self.f0_stability, 3),
            "brightness": round(self.brightness, 1),
            "rms": round(self.rms, 4),
            "silence_ratio": round(self.silence_ratio, 3),
        }

    def summary(self) -> str:
        if self.f0_median <= 0.0:
            return (
                f"長度 {self.duration:.1f}s｜音高：無法判定（可能是音樂、純音或噪音）"
                f"｜亮度 {self.brightness:.0f}Hz｜音量 {self.rms:.3f}｜靜音 {self.silence_ratio * 100:.0f}%"
            )
        return (
            f"長度 {self.duration:.1f}s｜基頻 {self.f0_median:.0f}Hz"
            f"（{self.f0_low:.0f}–{self.f0_high:.0f}Hz，穩定度 {self.f0_stability * 100:.0f}%，推測{self.gender_guess}）"
            f"｜亮度 {self.brightness:.0f}Hz｜音量 {self.rms:.3f}｜靜音 {self.silence_ratio * 100:.0f}%"
        )


def analyze_audio(path: Path, max_seconds: float = ANALYZE_MAX_SECONDS) -> AudioFeatures:
    audio = load_audio_mono(path, SAMPLE_RATE)
    features = AudioFeatures()
    if audio.size == 0:
        return features

    features.duration = audio.size / SAMPLE_RATE
    limit = min(audio.size, int(max_seconds * SAMPLE_RATE))
    segment = audio[:limit]
    features.analyzed_seconds = limit / SAMPLE_RATE
    features.rms = float(np.sqrt(np.mean(segment**2) + 1e-12))

    if segment.size < ANALYZE_FRAME:
        features.gender_guess = _guess_gender(0.0)
        return features

    window = np.hanning(ANALYZE_FRAME).astype(np.float32)
    lag_min = max(1, int(SAMPLE_RATE / ANALYZE_FMAX))
    lag_max = min(ANALYZE_FRAME - 1, int(SAMPLE_RATE / ANALYZE_FMIN))

    pitches: List[float] = []
    silent_frames = 0
    total_frames = 0
    for start in range(0, segment.size - ANALYZE_FRAME + 1, ANALYZE_HOP):
        total_frames += 1
        chunk = segment[start : start + ANALYZE_FRAME] * window
        if float(np.sqrt(np.mean(chunk**2) + 1e-12)) < ANALYZE_SILENCE_RMS:
            silent_frames += 1
            continue

        corr = np.correlate(chunk, chunk, mode="full")[ANALYZE_FRAME - 1 :]
        if corr[0] <= 0.0:
            continue
        search = corr[lag_min:lag_max]
        if search.size == 0:
            continue
        lag = lag_min + int(np.argmax(search))
        if corr[lag] < 0.3 * corr[0]:
            continue
        pitches.append(SAMPLE_RATE / lag)

    if pitches:
        values = np.array(pitches, dtype=np.float32)
        features.f0_median = float(np.median(values))
        features.f0_low = float(np.percentile(values, 10))
        features.f0_high = float(np.percentile(values, 90))
        spread = max(features.f0_high - features.f0_low, 1e-6)
        features.f0_stability = float(np.clip(1.0 - spread / max(features.f0_median, 1e-6), 0.0, 1.0))

    if total_frames:
        features.silence_ratio = silent_frames / total_frames

    features.brightness = _spectral_centroid(segment)
    features.gender_guess = _guess_gender(features.f0_median)
    return features


def recommend_settings(features: AudioFeatures) -> Tuple[str, EffectSettings, List[str]]:
    settings = EffectSettings()
    reasons: List[str] = []
    target_f0 = 205.0

    if features.f0_median > 0.0:
        settings.pitch_ratio = float(np.clip(target_f0 / features.f0_median, 0.70, 1.75))
        if features.f0_median < 150.0:
            reasons.append(f"基頻 {features.f0_median:.0f}Hz 偏低，音高放大到 {settings.pitch_ratio:.2f}x")
        elif features.f0_median > 240.0:
            reasons.append(f"基頻 {features.f0_median:.0f}Hz 偏高，音高縮小到 {settings.pitch_ratio:.2f}x")
        else:
            reasons.append(f"基頻 {features.f0_median:.0f}Hz 接近目標，音高微調為 {settings.pitch_ratio:.2f}x")
    else:
        reasons.append("無法判定音高，音高倍率維持 1.00")

    if features.brightness > 0.0:
        bright = float(np.clip((features.brightness - 1800.0) / 1200.0, -1.0, 1.0))
        settings.presence_amount = float(np.clip(0.14 + 0.16 * (1.0 - bright), 0.04, 0.42))
        settings.air_amount = float(np.clip(0.04 + 0.10 * (1.0 - bright), 0.0, 0.22))
        settings.lowpass_amount = float(np.clip(0.18 * bright, 0.0, 0.35))
        if bright < -0.3:
            reasons.append(f"亮度 {features.brightness:.0f}Hz 偏悶，加強亮度與空氣感")
        elif bright > 0.3:
            reasons.append(f"亮度 {features.brightness:.0f}Hz 偏亮，減少亮度補償並加上低通")
        else:
            reasons.append(f"亮度 {features.brightness:.0f}Hz 適中，亮度效果維持輕量")

    settings.highpass_amount = 0.24
    settings.compressor_amount = float(np.clip(0.20 + (0.15 - features.rms) * 0.8, 0.06, 0.45))
    if features.f0_median > 0.0 and features.f0_stability < 0.5:
        settings.compressor_amount = float(np.clip(settings.compressor_amount + 0.08, 0.0, 0.5))
        reasons.append(f"音高穩定度僅 {features.f0_stability * 100:.0f}%，提高壓縮量讓音色更整齊")

    if features.rms > 0.0:
        settings.output_gain = float(np.clip(0.20 / max(features.rms, 0.02), 0.7, 1.5))
        reasons.append(f"原始音量 {features.rms:.3f}，輸出增益調整為 {settings.output_gain:.2f}")

    if features.f0_median <= 0.0 and features.rms > 0.02:
        settings.robot_mix = 0.35
        reasons.append("沒有可辨識的人聲基頻，改用機器人效果增加變化")

    label = f"自動-{features.gender_guess}" if features.f0_median > 0.0 else "自動-未判定"
    return label, settings, reasons


def play_sound_file(path: Path, device: Optional[int], loop: bool = False, volume: float = 1.0) -> None:
    data, rate = _read_wav_stdlib(path)
    if data.size == 0:
        raise RuntimeError("音效檔沒有聲音資料。")
    if volume != 1.0:
        data = np.clip(data * volume, -1.0, 1.0).astype(np.float32)
    sd.play(data, rate, device=device, loop=loop)


# ==============================================================================
# 即時音訊引擎（含 ASIO 協議支援）
# ==============================================================================
class AudioEngine:
    def __init__(self, shared_settings: SharedSettings, log_queue: "queue.Queue[str]") -> None:
        self.shared_settings = shared_settings
        self.log_queue = log_queue
        self.stream: Optional[sd.Stream] = None
        self.processor = VoiceProcessor(SAMPLE_RATE)
        self.running = False
        self.is_asio = False

    def start(
        self,
        input_device: int,
        output_device: int,
        is_asio: bool = False,
        input_channel_idx: int = 0,
        output_channel_idx: int = 0,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        if self.running:
            return

        self.processor = VoiceProcessor(SAMPLE_RATE)
        self.is_asio = is_asio

        # 取得裝置實際支援的最大頻道數
        in_dev_info = sd.query_devices(input_device)
        out_dev_info = sd.query_devices(output_device)
        max_in_ch = int(in_dev_info["max_input_channels"])
        max_out_ch = int(out_dev_info["max_output_channels"])

        in_channels = max(1, min(max_in_ch, 2 if not is_asio else max_in_ch))
        out_channels = max(1, min(max_out_ch, 2 if not is_asio else max_out_ch))

        def callback(indata, outdata, frames, time_info, status) -> None:
            if status:
                self.log_queue.put(f"音訊狀態: {status}")

            # 根據聲道映射取得單聲道麥克風訊號
            ch_idx = input_channel_idx if input_channel_idx < indata.shape[1] else 0
            mono = indata[:, ch_idx].copy()
            settings = self.shared_settings.snapshot()
            processed = self.processor.process(mono, settings)

            # 輸出聲道指派
            outdata.fill(0)
            if is_asio:
                # ASIO 多聲道下輸出至指定的通道與相鄰立體聲通道
                target_left = output_channel_idx if output_channel_idx < outdata.shape[1] else 0
                target_right = target_left + 1 if (target_left + 1) < outdata.shape[1] else target_left
                outdata[:, target_left] = processed
                outdata[:, target_right] = processed
            else:
                for channel in range(outdata.shape[1]):
                    outdata[:, channel] = processed

        # ASIO 或專業驅動支援 AsioSettings
        extra_in = None
        extra_out = None
        if is_asio and hasattr(sd, "AsioSettings"):
            try:
                # 選擇專屬 ASIO 聲道 (1-based index)
                extra_in = sd.AsioSettings(channel_selectors=[input_channel_idx])
                in_channels = 1
            except Exception:
                pass

        stream_kwargs = {
            "samplerate": SAMPLE_RATE,
            "blocksize": block_size,
            "device": (input_device, output_device),
            "dtype": "float32",
            "latency": "low",
            "channels": (in_channels, out_channels),
            "callback": callback,
        }
        if extra_in is not None:
            stream_kwargs["extra_settings"] = (extra_in, extra_out)

        self.stream = sd.Stream(**stream_kwargs)
        self.stream.start()
        self.running = True
        asio_tag = " (ASIO 協議)" if is_asio else ""
        self.log_queue.put(f"變聲器已啟動{asio_tag}。輸入 #{input_device}，輸出 #{output_device}，緩衝區 {block_size}")

    def stop(self) -> None:
        if not self.running:
            return
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False
        self.log_queue.put("變聲器已停止。")


# ==============================================================================
# 整合型桌面 GUI (雙核心分頁：變聲器 + KTV 控制中心)
# ==============================================================================
class UnifiedKTVVoiceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("KTV & 即時變聲多功能整合系統 (含 ASIO 支援)")
        width = min(1100, self.root.winfo_screenwidth() - 60)
        height = min(1000, self.root.winfo_screenheight() - 60)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(800, 600)

        self.shared_settings = SharedSettings()
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.render_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.render_thread: Optional[threading.Thread] = None
        self.engine = AudioEngine(self.shared_settings, self.log_queue)
        self.library = VoiceLibrary(LIBRARY_DIR)

        self.devices: List[dict] = []
        self.hostapis: List[dict] = []
        self.input_map: Dict[str, int] = {}
        self.output_map: Dict[str, int] = {}
        self.device_hostapi_map: Dict[int, str] = {}
        self.device_channel_counts: Dict[int, int] = {}

        # 介面變數
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.input_ch_var = tk.StringVar(value="第 1 聲道 (預設)")
        self.output_ch_var = tk.StringVar(value="第 1-2 聲道 (立體聲)")
        self.buffer_size_var = tk.IntVar(value=512)
        self.preset_var = tk.StringVar(value="自然")
        self.status_var = tk.StringVar(value="尚未啟動")

        self.pitch_var = tk.DoubleVar(value=1.0)
        self.robot_var = tk.DoubleVar(value=0.0)
        self.echo_var = tk.DoubleVar(value=0.0)
        self.drive_var = tk.DoubleVar(value=0.0)
        self.compressor_var = tk.DoubleVar(value=0.0)
        self.highpass_var = tk.DoubleVar(value=0.0)
        self.lowpass_var = tk.DoubleVar(value=0.0)
        self.presence_var = tk.DoubleVar(value=0.0)
        self.air_var = tk.DoubleVar(value=0.0)
        self.wet_var = tk.DoubleVar(value=1.0)
        self.gain_var = tk.DoubleVar(value=1.0)
        self.input_file_var = tk.StringVar(value="尚未選擇音檔")
        self.render_status_var = tk.StringVar(value="")
        self.auto_library_var = tk.BooleanVar(value=True)
        self.soundboard_status_var = tk.StringVar(value="尚未加入任何音效")
        self.sound_loop_var = tk.BooleanVar(value=False)
        self.sound_volume_var = tk.DoubleVar(value=1.0)
        self.sound_volume_text_var = tk.StringVar(value="1.00")
        self.features_var = tk.StringVar(value="尚未辨識：選好音檔後按「辨識音檔特徵」。")
        self.auto_soundboard_var = tk.BooleanVar(value=True)
        self.last_features: Optional[AudioFeatures] = None

        self.soundboard = Soundboard(self.library, LIBRARY_DIR / "soundboard.json")
        self.effects = EffectStore(EFFECTS_PATH)

        self._job_handlers: Dict[str, Callable[[object], None]] = {
            "progress": self._on_render_progress,
            "done": self._on_render_finished,
            "features": self._on_features_ready,
            "auto_done": self._on_auto_finished,
            "import_done": self._on_import_finished,
        }

        self._build_ui()
        self.library.load()
        self.soundboard.load()
        self.effects.load()
        self._refresh_effect_choices()
        self._load_config()
        self.refresh_devices()
        self._on_preset_selected()
        self.refresh_library()

        # 啟動背景佇列輪詢
        self._poll_logs()
        self._poll_render_status()
        self._poll_system_logs()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        tab_voice = ttk.Frame(notebook)
        tab_ktv = ttk.Frame(notebook)

        notebook.add(tab_voice, text="🎙️ 即時變聲 & 音效庫 (含 ASIO)")
        notebook.add(tab_ktv, text="🎤 KTV 雲端點歌 & AI 伴奏中心")

        self._build_voice_tab(tab_voice)
        self._build_ktv_tab(tab_ktv)

    def _build_voice_tab(self, parent: ttk.Frame) -> None:
        page_canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        page_scroll = ttk.Scrollbar(parent, orient="vertical", command=page_canvas.yview)
        page_inner = ttk.Frame(page_canvas, padding=14)

        page_inner.bind("<Configure>", lambda _event: page_canvas.configure(scrollregion=page_canvas.bbox("all")))
        page_window = page_canvas.create_window((0, 0), window=page_inner, anchor="nw")
        page_canvas.configure(yscrollcommand=page_scroll.set)
        page_canvas.bind("<Configure>", lambda event: page_canvas.itemconfigure(page_window, width=event.width))

        page_canvas.pack(side="left", fill="both", expand=True)
        page_scroll.pack(side="right", fill="y")
        self._bind_wheel_scroll(page_canvas, page_inner)

        # 標題
        title = ttk.Label(page_inner, text="即時變聲與音效中心", font=("Microsoft JhengHei UI", 16, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(
            page_inner,
            text="支援 ASIO 專業錄音介面低延遲協議、虛擬音訊線 (VB-Audio / VAC) 與 Windows WASAPI。",
        )
        subtitle.pack(anchor="w", pady=(2, 10))

        # 1. 裝置選擇與 ASIO 設定
        dev_frame = ttk.LabelFrame(page_inner, text="音訊裝置與驅動協議設定", padding=12)
        dev_frame.pack(fill="x")

        ttk.Label(dev_frame, text="輸入麥克風 / 錄音裝置:").grid(row=0, column=0, sticky="w")
        self.input_box = ttk.Combobox(dev_frame, textvariable=self.input_var, state="readonly", width=62)
        self.input_box.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(3, 8))
        self.input_box.bind("<<ComboboxSelected>>", lambda _e: self._on_device_selection_changed())

        ttk.Label(dev_frame, text="輸出端 / 監聽或虛擬線:").grid(row=2, column=0, sticky="w")
        self.output_box = ttk.Combobox(dev_frame, textvariable=self.output_var, state="readonly", width=62)
        self.output_box.grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(3, 8))
        self.output_box.bind("<<ComboboxSelected>>", lambda _e: self._on_device_selection_changed())

        # ASIO 專屬聲道映射與 Buffer 控制區
        self.asio_options_frame = ttk.Frame(dev_frame)
        self.asio_options_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        ttk.Label(self.asio_options_frame, text="輸入通道:").pack(side="left")
        self.in_ch_box = ttk.Combobox(
            self.asio_options_frame,
            textvariable=self.input_ch_var,
            state="readonly",
            width=16,
            values=["第 1 聲道 (預設)", "第 2 聲道", "第 3 聲道", "第 4 聲道"],
        )
        self.in_ch_box.pack(side="left", padx=(4, 16))

        ttk.Label(self.asio_options_frame, text="輸出通道:").pack(side="left")
        self.out_ch_box = ttk.Combobox(
            self.asio_options_frame,
            textvariable=self.output_ch_var,
            state="readonly",
            width=18,
            values=["第 1-2 聲道 (立體聲)", "第 3-4 聲道", "第 5-6 聲道"],
        )
        self.out_ch_box.pack(side="left", padx=(4, 16))

        ttk.Label(self.asio_options_frame, text="緩衝區大小:").pack(side="left")
        self.buf_box = ttk.Combobox(
            self.asio_options_frame,
            textvariable=self.buffer_size_var,
            state="readonly",
            width=8,
            values=[128, 256, 512, 1024],
        )
        self.buf_box.pack(side="left", padx=(4, 0))

        # 按鈕區
        btn_frame = ttk.Frame(dev_frame)
        btn_frame.grid(row=1, column=1, rowspan=3, sticky="ns")
        ttk.Button(btn_frame, text="重新整理", command=self.refresh_devices).pack(fill="x", pady=(0, 6))
        ttk.Button(btn_frame, text="啟動變聲", command=self.start).pack(fill="x", pady=(0, 6))
        ttk.Button(btn_frame, text="停止", command=self.stop).pack(fill="x")
        dev_frame.columnconfigure(0, weight=1)

        # 2. 預設與自訂效果
        preset_frame = ttk.LabelFrame(page_inner, text="預設音色與自訂儲存", padding=12)
        preset_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(preset_frame, text="套用預設:").grid(row=0, column=0, sticky="w")
        self.preset_box = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            width=26,
        )
        self.preset_box.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.preset_box.bind("<<ComboboxSelected>>", lambda _e: self._on_preset_selected())

        effect_btn_frame = ttk.Frame(preset_frame)
        effect_btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        effect_buttons = [
            ("儲存效果", self.save_effect),
            ("更新效果", self.update_effect),
            ("重新命名", self.rename_effect),
            ("刪除效果", self.delete_effect),
        ]
        for label, command in effect_buttons:
            ttk.Button(effect_btn_frame, text=label, command=command).pack(side="left", padx=(0, 8))

        # 3. 音檔取樣與【KTV 歌庫連動】
        sample_frame = ttk.LabelFrame(page_inner, text="音檔取樣（支援從 KTV 歌庫直接連動）", padding=12)
        sample_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(sample_frame, text="來源檔案:").grid(row=0, column=0, sticky="w")
        ttk.Entry(sample_frame, textvariable=self.input_file_var, state="readonly").grid(
            row=1, column=0, sticky="ew", padx=(0, 8), pady=(2, 8)
        )

        sample_btn_sub = ttk.Frame(sample_frame)
        sample_btn_sub.grid(row=1, column=1, sticky="ns", pady=(2, 8))
        ttk.Button(sample_btn_sub, text="瀏覽檔案...", command=self.browse_input_file).pack(side="left", padx=(0, 6))
        ttk.Button(sample_btn_sub, text="🎵 從 KTV 歌庫選擇", command=self.pick_from_ktv_songs).pack(side="left")

        render_act_frame = ttk.Frame(sample_frame)
        render_act_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(render_act_frame, text="取樣並另存新檔", command=self.render_to_file).pack(side="left")
        self.render_progress = ttk.Progressbar(render_act_frame, orient="horizontal", mode="determinate", maximum=100)
        self.render_progress.pack(side="left", fill="x", expand=True, padx=(10, 10))
        ttk.Label(sample_frame, textvariable=self.render_status_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        sample_frame.columnconfigure(0, weight=1)

        # 4. 音檔特徵辨識
        feat_frame = ttk.LabelFrame(page_inner, text="音檔特徵辨識與自動變聲推薦", padding=12)
        feat_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(feat_frame, textvariable=self.features_var, wraplength=850, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        feat_btn_frame = ttk.Frame(feat_frame)
        feat_btn_frame.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(feat_btn_frame, text="辨識音檔特徵", command=self.analyze_source_file).pack(side="left", padx=(0, 8))
        ttk.Button(feat_btn_frame, text="依特徵自動變聲並存入聲音庫", command=self.auto_voice_to_soundboard).pack(side="left")
        ttk.Checkbutton(feat_frame, text="完成後自動加入音效選單", variable=self.auto_soundboard_var).grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=(8, 0)
        )
        feat_frame.columnconfigure(1, weight=1)

        # 5. 滑桿細部微調
        ctrl_frame = ttk.LabelFrame(page_inner, text="音訊效果細部微調 (即時套用)", padding=10)
        ctrl_frame.pack(fill="x", pady=(10, 0))
        ctrl_inner = ttk.Frame(ctrl_frame, padding=4)
        ctrl_inner.pack(fill="both", expand=True)

        controls = [
            ("音高倍率", self.pitch_var, 0.55, 1.9, self._update_settings),
            ("機器人混合", self.robot_var, 0.0, 1.0, self._update_settings),
            ("回音混合", self.echo_var, 0.0, 0.8, self._update_settings),
            ("失真強度", self.drive_var, 0.0, 0.75, self._update_settings),
            ("壓縮柔化", self.compressor_var, 0.0, 1.0, self._update_settings),
            ("去低頻 (高通)", self.highpass_var, 0.0, 1.0, self._update_settings),
            ("去高頻 (低通)", self.lowpass_var, 0.0, 1.0, self._update_settings),
            ("亮度強化 (Presence)", self.presence_var, 0.0, 1.0, self._update_settings),
            ("氣音細節 (Air)", self.air_var, 0.0, 1.0, self._update_settings),
            ("效果乾濕比", self.wet_var, 0.0, 1.0, self._update_settings),
            ("總輸出增益", self.gain_var, 0.4, 1.8, self._update_settings),
        ]

        for row, (label, var, min_v, max_v, callback) in enumerate(controls):
            ttk.Label(ctrl_inner, text=label).grid(row=row, column=0, sticky="w")
            scale = ttk.Scale(ctrl_inner, variable=var, from_=min_v, to=max_v, command=lambda _v, cb=callback: cb())
            scale.grid(row=row, column=1, sticky="ew", padx=(10, 10))
            value = ttk.Label(ctrl_inner, textvariable=tk.StringVar(value=f"{var.get():.2f}"))
            value.grid(row=row, column=2, sticky="e")
            var.trace_add("write", self._make_var_tracer(var, value))

        ctrl_inner.columnconfigure(1, weight=1)

        # 6. 聲音庫
        lib_frame = ttk.LabelFrame(page_inner, text="聲音庫（收藏與音色管理）", padding=12)
        lib_frame.pack(fill="x", pady=(10, 0))

        lib_list_frame = ttk.Frame(lib_frame)
        lib_list_frame.pack(fill="both", expand=True)

        columns = ("name", "preset", "duration", "created")
        self.library_tree = ttk.Treeview(lib_list_frame, columns=columns, show="headings", selectmode="browse", height=5)
        headings = {"name": "名稱", "preset": "預設", "duration": "長度(秒)", "created": "建立時間"}
        widths = {"name": 280, "preset": 120, "duration": 80, "created": 140}
        for key in columns:
            self.library_tree.heading(key, text=headings[key])
            self.library_tree.column(key, width=widths[key], anchor="w")
        self.library_tree.pack(side="left", fill="both", expand=True)
        lib_scroll = ttk.Scrollbar(lib_list_frame, orient="vertical", command=self.library_tree.yview)
        lib_scroll.pack(side="right", fill="y")
        self.library_tree.configure(yscrollcommand=lib_scroll.set)
        self.library_tree.bind("<Double-1>", lambda _e: self.play_library_entry())

        lib_btn_frame = ttk.Frame(lib_frame)
        lib_btn_frame.pack(fill="x", pady=(8, 0))
        lib_buttons = [
            ("播放預覽", self.play_library_entry),
            ("停止播放", self.stop_library_playback),
            ("載入滑桿", self.apply_library_settings),
            ("重新命名", self.rename_library_entry),
            ("匯出音檔", self.export_library_entry),
            ("匯入外部音檔", self.import_into_library),
            ("刪除", self.delete_library_entry),
        ]
        for idx, (label, cmd) in enumerate(lib_buttons):
            ttk.Button(lib_btn_frame, text=label, command=cmd).grid(row=0, column=idx, sticky="ew", padx=(0, 5))
        lib_btn_frame.columnconfigure(len(lib_buttons), weight=1)

        # 7. 音效選單（Soundboard，可在唱歌串流時即時插入音效）
        board_frame = ttk.LabelFrame(page_inner, text="音效選單 / 現場罐頭音效 (Soundboard)", padding=12)
        board_frame.pack(fill="x", pady=(10, 0))

        self.soundboard_menu = tk.Menu(self.root, tearoff=0)
        ttk.Menubutton(board_frame, text="快速音效清單", menu=self.soundboard_menu, direction="below").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(board_frame, textvariable=self.soundboard_status_var).grid(row=0, column=1, sticky="w", padx=(10, 0))

        board_btn_frame = ttk.Frame(board_frame)
        board_btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        board_buttons = [
            ("加入至音效板", self.add_to_soundboard),
            ("從音效板移除", self.remove_from_soundboard),
            ("播放音效", self.play_selected_sound),
            ("停止播放", self.stop_library_playback),
        ]
        for label, cmd in board_buttons:
            ttk.Button(board_btn_frame, text=label, command=cmd).pack(side="left", padx=(0, 8))

        ttk.Checkbutton(board_btn_frame, text="循環播放", variable=self.sound_loop_var).pack(side="left", padx=(4, 0))

        vol_frame = ttk.Frame(board_frame)
        vol_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(vol_frame, text="音效音量:").pack(side="left")
        ttk.Scale(vol_frame, variable=self.sound_volume_var, from_=0.0, to=1.5).pack(
            side="left", fill="x", expand=True, padx=(8, 8)
        )
        self.sound_volume_label = ttk.Label(vol_frame, textvariable=self.sound_volume_text_var)
        self.sound_volume_label.pack(side="right")
        self.sound_volume_var.trace_add("write", self._make_var_tracer(self.sound_volume_var, self.sound_volume_label))

        # 狀態列
        status_bar = ttk.Frame(page_inner)
        status_bar.pack(fill="x", pady=(10, 0))
        ttk.Label(status_bar, text="變聲引擎狀態:").pack(side="left")
        ttk.Label(status_bar, textvariable=self.status_var, font=("Microsoft JhengHei UI", 9, "bold")).pack(
            side="left", padx=(4, 0)
        )

        self.voice_log_text = tk.Text(page_inner, height=4, wrap="word", bg="#1e1e1e", fg="#00ff66", font=("Consolas", 9))
        self.voice_log_text.pack(fill="x", pady=(8, 0))
        self.voice_log_text.configure(state="disabled")

    def _build_ktv_tab(self, parent: ttk.Frame) -> None:
        inner = ttk.Frame(parent, padding=16)
        inner.pack(fill="both", expand=True)

        title_frame = ttk.Frame(inner)
        title_frame.pack(fill="x", pady=(0, 10))
        tk.Label(
            title_frame,
            text="🎤 KTV 雲端點歌系統與 AI 伴奏中心",
            font=("Microsoft JhengHei", 18, "bold"),
            fg="#2E7D32",
        ).pack(side="left")

        # 伺服器各端點連結
        info_frame = ttk.LabelFrame(inner, text="Web 連線端點（點擊直接於瀏覽器開啟）", padding=12)
        info_frame.pack(fill="x", pady=(0, 10))

        links = [
            ("📺 電視播放端", f"http://{LOCAL_IP}:{PORT}/player", "#1976D2"),
            ("📱 手機遙控端", f"http://{LOCAL_IP}:{PORT}/remote", "#D32F2F"),
            ("🕹️ 一體機介面", f"http://{LOCAL_IP}:{PORT}/combo", "#7B1FA2"),
            ("⚙️ 歌曲管理端", f"http://{LOCAL_IP}:{PORT}/admin", "#E65100"),
        ]
        for label, url, color in links:
            f = tk.Frame(info_frame)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=f"{label}: ", font=("Microsoft JhengHei", 10, "bold"), width=14, anchor="w").pack(side="left")
            link_lbl = tk.Label(f, text=url, font=("Consolas", 10, "underline"), fg=color, cursor="hand2")
            link_lbl.pack(side="left")
            link_lbl.bind("<Button-1>", lambda _e, u=url: webbrowser.open(u))

        # 歌庫狀態與連動按鈕
        stat_frame = ttk.LabelFrame(inner, text="KTV 歌庫統計與變聲器連動", padding=12)
        stat_frame.pack(fill="x", pady=(0, 10))

        stat_left = ttk.Frame(stat_frame)
        stat_left.pack(side="left", fill="both", expand=True)
        self.lbl_ktv_count = ttk.Label(stat_left, text="總歌曲數: 載入中...", font=("Microsoft JhengHei", 11, "bold"))
        self.lbl_ktv_count.pack(anchor="w", pady=(0, 4))
        self.lbl_ktv_size = ttk.Label(stat_left, text="佔用空間: 載入中...", font=("Microsoft JhengHei", 11, "bold"))
        self.lbl_ktv_size.pack(anchor="w")

        stat_right = ttk.Frame(stat_frame)
        stat_right.pack(side="right")
        ttk.Button(stat_right, text="🎵 挑選 KTV 歌曲傳入變聲器", command=self.pick_from_ktv_songs).pack(fill="x", pady=(0, 4))
        ttk.Button(stat_right, text="🔄 更新 yt-dlp 下載核心", command=self._trigger_ytdlp_update).pack(fill="x")

        # 系統日誌框 (接聽所有伺服器日誌與 audio-separator 進度)
        log_frame = ttk.LabelFrame(inner, text="即時伺服器與處理日誌 (含 audio-separator 分離)", padding=8)
        log_frame.pack(fill="both", expand=True)

        self.ktv_log_txt = tk.Text(log_frame, bg="#111", fg="#00e676", font=("Consolas", 9), wrap="word")
        self.ktv_log_txt.pack(fill="both", expand=True)
        self.ktv_log_txt.configure(state="disabled")

        self.update_ktv_stats()

    # ==========================================================================
    # KTV 歌庫連動特色 (方案 B)
    # ==========================================================================
    def pick_from_ktv_songs(self) -> None:
        songs = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith(".mp4")]
        if not songs:
            messagebox.showinfo("KTV 歌庫", "目前 KTV 歌庫中還沒有歌曲，請先在管理端加歌。")
            return

        pick_win = tk.Toplevel(self.root)
        pick_win.title("選擇 KTV 歌曲連動至變聲器")
        pick_win.geometry("520x400")
        pick_win.transient(self.root)
        pick_win.grab_set()

        ttk.Label(pick_win, text="請選擇要傳入變聲器的 KTV 歌曲（可進行特徵辨識、音色微調或轉調）:").pack(
            anchor="w", padx=12, pady=(10, 4)
        )

        listbox = tk.Listbox(pick_win, font=("Microsoft JhengHei", 10))
        listbox.pack(fill="both", expand=True, padx=12, pady=6)
        for s in songs:
            listbox.insert("end", s)

        def on_confirm():
            sel = listbox.curselection()
            if not sel:
                return
            chosen = songs[sel[0]]
            chosen_path = SONGS_DIR / chosen
            self.input_file_var.set(str(chosen_path))
            self.render_status_var.set(f"已從 KTV 連動: {chosen}")
            self.features_var.set(f"已載入 KTV 歌曲 {chosen}，正在分析特徵...")
            self._append_voice_log(f"已從 KTV 歌庫載入: {chosen_path}")
            self._start_job(lambda: self._analyze_worker(chosen_path), "正在辨識 KTV 歌曲特徵...", quiet=True)
            pick_win.destroy()

        btn_box = ttk.Frame(pick_win)
        btn_box.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Button(btn_box, text="確定連動", command=on_confirm).pack(side="right", padx=(6, 0))
        ttk.Button(btn_box, text="取消", command=pick_win.destroy).pack(side="right")

    def _trigger_ytdlp_update(self) -> None:
        handle_update_ytdlp()

    def update_ktv_stats(self) -> None:
        try:
            songs = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith(".mp4")]
            count = len(songs)
            total_size = sum((SONGS_DIR / f).stat().st_size for f in songs)
            size_mb = total_size / (1024 * 1024)
            self.lbl_ktv_count.config(text=f"🎵 總歌曲數: {count} 首")
            self.lbl_ktv_size.config(text=f"💾 佔用空間: {size_mb:.2f} MB")
        except Exception:
            pass
        self.root.after(5000, self.update_ktv_stats)

    # ==========================================================================
    # 音訊裝置與 ASIO 管理
    # ==========================================================================
    def refresh_devices(self) -> None:
        self.hostapis = sd.query_hostapis()
        self.devices = sd.query_devices()
        self.input_map.clear()
        self.output_map.clear()
        self.device_hostapi_map.clear()
        self.device_channel_counts.clear()

        input_names: List[str] = []
        output_names: List[str] = []

        for idx, device in enumerate(self.devices):
            hostapi_name = self.hostapis[device["hostapi"]]["name"]
            self.device_hostapi_map[idx] = hostapi_name
            self.device_channel_counts[idx] = max(device["max_input_channels"], device["max_output_channels"])

            asio_mark = " [ASIO 專業低延遲]" if "ASIO" in hostapi_name.upper() else f" [{hostapi_name}]"

            if device["max_input_channels"] > 0:
                label = f"{idx}:{asio_mark} {device['name']}"
                input_names.append(label)
                self.input_map[label] = idx

            if device["max_output_channels"] > 0:
                label = f"{idx}:{asio_mark} {device['name']}"
                output_names.append(label)
                self.output_map[label] = idx

        self.input_box["values"] = input_names
        self.output_box["values"] = output_names

        if self.input_var.get() not in self.input_map:
            pref_in = next(
                (name for name in input_names if "[ASIO" in name),
                next(
                    (name for name in input_names if "[Windows WASAPI]" in name and "mic" in name.lower()),
                    next((name for name in input_names if "[Windows WASAPI]" in name), input_names[0] if input_names else ""),
                ),
            )
            if pref_in:
                self.input_var.set(pref_in)

        if self.output_var.get() not in self.output_map:
            pref_out = next(
                (name for name in output_names if "[ASIO" in name),
                next(
                    (name for name in output_names if "cable" in name.lower() or "vb-audio" in name.lower()),
                    next((name for name in output_names if "[Windows WASAPI]" in name), output_names[0] if output_names else ""),
                ),
            )
            if pref_out:
                self.output_var.set(pref_out)

        self._on_device_selection_changed()
        self._append_voice_log("音訊裝置清單已重新整理完成。")

    def _on_device_selection_changed(self) -> None:
        in_label = self.input_var.get()
        out_label = self.output_var.get()
        in_idx = self.input_map.get(in_label)
        out_idx = self.output_map.get(out_label)

        is_asio = False
        if in_idx is not None and "ASIO" in self.device_hostapi_map.get(in_idx, "").upper():
            is_asio = True
        if out_idx is not None and "ASIO" in self.device_hostapi_map.get(out_idx, "").upper():
            is_asio = True

        if is_asio:
            self.buffer_size_var.set(256)
            in_ch_count = int(sd.query_devices(in_idx)["max_input_channels"]) if in_idx is not None else 2
            self.in_ch_box["values"] = [f"第 {i+1} 聲道" for i in range(max(1, in_ch_count))]
            if self.input_ch_var.get() not in self.in_ch_box["values"]:
                self.input_ch_var.set("第 1 聲道")
            self._append_voice_log("⚡ 已切換至 ASIO 專業模式，建議緩衝區 128~256 以取得最低延遲。")
        else:
            self.in_ch_box["values"] = ["第 1 聲道 (預設)", "第 2 聲道"]

    # ==========================================================================
    # 變聲核心操作
    # ==========================================================================
    def start(self) -> None:
        if self.engine.running:
            messagebox.showinfo("已在執行中", "變聲器目前已在執行中。")
            return

        try:
            in_dev = self.input_map[self.input_var.get()]
            out_dev = self.output_map[self.output_var.get()]
        except KeyError:
            messagebox.showerror("裝置未選擇", "請先選好輸入與輸出裝置。")
            return

        in_host = self.device_hostapi_map.get(in_dev, "")
        out_host = self.device_hostapi_map.get(out_dev, "")
        is_asio = "ASIO" in in_host.upper() or "ASIO" in out_host.upper()

        in_ch_str = self.input_ch_var.get()
        in_ch_idx = 0
        if "第" in in_ch_str:
            try:
                in_ch_idx = int(in_ch_str.replace("第", "").replace("聲道", "").replace("(預設)", "").strip()) - 1
            except Exception:
                in_ch_idx = 0

        self._update_settings()
        block_size = int(self.buffer_size_var.get())

        try:
            self.engine.start(
                input_device=in_dev,
                output_device=out_dev,
                is_asio=is_asio,
                input_channel_idx=max(0, in_ch_idx),
                block_size=block_size,
            )
        except Exception as exc:
            messagebox.showerror("啟動失敗", f"無法啟動音訊串流:\n{exc}\n\n若使用 ASIO 請確認驅動未被其他程式獨佔。")
            self._append_voice_log(f"啟動失敗: {exc}")
            return

        self.status_var.set("執行中" + (" (ASIO 協議)" if is_asio else ""))
        self._save_config()
        messagebox.showinfo("啟動成功", "變聲引擎已啟動！您可以在耳機監聽或將輸出送入 Discord/KTV。")

    def stop(self) -> None:
        self.engine.stop()
        self.status_var.set("已停止")

    def _bind_wheel_scroll(self, canvas: tk.Canvas, inner: ttk.Frame) -> None:
        def on_wheel(event: tk.Event) -> str:
            if event.delta:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        canvas.bind("<MouseWheel>", on_wheel)
        inner.bind("<MouseWheel>", on_wheel)

    def _make_var_tracer(self, variable: tk.DoubleVar, label: ttk.Label):
        def tracer(*_args) -> None:
            label.config(text=f"{variable.get():.2f}")

        return tracer

    def _append_voice_log(self, text: str) -> None:
        self.voice_log_text.configure(state="normal")
        self.voice_log_text.insert("end", text + "\n")
        self.voice_log_text.see("end")
        self.voice_log_text.configure(state="disabled")

    def _poll_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_voice_log(msg)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_logs)

    def _poll_system_logs(self) -> None:
        try:
            while not system_log_queue.empty():
                msg = system_log_queue.get_nowait()
                self.ktv_log_txt.config(state="normal")
                self.ktv_log_txt.insert("end", msg + "\n")
                self.ktv_log_txt.see("end")
                self.ktv_log_txt.config(state="disabled")
        except Exception:
            pass
        self.root.after(120, self._poll_system_logs)

    # 檔案取樣
    def browse_input_file(self) -> None:
        path = filedialog.askopenfilename(title="選擇音檔", filetypes=AUDIO_INPUT_TYPES)
        if not path:
            return
        self.input_file_var.set(path)
        self.render_status_var.set(f"已選擇: {Path(path).name}")
        self.features_var.set(f"已選擇 {Path(path).name}，正在辨識特徵...")
        self.last_features = None
        self._append_voice_log(f"已選擇音檔: {path}")
        self._start_job(lambda: self._analyze_worker(Path(path)), "正在辨識特徵...", quiet=True)

    def render_to_file(self) -> None:
        if self.render_thread is not None and self.render_thread.is_alive():
            messagebox.showinfo("處理中", "目前正在處理音檔，請稍候。")
            return

        source = self.input_file_var.get()
        source_path = Path(source)
        if source == "尚未選擇音檔" or not source_path.is_file():
            messagebox.showerror("音檔未選擇", "請先選擇存在的音檔。")
            return

        suggested = source_path.with_name(f"{source_path.stem}_變聲.wav")
        target = filedialog.asksaveasfilename(
            title="另存變聲新檔",
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
            defaultextension=".wav",
            filetypes=AUDIO_OUTPUT_TYPES,
        )
        if not target:
            return

        settings = self.shared_settings.snapshot()
        self._update_settings()
        self.render_progress["value"] = 0
        self.render_status_var.set("正在處理...")
        self._append_voice_log(f"開始離線取樣: {source_path.name} -> {Path(target).name}")
        self._start_job(lambda: self._render_worker(source_path, Path(target), settings), "正在處理...")

    def _render_worker(self, source: Path, target: Path, settings: EffectSettings) -> None:
        try:
            duration = render_file(
                source,
                target,
                settings,
                progress=lambda ratio: self.render_queue.put(("progress", ratio)),
            )
        except Exception as exc:
            self.render_queue.put(("error", str(exc)))
            return
        self.render_queue.put(("done", {"path": str(target), "duration": duration}))

    def _poll_render_status(self) -> None:
        try:
            while True:
                kind, payload = self.render_queue.get_nowait()
                handler = self._job_handlers.get(kind, self._on_job_error)
                handler(payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_render_status)

    def _on_render_progress(self, payload: object) -> None:
        self.render_progress["value"] = float(payload) * 100.0  # type: ignore[arg-type]

    def _on_job_error(self, payload: object) -> None:
        self.render_progress["value"] = 0
        self.render_status_var.set("處理失敗")
        self._append_voice_log(f"處理失敗: {payload}")
        messagebox.showerror("處理失敗", str(payload))

    def _on_render_finished(self, payload: Dict[str, object]) -> None:
        target = Path(str(payload["path"]))
        duration = float(payload.get("duration", 0.0))  # type: ignore[arg-type]
        self.render_status_var.set(f"完成: {target.name}")

        if self.auto_library_var.get():
            preset = self.preset_var.get()
            source_name = Path(self.input_file_var.get()).name
            try:
                entry = self.library.add_wav(target, target.stem, source_name, preset, duration, self.shared_settings.snapshot())
            except Exception as exc:
                self._append_voice_log(f"加入聲音庫失敗: {exc}")
                return
            self.refresh_library()
            self._append_voice_log(f"已自動收藏進聲音庫: {entry['name']}")
            messagebox.showinfo("取樣完成", f"已儲存音檔:\n{target}\n\n已同步收藏至聲音庫: {entry['name']}")
        else:
            messagebox.showinfo("取樣完成", f"已儲存音檔:\n{target}")

    def _source_path(self) -> Optional[Path]:
        source = self.input_file_var.get()
        path = Path(source)
        if source == "尚未選擇音檔" or not path.is_file():
            messagebox.showerror("音檔未選擇", "請先選擇一個存在的音訊或視訊檔案。")
            return None
        return path

    def _start_job(self, worker: Callable[[], None], busy_text: str, quiet: bool = False) -> bool:
        if self.render_thread is not None and self.render_thread.is_alive():
            if not quiet:
                messagebox.showinfo("處理中", "目前正在處理其他任務，請稍候。")
            return False
        self.render_progress["value"] = 0
        self.render_status_var.set(busy_text)
        self.render_thread = threading.Thread(target=worker, daemon=True)
        self.render_thread.start()
        return True

    def analyze_source_file(self) -> None:
        source = self._source_path()
        if source is None:
            return
        self.features_var.set(f"正在分析 {source.name} ...")
        self._append_voice_log(f"開始特徵分析: {source}")
        self._start_job(lambda: self._analyze_worker(source), "正在辨識特徵...")

    def _analyze_worker(self, source: Path) -> None:
        try:
            features = analyze_audio(source)
        except Exception as exc:
            self.render_queue.put(("error", str(exc)))
            return
        self.render_queue.put(("features", {"path": str(source), "features": features}))

    def _on_features_ready(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        features = data.get("features")
        if not isinstance(features, AudioFeatures):
            return

        self.last_features = features
        label, settings, reasons = recommend_settings(features)
        detail = "\n".join(f"・{reason}" for reason in reasons)
        self.features_var.set(f"特徵: {features.summary()}\n建議設定（{label}）:\n{detail}")
        self.render_progress["value"] = 100
        self.render_status_var.set("特徵辨識完成")
        self._append_voice_log(f"特徵分析完成: {features.summary()}")

    def auto_voice_to_soundboard(self) -> None:
        source = self._source_path()
        if source is None:
            return
        self.features_var.set(f"正在分析並變聲 {source.name} ...")
        self._start_job(lambda: self._auto_worker(source), "正在依特徵自動變聲...")

    def _auto_worker(self, source: Path) -> None:
        try:
            features = analyze_audio(source)
            label, settings, reasons = recommend_settings(features)
            name = f"{source.stem}_{label}"
            self.render_queue.put(("progress", 0.05))
            entry, duration = self.library.render_into_library(
                source,
                name,
                label,
                settings,
                features=features.to_dict(),
                progress=lambda ratio: self.render_queue.put(("progress", ratio)),
            )
        except Exception as exc:
            self.render_queue.put(("error", str(exc)))
            return

        self.render_queue.put(
            (
                "auto_done",
                {
                    "entry": entry,
                    "features": features,
                    "reasons": reasons,
                    "settings": settings,
                    "duration": duration,
                },
            )
        )

    def _on_auto_finished(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        entry = data.get("entry")
        features = data.get("features")
        settings = data.get("settings")
        reasons = data.get("reasons") or []
        if not isinstance(entry, dict) or not isinstance(settings, EffectSettings):
            return

        self.last_features = features if isinstance(features, AudioFeatures) else None
        preset = str(entry.get("preset", ""))
        if preset in PRESETS or self.effects.find_by_name(_strip_effect_mark(preset)) is not None:
            self.preset_var.set(preset)
        self._apply_settings_to_ui(settings)

        if self.auto_soundboard_var.get():
            try:
                self.soundboard.add(str(entry.get("id", "")), loop=self.sound_loop_var.get())
            except Exception:
                pass

        self.refresh_library()
        self.render_progress["value"] = 100
        self.render_status_var.set(f"完成: {entry.get('name')}")
        self._append_voice_log(f"已依特徵變聲並存入聲音庫: {entry.get('name')}")
        messagebox.showinfo("自動變聲完成", f"已收藏進聲音庫:\n{entry.get('name')}")

    # 聲音庫與音效選單管理
    def refresh_library(self) -> None:
        self.stop_library_playback()
        self.soundboard.prune()
        for item in self.library_tree.get_children():
            self.library_tree.delete(item)
        for entry in self.library.entries():
            self.library_tree.insert(
                "",
                "end",
                iid=str(entry["id"]),
                values=(
                    entry.get("name", ""),
                    entry.get("preset", ""),
                    entry.get("duration", ""),
                    entry.get("created_at", ""),
                ),
            )
        self.refresh_soundboard_menu()

    def refresh_soundboard_menu(self) -> None:
        items = self.soundboard.items()
        self.soundboard_menu.delete(0, "end")
        if not items:
            self.soundboard_menu.add_command(label="（尚無音效，請先在聲音庫中加入）", state="disabled")
        else:
            for index, item in enumerate(items):
                label = str(item["name"]) + ("（循環）" if item.get("loop") else "")
                submenu = tk.Menu(self.soundboard_menu, tearoff=0)
                submenu.add_command(label="播放", command=lambda i=index: self.play_soundboard_item(i))
                submenu.add_command(label="切換循環", command=lambda i=index: self.toggle_soundboard_loop(i))
                submenu.add_command(label="從清單移除", command=lambda i=index: self.remove_soundboard_item(i))
                self.soundboard_menu.add_cascade(label=label, menu=submenu)

            self.soundboard_menu.add_separator()
            self.soundboard_menu.add_command(label="全部停止播放", command=self.stop_library_playback)

        self.soundboard_status_var.set(f"音效選單共 {len(items)} 個項目")

    def add_to_soundboard(self) -> None:
        entry = self._selected_library_entry()
        if entry is None:
            return
        added = self.soundboard.add(str(entry.get("id", "")), loop=self.sound_loop_var.get())
        if not added:
            messagebox.showinfo("已存在", f"「{entry.get('name')}」已經在音效選單裡了。")
            return
        self.refresh_soundboard_menu()
        self._append_voice_log(f"已加入音效板: {entry.get('name')}")

    def remove_from_soundboard(self) -> None:
        entry = self._selected_library_entry()
        if entry is None:
            return
        idx = self.soundboard.index_of(str(entry.get("id", "")))
        if idx is not None:
            self.remove_soundboard_item(idx)

    def remove_soundboard_item(self, index: int) -> None:
        self.soundboard.remove_at(index)
        self.refresh_soundboard_menu()

    def toggle_soundboard_loop(self, index: int) -> None:
        items = self.soundboard.items()
        if 0 <= index < len(items):
            cur = bool(items[index].get("loop"))
            self.soundboard.set_loop(index, not cur)
            self.refresh_soundboard_menu()

    def play_soundboard_item(self, index: int) -> None:
        items = self.soundboard.items()
        if not 0 <= index < len(items):
            return
        path = Path(str(items[index]["path"]))
        if not path.is_file():
            messagebox.showerror("檔案遺失", f"找不到音檔: {path}")
            return
        out_dev = self.output_map.get(self.output_var.get())
        try:
            play_sound_file(path, out_dev, loop=bool(items[index].get("loop")), volume=self.sound_volume_var.get())
        except Exception as exc:
            messagebox.showerror("播放失敗", str(exc))

    def play_selected_sound(self) -> None:
        entry = self._selected_library_entry()
        if entry is None:
            return
        path = self.library.audio_path(entry)
        out_dev = self.output_map.get(self.output_var.get())
        try:
            play_sound_file(path, out_dev, loop=self.sound_loop_var.get(), volume=self.sound_volume_var.get())
        except Exception as exc:
            messagebox.showerror("播放失敗", str(exc))

    def _selected_library_entry(self) -> Optional[Dict[str, object]]:
        sel = self.library_tree.selection()
        if not sel:
            messagebox.showinfo("未選取", "請先在聲音庫中點選一個項目。")
            return None
        return self.library.find(sel[0])

    def play_library_entry(self) -> None:
        entry = self._selected_library_entry()
        if entry is None:
            return
        path = self.library.audio_path(entry)
        try:
            data, rate = _read_wav_stdlib(path)
            sd.play(data, rate)
        except Exception as exc:
            messagebox.showerror("播放失敗", str(exc))

    def stop_library_playback(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass

    def apply_library_settings(self) -> None:
        entry = self._selected_library_entry()
        if entry is None:
            return
        settings = EffectSettings.from_dict(entry.get("settings") or {})  # type: ignore[arg-type]
        self._apply_settings_to_ui(settings)
        self._append_voice_log(f"已套用聲音庫「{entry.get('name')}」的效果參數。")

    def rename_library_entry(self) -> None:
        entry = self._selected_library_entry()
        if entry is None:
            return
        name = simpledialog.askstring("重新命名", "請輸入新名稱:", initialvalue=str(entry.get("name", "")), parent=self.root)
        if name and name.strip():
            self.library.rename(str(entry["id"]), name.strip())
            self.refresh_library()

    def delete_library_entry(self) -> None:
        entry = self._selected_library_entry()
        if entry is None:
            return
        if messagebox.askyesno("確認刪除", f"確定要從聲音庫刪除「{entry.get('name')}」嗎?"):
            self.library.delete(str(entry["id"]))
            self.refresh_library()

    def export_library_entry(self) -> None:
        entry = self._selected_library_entry()
        if entry is None:
            return
        source = self.library.audio_path(entry)
        target = filedialog.asksaveasfilename(
            title="匯出音檔",
            initialfile=f"{entry.get('name', 'voice')}.wav",
            defaultextension=".wav",
            filetypes=AUDIO_OUTPUT_TYPES,
        )
        if target:
            shutil.copyfile(source, target)
            messagebox.showinfo("匯出完成", f"已成功匯出至:\n{target}")

    def import_into_library(self) -> None:
        path = filedialog.askopenfilename(title="匯入音檔", filetypes=AUDIO_INPUT_TYPES)
        if not path:
            return
        settings = self.shared_settings.snapshot()
        preset = self.preset_var.get()
        self._start_job(
            lambda: self._import_worker(Path(path), Path(path).stem, preset, settings),
            f"正在匯入: {Path(path).name}",
        )

    def _import_worker(self, source: Path, name: str, preset: str, settings: EffectSettings) -> None:
        try:
            entry = self.library.import_audio(source, name, preset, settings)
        except Exception as exc:
            self.render_queue.put(("error", str(exc)))
            return
        self.render_queue.put(("import_done", entry))

    def _on_import_finished(self, payload: Dict[str, object]) -> None:
        self.render_progress["value"] = 100
        self.render_status_var.set("匯入完成")
        self.refresh_library()
        messagebox.showinfo("匯入完成", f"已收藏音檔: {payload.get('name')}")

    # 效果預設與自訂
    def _refresh_effect_choices(self, selected: str = "") -> None:
        names = list(PRESETS.keys()) + [f"{USER_EFFECT_MARK}{name}" for name in self.effects.names()]
        self.preset_box["values"] = names
        if selected:
            self.preset_var.set(selected)
        elif self.preset_var.get() not in names:
            self.preset_var.set("自然")

    def _on_preset_selected(self) -> None:
        label = self.preset_var.get()
        if label in PRESETS:
            self._apply_preset(label)
            return
        name = _strip_effect_mark(label)
        effect = self.effects.find_by_name(name)
        if effect is not None:
            settings = EffectSettings.from_dict(effect["settings"])  # type: ignore[arg-type]
            self._apply_settings_to_ui(settings)

    def _apply_preset(self, name: str) -> None:
        self._apply_settings_to_ui(PRESETS.get(name, PRESETS["自然"]))

    def _apply_settings_to_ui(self, settings: EffectSettings) -> None:
        self.pitch_var.set(settings.pitch_ratio)
        self.robot_var.set(settings.robot_mix)
        self.echo_var.set(settings.echo_mix)
        self.drive_var.set(settings.distortion_drive)
        self.compressor_var.set(settings.compressor_amount)
        self.highpass_var.set(settings.highpass_amount)
        self.lowpass_var.set(settings.lowpass_amount)
        self.presence_var.set(settings.presence_amount)
        self.air_var.set(settings.air_amount)
        self.wet_var.set(settings.wet_mix)
        self.gain_var.set(settings.output_gain)
        self._update_settings()

    def _update_settings(self) -> None:
        self.shared_settings.update(
            pitch_ratio=self.pitch_var.get(),
            robot_mix=self.robot_var.get(),
            echo_mix=self.echo_var.get(),
            distortion_drive=self.drive_var.get(),
            compressor_amount=self.compressor_var.get(),
            highpass_amount=self.highpass_var.get(),
            lowpass_amount=self.lowpass_var.get(),
            presence_amount=self.presence_var.get(),
            air_amount=self.air_var.get(),
            wet_mix=self.wet_var.get(),
            output_gain=self.gain_var.get(),
        )

    def save_effect(self) -> None:
        settings = self.shared_settings.snapshot()
        name = simpledialog.askstring("儲存自訂效果", "請輸入效果名稱:", parent=self.root)
        if name and name.strip():
            try:
                self.effects.add(name.strip(), settings)
                self._refresh_effect_choices(selected=f"{USER_EFFECT_MARK}{name.strip()}")
                messagebox.showinfo("已儲存", f"效果「{name.strip()}」已儲存！")
            except Exception as exc:
                messagebox.showerror("無法儲存", str(exc))

    def update_effect(self) -> None:
        label = self.preset_var.get()
        if label in PRESETS:
            messagebox.showinfo("請選自訂效果", "內建預設無法被覆蓋，請先儲存為新的自訂效果。")
            return
        name = _strip_effect_mark(label)
        effect = self.effects.find_by_name(name)
        if effect:
            self.effects.update_settings(str(effect["id"]), self.shared_settings.snapshot())
            messagebox.showinfo("已更新", f"自訂效果「{name}」已更新！")

    def rename_effect(self) -> None:
        label = self.preset_var.get()
        if label in PRESETS:
            messagebox.showinfo("請選自訂效果", "無法重新命名內建預設。")
            return
        name = _strip_effect_mark(label)
        effect = self.effects.find_by_name(name)
        if effect:
            new_name = simpledialog.askstring("重新命名", "請輸入新名稱:", initialvalue=name, parent=self.root)
            if new_name and new_name.strip():
                try:
                    self.effects.rename(str(effect["id"]), new_name.strip())
                    self._refresh_effect_choices(selected=f"{USER_EFFECT_MARK}{new_name.strip()}")
                except Exception as exc:
                    messagebox.showerror("失敗", str(exc))

    def delete_effect(self) -> None:
        label = self.preset_var.get()
        if label in PRESETS:
            messagebox.showinfo("請選自訂效果", "無法刪除內建預設。")
            return
        name = _strip_effect_mark(label)
        effect = self.effects.find_by_name(name)
        if effect and messagebox.askyesno("確認刪除", f"確定要刪除「{name}」嗎?"):
            self.effects.delete(str(effect["id"]))
            self._refresh_effect_choices()

    # 設定儲存與關閉
    def _save_config(self) -> None:
        data = {
            "input": self.input_var.get(),
            "output": self.output_var.get(),
            "preset": self.preset_var.get(),
            "buffer_size": self.buffer_size_var.get(),
            "input_channel": self.input_ch_var.get(),
            "output_channel": self.output_ch_var.get(),
            "pitch_ratio": self.pitch_var.get(),
            "robot_mix": self.robot_var.get(),
            "echo_mix": self.echo_var.get(),
            "distortion_drive": self.drive_var.get(),
            "compressor_amount": self.compressor_var.get(),
            "highpass_amount": self.highpass_var.get(),
            "lowpass_amount": self.lowpass_var.get(),
            "presence_amount": self.presence_var.get(),
            "air_amount": self.air_var.get(),
            "wet_mix": self.wet_var.get(),
            "output_gain": self.gain_var.get(),
            "auto_library": self.auto_library_var.get(),
            "sound_volume": self.sound_volume_var.get(),
            "sound_loop": self.sound_loop_var.get(),
        }
        CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return

        self.input_var.set(data.get("input", ""))
        self.output_var.set(data.get("output", ""))
        self.preset_var.set(data.get("preset", "自然"))
        self.buffer_size_var.set(int(data.get("buffer_size", 512)))
        self.input_ch_var.set(data.get("input_channel", "第 1 聲道 (預設)"))
        self.output_ch_var.set(data.get("output_channel", "第 1-2 聲道 (立體聲)"))
        self.pitch_var.set(data.get("pitch_ratio", 1.0))
        self.robot_var.set(data.get("robot_mix", 0.0))
        self.echo_var.set(data.get("echo_mix", 0.0))
        self.drive_var.set(data.get("distortion_drive", 0.0))
        self.compressor_var.set(data.get("compressor_amount", 0.0))
        self.highpass_var.set(data.get("highpass_amount", 0.0))
        self.lowpass_var.set(data.get("lowpass_amount", 0.0))
        self.presence_var.set(data.get("presence_amount", 0.0))
        self.air_var.set(data.get("air_amount", 0.0))
        self.wet_var.set(data.get("wet_mix", 1.0))
        self.gain_var.set(data.get("output_gain", 1.0))
        self.auto_library_var.set(bool(data.get("auto_library", True)))
        self.sound_volume_var.set(float(data.get("sound_volume", 1.0)))
        self.sound_loop_var.set(bool(data.get("sound_loop", False)))
        self._update_settings()

    def on_close(self) -> None:
        self._save_config()
        self.stop_library_playback()
        self.engine.stop()
        self.root.destroy()


# ==============================================================================
# 程式入口點
# ==============================================================================
def main() -> None:
    # 多進程防護（打包 EXE 與 Spleeter 子進程必備）
    multiprocessing.freeze_support()

    # 啟動背景 Flask + SocketIO 伺服器
    server_t = threading.Thread(target=run_server_thread, daemon=True)
    server_t.start()

    # 啟動整合型 GUI 介面
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    UnifiedKTVVoiceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
