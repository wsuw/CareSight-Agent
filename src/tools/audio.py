import os
import sys
import tempfile
import asyncio
import queue
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import edge_tts
import pygame
from src.config.settings import settings

class AudioProcessor:
    """
    音频处理模块
    处理语音录制、Whisper 语音识别和 Edge-TTS 语音合成。
    """
    def __init__(self):
        # 延迟加载，防止卡启动，这里我们可以先不加载或加载模型
        print("🎙️ 正在加载 Whisper 模型...")
        self.model = WhisperModel(settings.WHISPER_MODEL, device="cpu", compute_type="int8")
        
        # 初始化 pygame 音频混音器用于播放合成的声音
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
        pygame.mixer.init()

    def record_audio(self, fs=16000) -> str:
        """
        利用麦克风录制音频并保存至临时文件，通过回车控制起止
        """
        print("\n🎤 准备开始录音... (按一次 [回车键] 结束并发送)")
        q = queue.Queue()
        
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        # 生成临时文件
        temp_audio_file = tempfile.mktemp(suffix='.wav')
        try:
            with sf.SoundFile(temp_audio_file, mode='x', samplerate=fs, channels=1, subtype='PCM_16') as file:
                with sd.InputStream(samplerate=fs, channels=1, callback=callback):
                    # 等待用户敲回车结束
                    input("🔊 正在录音中... [结束请按回车]")
                
                # 用户敲击回车，流关闭，将多余的数据写入文件
                while not q.empty():
                    file.write(q.get())
            return temp_audio_file
        except Exception as e:
            print(f"❌ 录音发生错误: {e}")
            return ""

    def transcribe(self, audio_file_path: str) -> str:
        """
        使用 Faster-Whisper 转录录音文件
        """
        if not audio_file_path or not os.path.exists(audio_file_path):
            return ""
        
        print("🧠 正在识别您的语音...")
        segments, info = self.model.transcribe(audio_file_path, language="zh")
        text = "".join([segment.text for segment in segments]).strip()
        
        # 识别后清理
        try:
            os.remove(audio_file_path)
        except OSError:
            pass
            
        print(f"✅ 语音识别结果: {text}")
        return text

    def speak(self, text: str):
        """
        使用 Edge-TTS 合成并播放语音
        """
        if not text:
            return
            
        print("🔊 正在进行语音播报...")
        
        import re
        # 清理文本中的 Markdown 标记，防止 TTS 读出 "星号" 或 "井号"
        clean_text = re.sub(r'\*+', '', text)
        clean_text = re.sub(r'#+\s*', '', clean_text)
        clean_text = re.sub(r'`+', '', clean_text)
        # 移除方括号 (常用于链接或标记)
        clean_text = re.sub(r'\[|\]|<|>', '', clean_text)
        
        # 可以选择 zh-CN-XiaoxiaoNeural (女), zh-CN-YunxiNeural (男)
        voice = "zh-CN-XiaoxiaoNeural"
        temp_mp3 = tempfile.mktemp(suffix='.mp3')
        
        async def _synthesize():
            communicate = edge_tts.Communicate(clean_text, voice)
            await communicate.save(temp_mp3)
            
        asyncio.run(_synthesize())
        
        try:
            pygame.mixer.music.load(temp_mp3)
            pygame.mixer.music.play()
            
            # 阻塞主线程以防退出，直到播放完毕
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"⚠️ 语音播放失败: {e}")
        finally:
            # 清理
            try:
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
            except OSError:
                pass


audio_processor = AudioProcessor()
