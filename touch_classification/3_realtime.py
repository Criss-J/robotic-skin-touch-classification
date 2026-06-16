"""
Robotic Skin - 실시간 터치 분류기 (버퍼 방식)
==============================================
동작 방식:
    - 신호 없음 → idle 자동 표시
    - 신호 감지 → 현재 버퍼 512ms 그대로 CNN 판단
    - onset 위치 신경 X → 파형 모양으로만 판단

실행:
    python 3_realtime_classify.py
"""

import socket
import threading
import numpy as np
import time
import platform
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from collections import deque
import tensorflow as tf

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
UDP_IP      = "0.0.0.0"
UDP_PORT_1  = 12345
UDP_PORT_2  = 12346
UDP_PORT_3  = 12347
SAMPLE_RATE = 10000
WINDOW_SIZE = 5120
DISPLAY_SIZE= 10000
MODEL_PATH  = "touch_classifier_3mic.h5"

ONSET_THRESHOLD = 0.003  # 신호 감지 임계값
COOLDOWN        = 1.5    # 분류 후 대기 시간

TOUCH_CLASSES = ['idle', 'scratch', 'tap']
LABELS = {
    'idle':    'IDLE',
    'scratch': 'SCRATCH',
    'tap':     'TAP',
}
COLORS = {
    'idle':    '#95a5a6',
    'scratch': '#e74c3c',
    'tap':     '#2ecc71',
}
DESCRIPTIONS = {
    'idle':    '아무것도 안함',
    'scratch': '손톱으로 긁기',
    'tap':     '손가락으로 치기',
}

STATE_LISTENING = 'listening'
STATE_RESULT    = 'result'

system = platform.system()
plt.rcParams['axes.unicode_minus'] = False
if system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    fonts = [f.name for f in fm.fontManager.ttflist]
    plt.rcParams['font.family'] = 'NanumGothic' if 'NanumGothic' in fonts else 'DejaVu Sans'


# ──────────────────────────────────────────
# Feature Map
# ──────────────────────────────────────────
def make_feature_maps(audio_3ch):
    """학습 코드와 완전히 동일한 방식으로 feature 추출"""
    mic1 = audio_3ch[0]
    mic2 = audio_3ch[1]
    mic3 = audio_3ch[2] if audio_3ch.shape[0] > 2 else audio_3ch[0]

    # DC offset 제거
    mic1 = mic1 - np.mean(mic1)
    mic2 = mic2 - np.mean(mic2)
    mic3 = mic3 - np.mean(mic3)

    stride    = 64
    frame_len = 128
    n_frames  = (WINDOW_SIZE - frame_len) // stride + 1

    def rms_frames(mic):
        return np.array([np.sqrt(np.mean(mic[i*stride:i*stride+frame_len]**2))
                         for i in range(n_frames)])

    int1 = rms_frames(mic1)
    int2 = rms_frames(mic2)
    int3 = rms_frames(mic3)
    intensity = np.stack([int1, int2, int3], axis=1)  # (n_frames, 3)

    # 가장 강한 마이크
    stds     = [mic1.std(), mic2.std(), mic3.std()]
    strongest = [mic1, mic2, mic3][np.argmax(stds)]

    # 피크 구간 중심 정렬
    win = 1024
    best_start = 0
    best_rms   = 0.0
    for i in range(0, len(strongest) - win, 128):
        r = np.sqrt(np.mean(strongest[i:i+win]**2))
        if r > best_rms:
            best_rms   = r
            best_start = i

    center = best_start + win // 2
    half   = WINDOW_SIZE // 2
    start  = max(0, center - half)
    end    = start + WINDOW_SIZE
    if end > len(strongest):
        end   = len(strongest)
        start = max(0, end - WINDOW_SIZE)
    seg = strongest[start:end]
    if len(seg) < WINDOW_SIZE:
        seg = np.pad(seg, (0, WINDOW_SIZE - len(seg)))

    _, _, Sxx = signal.spectrogram(seg, fs=SAMPLE_RATE, nperseg=128, noverlap=64)
    log_Sxx   = np.log1p(Sxx[:64])

    min_f     = min(intensity.shape[0], log_Sxx.shape[1])
    intensity = intensity[:min_f]
    log_Sxx   = log_Sxx[:, :min_f].T

    X_int  = intensity[np.newaxis, :, :, np.newaxis]
    X_spec = log_Sxx[np.newaxis, :, :, np.newaxis]
    return X_int, X_spec


# ──────────────────────────────────────────
# UDP 수신 스레드
# ──────────────────────────────────────────
class UDPReceiver(threading.Thread):
    def __init__(self, buffer, port):
        super().__init__(daemon=True)
        self.buffer    = buffer
        self.running   = True
        self.connected = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, port))
        self.sock.settimeout(1.0)

    def run(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(8192)
                samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2**31
                self.buffer.extend(samples)
                self.connected = True
            except socket.timeout:
                self.connected = False
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.sock.close()


# ──────────────────────────────────────────
# 실시간 분류 GUI
# ──────────────────────────────────────────
class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robotic Skin - Auto Touch Classifier")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("1100x680")

        self.buf1 = deque(maxlen=SAMPLE_RATE * 5)
        self.buf2 = deque(maxlen=SAMPLE_RATE * 5)
        self.buf3 = deque(maxlen=SAMPLE_RATE * 5)

        self.state        = STATE_LISTENING
        self.result_start = 0.0

        self.current_class = 'idle'
        self.current_conf  = 0.0
        self.probabilities = np.zeros(len(TOUCH_CLASSES))
        self.history       = []

        self.threshold = tk.DoubleVar(value=ONSET_THRESHOLD)

        print("Loading model...")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded!")
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            self.model = None

        self.rec1 = UDPReceiver(self.buf1, UDP_PORT_1)
        self.rec2 = UDPReceiver(self.buf2, UDP_PORT_2)
        self.rec3 = UDPReceiver(self.buf3, UDP_PORT_3)
        self.rec1.start()
        self.rec2.start()
        self.rec3.start()

        self._build_ui()
        self._update_loop()

    def _build_ui(self):
        # 상단
        top = tk.Frame(self.root, bg='#0f3460', pady=12)
        top.pack(fill='x', padx=10, pady=(10, 0))

        # 결과
        result_frame = tk.Frame(top, bg='#0f3460')
        result_frame.pack(side='left', padx=20)

        self.class_label = tk.Label(
            result_frame, text="IDLE",
            font=('Helvetica', 40, 'bold'),
            bg='#0f3460', fg='#95a5a6')
        self.class_label.pack(anchor='w')

        self.desc_label = tk.Label(
            result_frame, text="아무것도 안함",
            font=('Helvetica', 13),
            bg='#0f3460', fg='#aaa')
        self.desc_label.pack(anchor='w')

        self.conf_label = tk.Label(
            result_frame, text="Confidence: 0.0%",
            font=('Helvetica', 11),
            bg='#0f3460', fg='#aaa')
        self.conf_label.pack(anchor='w')

        self.conn_label = tk.Label(
            result_frame, text="ESP32: Waiting...",
            font=('Helvetica', 10),
            bg='#0f3460', fg='#f39c12')
        self.conn_label.pack(anchor='w', pady=(3, 0))

        # 가운데: 상태 + RMS + 슬라이더
        center_frame = tk.Frame(top, bg='#0f3460')
        center_frame.pack(side='left', expand=True)

        self.state_label = tk.Label(
            center_frame, text="LISTENING...",
            font=('Helvetica', 20, 'bold'),
            bg='#0f3460', fg='#2ecc71')
        self.state_label.pack()

        self.rms_label = tk.Label(
            center_frame, text="RMS: 0.0000",
            font=('Courier', 13),
            bg='#0f3460', fg='#3498db')
        self.rms_label.pack(pady=(5, 0))

        tk.Label(center_frame, text="임계값 조정",
                 font=('Helvetica', 9), bg='#0f3460', fg='#aaa').pack(pady=(8, 0))
        tk.Scale(
            center_frame,
            from_=0.001, to=0.020,
            resolution=0.001,
            orient='horizontal',
            variable=self.threshold,
            bg='#0f3460', fg='white',
            highlightthickness=0,
            length=200
        ).pack()

        # 히스토리
        hist_frame = tk.Frame(top, bg='#0f3460')
        hist_frame.pack(side='right', padx=20)
        tk.Label(hist_frame, text="Recent",
                 font=('Helvetica', 10), bg='#0f3460', fg='#aaa').pack()
        self.hist_label = tk.Label(
            hist_frame, text="",
            font=('Helvetica', 11),
            bg='#0f3460', fg='white',
            wraplength=180, justify='center')
        self.hist_label.pack()

        # 파형 + 확률 바
        mid = tk.Frame(self.root, bg='#1a1a2e')
        mid.pack(fill='both', expand=True, padx=10, pady=8)

        plt.style.use('dark_background')
        self.fig, axes = plt.subplots(
            1, 2, figsize=(10, 3.5),
            gridspec_kw={'width_ratios': [2, 1]},
            facecolor='#1a1a2e'
        )
        self.ax_wave, self.ax_prob = axes
        self.fig.tight_layout(pad=2)

        self.ax_wave.set_facecolor('#0d1117')
        self.ax_wave.set_title('MIC 1 Waveform', color='#e0e0e0', fontsize=11)
        self.ax_wave.set_ylim(-0.05, 0.05)
        self.ax_wave.set_xlabel('Time (ms)', color='#aaa', fontsize=9)
        self.ax_wave.tick_params(colors='#aaa')
        self.ax_wave.grid(True, alpha=0.2)

        self.thr_line_pos = self.ax_wave.axhline(
            ONSET_THRESHOLD, color='#f39c12', lw=1.0, linestyle='--', alpha=0.8)
        self.thr_line_neg = self.ax_wave.axhline(
            -ONSET_THRESHOLD, color='#f39c12', lw=1.0, linestyle='--', alpha=0.8)

        x = np.linspace(0, 1000, DISPLAY_SIZE)
        self.line_wave, = self.ax_wave.plot(
            x, np.zeros(DISPLAY_SIZE), color='#00d4ff', lw=0.7)

        self.ax_prob.set_facecolor('#0d1117')
        self.ax_prob.set_title('Class Probability', color='#e0e0e0', fontsize=11)
        self.ax_prob.set_xlim(0, 1)
        self.ax_prob.tick_params(colors='#aaa', labelsize=10)
        self.ax_prob.grid(True, axis='x', alpha=0.2)

        bar_colors = [COLORS[c] for c in TOUCH_CLASSES]
        self.bars = self.ax_prob.barh(
            [LABELS[c] for c in TOUCH_CLASSES],
            np.zeros(len(TOUCH_CLASSES)),
            color=bar_colors, alpha=0.4, height=0.5)
        self.ax_prob.set_yticks(range(len(TOUCH_CLASSES)))
        self.ax_prob.set_yticklabels(
            [LABELS[c] for c in TOUCH_CLASSES],
            color='#ccc', fontsize=11)

        canvas = FigureCanvasTkAgg(self.fig, master=mid)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas = canvas
        self.root.focus_set()

    def _infer(self, b1, b2):
        if self.model is None:
            return

        # 현재 버퍼 최근 512ms 그대로 사용
        w1 = np.array(b1[-WINDOW_SIZE:])
        w2 = np.array(b2[-WINDOW_SIZE:]) if len(b2) >= WINDOW_SIZE else w1

        data = np.stack([w1, w2])
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak

        X_int, X_spec = make_feature_maps(data)
        pred = self.model.predict([X_int, X_spec], verbose=0)[0]

        self.probabilities = pred
        idx = np.argmax(pred)
        self.current_class = TOUCH_CLASSES[idx]
        self.current_conf  = float(pred[idx])

        if self.current_conf > 0.5 and self.current_class != 'idle':
            self.history.append(LABELS[self.current_class])
            if len(self.history) > 6:
                self.history.pop(0)

        print(f"Result: {self.current_class} ({self.current_conf*100:.1f}%)")

    def _update_loop(self):
        try:
            b1  = list(self.buf1)
            b2  = list(self.buf2)
            now = time.time()
            thr = self.threshold.get()

            # 연결 상태
            if self.rec1.connected or self.rec2.connected:
                self.conn_label.config(text="ESP32: Connected", fg='#2ecc71')
            else:
                self.conn_label.config(text="ESP32: Waiting...", fg='#f39c12')

            if len(b1) >= DISPLAY_SIZE:
                d1  = np.array(b1[-DISPLAY_SIZE:])
                rms = float(np.sqrt(np.mean(d1**2)))

                self.line_wave.set_ydata(d1)
                peak = max(abs(d1.max()), abs(d1.min()), 0.005)
                self.ax_wave.set_ylim(-peak * 1.3, peak * 1.3)
                self.thr_line_pos.set_ydata([thr, thr])
                self.thr_line_neg.set_ydata([-thr, -thr])
                self.rms_label.config(text=f"RMS: {rms:.4f}")

                # 상태 머신
                if self.state == STATE_LISTENING:
                    self.state_label.config(text="LISTENING...", fg='#2ecc71')

                    # 신호 감지 → 바로 추론
                    if rms > thr and len(b1) >= WINDOW_SIZE:
                        self._infer(b1, b2)
                        self.state        = STATE_RESULT
                        self.result_start = now

                    else:
                        # 신호 없음 → idle
                        self.current_class = 'idle'
                        self.current_conf  = 1.0
                        self.probabilities = np.zeros(len(TOUCH_CLASSES))
                        self.probabilities[0] = 1.0

                elif self.state == STATE_RESULT:
                    elapsed = now - self.result_start
                    remain  = max(0.0, COOLDOWN - elapsed)
                    self.state_label.config(
                        text=f"RESULT ({remain:.1f}s)", fg='#f39c12')

                    if elapsed >= COOLDOWN:
                        self.state = STATE_LISTENING

                # UI 갱신
                cls   = self.current_class
                color = COLORS[cls]
                self.class_label.config(text=LABELS[cls], fg=color)
                self.desc_label.config(text=DESCRIPTIONS[cls], fg=color)
                self.conf_label.config(
                    text=f"Confidence: {self.current_conf*100:.1f}%")
                self.hist_label.config(text=' → '.join(self.history))

                alpha = 0.9 if self.state == STATE_RESULT else 0.3
                for bar, prob in zip(self.bars, self.probabilities):
                    bar.set_width(prob)
                    bar.set_alpha(alpha)

                self.canvas.draw_idle()

        except Exception as e:
            print(f"[Error] {e}")

        self.root.after(100, self._update_loop)

    def on_close(self):
        self.rec1.stop()
        self.rec2.stop()
        self.rec3.stop()
        self.root.destroy()


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  Robotic Skin - Auto Touch Classifier")
    print("=" * 50)
    print(f"  Classes  : idle / scratch / tap")
    print(f"  Mode     : Auto (buffer-based)")
    print(f"  Threshold: {ONSET_THRESHOLD}")
    print("=" * 50)

    root = tk.Tk()
    app  = ClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
