"""
Robotic Skin - 터치 데이터 수집 GUI (자동 감지 방식)
=====================================================
학습이랑 실시간 테스트 조건을 완전히 동일하게!

동작:
    신호 감지 → 자동으로 현재 버퍼 512ms 저장
    (S 누를 필요 없음)

클래스: idle / scratch / tap
"""

import socket
import threading
import numpy as np
import os
import time
import platform
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from collections import deque

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
TARGET_SAMPLES  = 100
ONSET_THRESHOLD = 0.001  # 신호 감지 임계값 (scratch 감지를 위해 낮춤)
COOLDOWN        = 1.0    # 저장 후 대기 (중복 저장 방지)

TOUCH_CLASSES = [
    ('idle',    'Idle    (아무것도 X)',  '#95a5a6'),
    ('scratch', 'Scratch (손톱 긁기)',   '#e74c3c'),
    ('tap',     'Tap     (손가락 치기)', '#2ecc71'),
]

DATA_DIR = "touch_dataset"

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
# 메인 GUI
# ──────────────────────────────────────────
class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robotic Skin - Auto Data Collector")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("1300x780")

        self.buf1 = deque(maxlen=SAMPLE_RATE * 5)
        self.buf2 = deque(maxlen=SAMPLE_RATE * 5)
        self.buf3 = deque(maxlen=SAMPLE_RATE * 5)

        self.current_class   = tk.StringVar(value='idle')
        self.counts          = self._load_counts()
        self.last_saved_time = 0
        self.is_collecting   = tk.BooleanVar(value=False)
        self.is_collecting   = tk.BooleanVar(value=False)  # 수집 ON/OFF
        self.threshold       = tk.DoubleVar(value=ONSET_THRESHOLD)

        self.rec1 = UDPReceiver(self.buf1, UDP_PORT_1)
        self.rec2 = UDPReceiver(self.buf2, UDP_PORT_2)
        self.rec3 = UDPReceiver(self.buf3, UDP_PORT_3)
        self.rec1.start()
        self.rec2.start()
        self.rec3.start()

        self._build_ui()
        self._update_loop()

    def _load_counts(self):
        counts = {}
        for cls, _, _ in TOUCH_CLASSES:
            path = os.path.join(DATA_DIR, cls)
            os.makedirs(path, exist_ok=True)
            counts[cls] = len([f for f in os.listdir(path) if f.endswith('.npy')])
        return counts

    def _build_ui(self):
        # 왼쪽 패널
        left = tk.Frame(self.root, bg='#16213e', width=240)
        left.pack(side='left', fill='y', padx=(10, 0), pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="터치 클래스 선택",
                 font=('Helvetica', 13, 'bold'),
                 bg='#16213e', fg='#e0e0e0').pack(pady=(15, 10))

        self.class_buttons = {}
        for cls, label, color in TOUCH_CLASSES:
            frame = tk.Frame(left, bg='#16213e')
            frame.pack(fill='x', padx=8, pady=4)

            btn = tk.Radiobutton(
                frame, text=label,
                variable=self.current_class, value=cls,
                font=('Helvetica', 10),
                bg='#16213e', fg='#e0e0e0',
                selectcolor='#0f3460', activebackground='#16213e',
                activeforeground=color, indicatoron=False,
                relief='flat', cursor='hand2', padx=8, pady=7,
                width=20, anchor='w',
                command=self._on_class_change
            )
            btn.pack(side='left', fill='x', expand=True)

            cnt_lbl = tk.Label(frame,
                               text=f"{self.counts[cls]}/{TARGET_SAMPLES}",
                               font=('Helvetica', 9),
                               bg='#16213e', fg=color, width=7)
            cnt_lbl.pack(side='right')
            self.class_buttons[cls] = (btn, cnt_lbl, color)

        tk.Frame(left, bg='#2d3561', height=1).pack(fill='x', padx=8, pady=12)

        tk.Label(left, text="전체 진행률",
                 font=('Helvetica', 10), bg='#16213e', fg='#aaa').pack()
        self.total_label = tk.Label(left, text="",
                                     font=('Helvetica', 14, 'bold'),
                                     bg='#16213e', fg='#3498db')
        self.total_label.pack(pady=6)
        self._update_total_label()

        tk.Frame(left, bg='#2d3561', height=1).pack(fill='x', padx=8, pady=8)

        # 임계값 슬라이더
        tk.Label(left, text="감지 임계값",
                 font=('Helvetica', 10), bg='#16213e', fg='#aaa').pack()
        tk.Scale(
            left,
            from_=0.001, to=0.020,
            resolution=0.001,
            orient='horizontal',
            variable=self.threshold,
            bg='#16213e', fg='white',
            highlightthickness=0,
            length=200
        ).pack(pady=(0, 10))

        # 수집 가이드
        tk.Frame(left, bg='#2d3561', height=1).pack(fill='x', padx=8, pady=4)
        guide = [
            "[ 수집 가이드 ]",
            "",
            "1. 클래스 선택",
            "2. [수집 시작] 버튼",
            "3. 터치하면 자동 저장!",
            "4. 200개 채우면 완료",
            "",
            "Idle: 그냥 가만히",
            "  → 노이즈만 있을때",
            "  → [수집시작] 누르고",
            "     그냥 놔두기",
            "",
            "Scratch: 긁는 순간",
            "  → 긁으면 자동 저장",
            "",
            "Tap: 치는 순간",
            "  → 탁! 치면 자동 저장",
        ]
        for g in guide:
            tk.Label(left, text=g,
                     font=('Helvetica', 8),
                     bg='#16213e', fg='#7f8c8d', anchor='w').pack(fill='x', padx=12)

        # 오른쪽 메인
        right = tk.Frame(self.root, bg='#1a1a2e')
        right.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        # 상태바
        status_frame = tk.Frame(right, bg='#0f3460', pady=8)
        status_frame.pack(fill='x')

        self.status_label = tk.Label(
            status_frame, text="ESP32 연결 대기 중...",
            font=('Helvetica', 11), bg='#0f3460', fg='#f39c12')
        self.status_label.pack(side='left', padx=15)

        self.rms_label = tk.Label(
            status_frame, text="RMS: 0.0000",
            font=('Courier', 10), bg='#0f3460', fg='#2ecc71')
        self.rms_label.pack(side='right', padx=15)

        # matplotlib
        plt.style.use('dark_background')
        self.fig, axes = plt.subplots(
            4, 1, figsize=(8, 7),
            gridspec_kw={'height_ratios': [2, 2, 2, 1.5]},
            facecolor='#1a1a2e'
        )
        self.ax_mic1, self.ax_mic2, self.ax_mic3, self.ax_spec = axes
        self.fig.tight_layout(pad=2.0)

        x = np.linspace(0, 1000, DISPLAY_SIZE)

        # MIC1
        self.ax_mic1.set_facecolor('#0d1117')
        self.ax_mic1.set_title('MIC 1 Waveform', color='#00d4ff', fontsize=10)
        self.ax_mic1.set_ylim(-0.05, 0.05)
        self.ax_mic1.set_ylabel('Amplitude', color='#aaa', fontsize=8)
        self.ax_mic1.tick_params(colors='#aaa', labelsize=7)
        self.ax_mic1.grid(True, alpha=0.2)
        self.thr_pos = self.ax_mic1.axhline(
            ONSET_THRESHOLD, color='#f39c12', lw=1.0, linestyle='--', alpha=0.8)
        self.thr_neg = self.ax_mic1.axhline(
            -ONSET_THRESHOLD, color='#f39c12', lw=1.0, linestyle='--', alpha=0.8)
        self.line1, = self.ax_mic1.plot(x, np.zeros(DISPLAY_SIZE), color='#00d4ff', lw=0.7)

        # 저장 플래시 효과
        self.save_flash = self.ax_mic1.axvspan(
            0, 1000, alpha=0.0, color='#2ecc71')

        # MIC2
        self.ax_mic2.set_facecolor('#0d1117')
        self.ax_mic2.set_title('MIC 2 Waveform', color='#ff6b9d', fontsize=10)
        self.ax_mic2.set_ylim(-0.05, 0.05)
        self.ax_mic2.set_ylabel('Amplitude', color='#aaa', fontsize=8)
        self.ax_mic2.tick_params(colors='#aaa', labelsize=7)
        self.ax_mic2.grid(True, alpha=0.2)
        self.line2, = self.ax_mic2.plot(x, np.zeros(DISPLAY_SIZE), color='#ff6b9d', lw=0.7)

        # MIC3
        self.ax_mic3.set_facecolor('#0d1117')
        self.ax_mic3.set_title('MIC 3 Waveform', color='#f39c12', fontsize=10)
        self.ax_mic3.set_ylim(-0.05, 0.05)
        self.ax_mic3.set_xlabel('Time (ms)', color='#aaa', fontsize=8)
        self.ax_mic3.set_ylabel('Amplitude', color='#aaa', fontsize=8)
        self.ax_mic3.tick_params(colors='#aaa', labelsize=7)
        self.ax_mic3.grid(True, alpha=0.2)
        self.line3, = self.ax_mic3.plot(x, np.zeros(DISPLAY_SIZE), color='#f39c12', lw=0.7)

        # 스펙트로그램
        self.ax_spec.set_facecolor('#0d1117')
        self.ax_spec.set_title('Spectrogram - MIC1', color='#e0e0e0', fontsize=10)
        self.ax_spec.set_xlabel('Time (ms)', color='#aaa', fontsize=8)
        self.ax_spec.set_ylabel('Freq (Hz)', color='#aaa', fontsize=8)
        self.ax_spec.tick_params(colors='#aaa', labelsize=7)
        self.spec_img = self.ax_spec.imshow(
            np.zeros((64, 40)), aspect='auto', origin='lower',
            extent=[0, 512, 0, 5000], cmap='inferno', vmin=-5, vmax=5)

        canvas = FigureCanvasTkAgg(self.fig, master=right)
        canvas.get_tk_widget().pack(fill='both', expand=True, pady=(5, 0))
        self.canvas = canvas

        # 하단 버튼 (수동 저장)
        btn_frame = tk.Frame(right, bg='#1a1a2e')
        btn_frame.pack(fill='x', pady=(8, 0))

        self.collect_btn = tk.Button(
            btn_frame, text="▶ 자동수집 시작",
            font=('Helvetica', 12, 'bold'), bg='#27ae60', fg='white',
            activebackground='#2ecc71', relief='flat', cursor='hand2',
            padx=20, pady=10, command=self._toggle_collect)
        self.collect_btn.pack(side='left', padx=(0, 6))

        self.save_btn = tk.Button(
            btn_frame, text="💾 수동저장 [S]",
            font=('Helvetica', 12, 'bold'), bg='#2980b9', fg='white',
            activebackground='#3498db', relief='flat', cursor='hand2',
            padx=20, pady=10, command=self._manual_save)
        self.save_btn.pack(side='left', padx=(0, 10))

        self.save_status = tk.Label(
            btn_frame,
            text="자동수집: 신호 감지 시 자동저장 | 수동: 터치 후 [S]",
            font=('Helvetica', 10),
            bg='#1a1a2e', fg='#7f8c8d')
        self.save_status.pack(side='left', padx=10)

        # 키보드 단축키
        self.root.bind('<s>', lambda e: self._manual_save())
        self.root.bind('<S>', lambda e: self._manual_save())
        self.root.focus_set()

        self._on_class_change()

    def _toggle_collect(self):
        """자동 수집 토글"""
        if self.is_collecting.get():
            self.is_collecting.set(False)
            self.collect_btn.config(text="▶ 자동수집 시작", bg='#27ae60')
            self.save_status.config(text="자동수집 중지됨", fg='#e74c3c')
        else:
            self.is_collecting.set(True)
            cls = self.current_class.get()
            self.collect_btn.config(text="■ 자동수집 중지", bg='#e74c3c')
            self.save_status.config(
                text=f"[{cls}] 자동수집 중... 터치하면 자동저장!",
                fg='#2ecc71')

    def _manual_save(self):
        """
        자동 감지 저장 방식 (저번에 97.5% 달성한 방식)
        핵심: 신호 감지 → 현재 버퍼 저장
        실시간 분류도 동일하게 신호 감지 → 현재 버퍼 판단
        → 학습 조건 = 실시간 조건 → 성공!
        """
        b1 = list(self.buf1)
        b2 = list(self.buf2)
        b3 = list(self.buf3)

        if len(b1) < WINDOW_SIZE:
            self.save_status.config(text="버퍼 부족! 잠시 후 시도하세요.", fg='#e74c3c')
            return

        cls = self.current_class.get()
        now = time.time()
        if now - self.last_saved_time < COOLDOWN:
            return

        # RMS 확인
        arr1 = np.array(b1[-WINDOW_SIZE:])
        arr1 = arr1 - np.mean(arr1)
        rms1 = float(np.sqrt(np.mean(arr1**2)))

        # idle이 아닌데 신호가 없으면 경고
        if cls != 'idle' and rms1 < ONSET_THRESHOLD:
            self.save_status.config(
                text=f"❌ 신호 없음! {cls} 터치 후 저장하세요.",
                fg='#e74c3c')
            return

        # idle인데 신호가 있으면 경고
        if cls == 'idle' and rms1 > ONSET_THRESHOLD:
            self.save_status.config(
                text="❌ 신호 감지됨! 조용한 상태에서 저장하세요.",
                fg='#e74c3c')
            return

        w1   = np.array(b1[-WINDOW_SIZE:])
        w2   = np.array(b2[-WINDOW_SIZE:]) if len(b2) >= WINDOW_SIZE else w1
        w3   = np.array(b3[-WINDOW_SIZE:]) if len(b3) >= WINDOW_SIZE else w1
        data = np.stack([w1, w2, w3])

        idx  = self.counts[cls]
        path = os.path.join(DATA_DIR, cls, f"{idx:04d}.npy")
        np.save(path, data)
        self.counts[cls] += 1
        self.last_saved_time = now

        # UI 갱신
        _, lbl, color = self.class_buttons[cls]
        lbl.config(text=f"{self.counts[cls]}/{TARGET_SAMPLES}")
        self._update_total_label()

        self.save_status.config(
            text=f"✅ [{cls}] {self.counts[cls]}/{TARGET_SAMPLES} 저장 완료!",
            fg='#2ecc71')

        # 저장 플래시
        self.save_flash.set_alpha(0.2)
        self.root.after(200, lambda: self.save_flash.set_alpha(0.0))

        print(f"[Saved] {cls} #{self.counts[cls]-1:04d}  RMS={rms1:.5f}")

        if self.counts[cls] >= TARGET_SAMPLES:
            messagebox.showinfo("완료!", f"[{cls}] {TARGET_SAMPLES}개 수집 완료!")

    def _toggle_collect(self):
        if self.is_collecting.get():
            self.is_collecting.set(False)
            self.collect_btn.config(text="▶ 수집 시작", bg='#27ae60')
            self.save_status.config(text="수집 중지됨", fg='#e74c3c')
        else:
            self.is_collecting.set(True)
            self.collect_btn.config(text="■ 수집 중지", bg='#e74c3c')
            cls = self.current_class.get()
            self.save_status.config(
                text=f"[{cls}] 수집 중... 터치하면 자동 저장!",
                fg='#2ecc71')

    def _on_class_change(self):
        cls = self.current_class.get()
        for c, (btn, lbl, color) in self.class_buttons.items():
            if c == cls:
                btn.config(fg=color, font=('Helvetica', 10, 'bold'))
            else:
                btn.config(fg='#e0e0e0', font=('Helvetica', 10))

        # 수집 중이면 자동으로 중지
        if self.is_collecting.get():
            self.is_collecting.set(False)
            self.collect_btn.config(text="▶ 자동수집 시작", bg='#27ae60')

        self.save_status.config(
            text="클래스 선택 후 [수집 시작] 누르고 터치하세요!", fg='#7f8c8d')

    def _auto_save(self, b1, b2):
        """신호 감지 시 자동 저장 - 현재 버퍼 그대로 저장"""
        cls = self.current_class.get()
        now = time.time()

        # 쿨다운 체크
        if now - self.last_saved_time < COOLDOWN:
            return

        # idle 클래스는 신호가 없을 때만 저장
        if cls == 'idle':
            return  # idle은 아래 _auto_save_idle에서 처리

        # 현재 버퍼 최근 512ms 그대로 저장
        b3 = list(self.buf3)
        w1 = np.array(b1[-WINDOW_SIZE:])
        w2 = np.array(b2[-WINDOW_SIZE:]) if len(b2) >= WINDOW_SIZE else w1
        w3 = np.array(b3[-WINDOW_SIZE:]) if len(b3) >= WINDOW_SIZE else w1
        data = np.stack([w1, w2, w3])

        idx  = self.counts[cls]
        path = os.path.join(DATA_DIR, cls, f"{idx:04d}.npy")
        np.save(path, data)
        self.counts[cls] += 1
        self.last_saved_time = now

        # UI 갱신
        _, lbl, color = self.class_buttons[cls]
        lbl.config(text=f"{self.counts[cls]}/{TARGET_SAMPLES}")
        self._update_total_label()

        self.save_status.config(
            text=f"✅ [{cls}] {self.counts[cls]}/{TARGET_SAMPLES} 자동 저장!",
            fg='#2ecc71')

        # 저장 플래시
        self.save_flash.set_alpha(0.2)
        self.root.after(200, lambda: self.save_flash.set_alpha(0.0))

        if self.counts[cls] >= TARGET_SAMPLES:
            self.is_collecting.set(False)
            self.collect_btn.config(text="▶ 수집 시작", bg='#27ae60')
            messagebox.showinfo("완료!", f"[{cls}] {TARGET_SAMPLES}개 수집 완료!")

    def _auto_save_idle(self, b1, b2):
        """idle: 신호 없을 때 자동 저장"""
        cls = self.current_class.get()
        if cls != 'idle':
            return

        now = time.time()
        if now - self.last_saved_time < COOLDOWN:
            return

        b3   = list(self.buf3)
        w1   = np.array(b1[-WINDOW_SIZE:])
        w2   = np.array(b2[-WINDOW_SIZE:]) if len(b2) >= WINDOW_SIZE else w1
        w3   = np.array(b3[-WINDOW_SIZE:]) if len(b3) >= WINDOW_SIZE else w1
        data = np.stack([w1, w2, w3])

        idx  = self.counts[cls]
        path = os.path.join(DATA_DIR, cls, f"{idx:04d}.npy")
        np.save(path, data)
        self.counts[cls] += 1
        self.last_saved_time = now

        _, lbl, color = self.class_buttons[cls]
        lbl.config(text=f"{self.counts[cls]}/{TARGET_SAMPLES}")
        self._update_total_label()

        self.save_status.config(
            text=f"✅ [idle] {self.counts[cls]}/{TARGET_SAMPLES} 자동 저장!",
            fg='#2ecc71')

        if self.counts[cls] >= TARGET_SAMPLES:
            self.is_collecting.set(False)
            self.collect_btn.config(text="▶ 수집 시작", bg='#27ae60')
            messagebox.showinfo("완료!", f"[idle] {TARGET_SAMPLES}개 수집 완료!")

    def _update_total_label(self):
        total   = sum(self.counts.values())
        maximum = TARGET_SAMPLES * len(TOUCH_CLASSES)
        self.total_label.config(text=f"{total} / {maximum}")

    def _update_loop(self):
        try:
            b1  = list(self.buf1)
            b2  = list(self.buf2)
            b3  = list(self.buf3)
            thr = self.threshold.get()

            # 연결 상태
            if self.rec1.connected or self.rec2.connected or self.rec3.connected:
                self.status_label.config(text="ESP32 연결됨", fg='#2ecc71')
            else:
                self.status_label.config(text="ESP32 연결 대기 중...", fg='#f39c12')

            rms1 = rms2 = 0.0

            if len(b1) >= DISPLAY_SIZE:
                d1   = np.array(b1[-DISPLAY_SIZE:])
                rms1 = float(np.sqrt(np.mean(d1**2)))
                self.line1.set_ydata(d1)
                peak = max(abs(d1.max()), abs(d1.min()), 0.005)
                self.ax_mic1.set_ylim(-peak * 1.3, peak * 1.3)

                # 임계값 선 업데이트
                self.thr_pos.set_ydata([thr, thr])
                self.thr_neg.set_ydata([-thr, -thr])

                # 자동 감지 저장
                if self.is_collecting.get() and len(b1) >= WINDOW_SIZE:
                    cls = self.current_class.get()
                    if cls == 'idle':
                        if rms1 < thr:
                            self._manual_save()
                    elif cls == 'tap':
                        if rms1 > thr:
                            self._manual_save()
                    # scratch는 자동 감지 안 함 → 수동 [S] 사용

            if len(b2) >= DISPLAY_SIZE:
                d2   = np.array(b2[-DISPLAY_SIZE:])
                rms2 = float(np.sqrt(np.mean(d2**2)))
                self.line2.set_ydata(d2)
                peak = max(abs(d2.max()), abs(d2.min()), 0.005)
                self.ax_mic2.set_ylim(-peak * 1.3, peak * 1.3)

            if len(b3) >= DISPLAY_SIZE:
                d3   = np.array(b3[-DISPLAY_SIZE:])
                self.line3.set_ydata(d3)
                peak = max(abs(d3.max()), abs(d3.min()), 0.005)
                self.ax_mic3.set_ylim(-peak * 1.3, peak * 1.3)

            self.rms_label.config(text=f"RMS: {rms1:.4f}")

            if len(b1) >= WINDOW_SIZE:
                _, _, Sxx = signal.spectrogram(
                    np.array(b1[-WINDOW_SIZE:]),
                    fs=SAMPLE_RATE, nperseg=128, noverlap=64)
                log_Sxx = np.log1p(Sxx[:64])
                self.spec_img.set_data(log_Sxx)
                self.spec_img.set_clim(log_Sxx.min(), log_Sxx.max() + 1e-6)

            self.canvas.draw_idle()

        except Exception as e:
            print(f"[Error] {e}")

        self.root.after(80, self._update_loop)

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
    print("  Robotic Skin - Auto Data Collector")
    print("=" * 50)
    print(f"  Classes  : idle / scratch / tap")
    print(f"  Method   : Auto detection & save")
    print(f"  Threshold: {ONSET_THRESHOLD}")
    print("=" * 50)

    root = tk.Tk()
    app  = DataCollectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
