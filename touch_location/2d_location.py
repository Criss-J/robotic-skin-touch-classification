"""
Robotic Skin - 2D Touch Location Estimator (Triangle Array)
============================================================
마이크 3개 삼각형 배열로 2D 위치 추정

마이크 배치:
  MIC1: (  0,   0) mm  → 왼쪽 아래  (ESP32 #1, UDP 12345)
  MIC2: (150,   0) mm  → 오른쪽 아래 (ESP32 #1, UDP 12346)
  MIC3: ( 75, 130) mm  → 위쪽       (ESP32 #2, UDP 12347)

실행:
  python 6_2d_location.py
"""

import socket
import threading
import numpy as np
import time
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit, minimize
from collections import deque

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
UDP_IP      = "0.0.0.0"
UDP_PORT_1  = 12345   # MIC1
UDP_PORT_2  = 12346   # MIC2
UDP_PORT_3  = 12347   # MIC3
SAMPLE_RATE = 10000
WINDOW_SIZE = 1000    # 100ms

# 마이크 위치 (mm) - 실제 측정값
# 가로: 170mm, 높이: 115mm
MIC_POSITIONS = np.array([
    [170,   0],   # MIC1 오른쪽 아래  (ESP32 #1, UDP 12345)
    [ 85, 115],   # MIC2 위쪽 꼭짓점 (ESP32 #1, UDP 12346)
    [  0,   0],   # MIC3 왼쪽 아래   (ESP32 #2, UDP 12347)
])

# 감쇠 파라미터 초기값
a_param = 1.5
b_param = 0.5

# 캘리브레이션 포인트 (삼각형 내부, mm)
CALIB_POSITIONS = [
    ( 85,  20),   # 1번: 중앙 아래
    ( 35,  55),   # 2번: 왼쪽 중간
    (135,  55),   # 3번: 오른쪽 중간
    ( 85,  85),   # 4번: 중앙 위
    ( 85,  50),   # 5번: 정중앙
]
CALIB_SAMPLES = 10

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'


# ──────────────────────────────────────────
# UDP 수신
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

    def stop(self):
        self.running = False
        self.sock.close()


# ──────────────────────────────────────────
# PAT 알고리즘
# ──────────────────────────────────────────
def calc_rms(data):
    arr = np.array(data, dtype=np.float32)
    arr = arr - np.mean(arr)  # DC offset 제거
    return float(np.sqrt(np.mean(arr**2)))


def find_peak_rms(buf, window=1000, stride=200):
    """최근 2초에서 가장 강한 100ms 구간 RMS 반환"""
    arr = np.array(buf[-SAMPLE_RATE*2:])
    best = 0.0
    for i in range(0, max(1, len(arr) - window), stride):
        r = calc_rms(arr[i:i+window])
        if r > best:
            best = r
    return best


def attenuation_model(r, a, b):
    """감쇠 모델: I = I_0 / (r^a + b)"""
    return 1.0 / (np.array(r)**a + b)


def calibrate(positions, rms_list, mic_positions, a_init=1.5, b_init=0.5):
    """
    캘리브레이션: a, b 파라미터 추정
    positions: 탭핑 위치 [(x1,y1), ...]
    rms_list:  각 위치에서 [rms1, rms2, rms3]
    """
    distances      = []
    rms_normalized = []

    for pos, rms in zip(positions, rms_list):
        pos = np.array(pos)
        max_rms = max(rms)
        if max_rms == 0:
            continue
        for i, r_val in enumerate(rms):
            dist = np.linalg.norm(pos - mic_positions[i])
            dist = max(dist, 1.0)
            distances.append(dist)
            rms_normalized.append(r_val / max_rms)

    distances      = np.array(distances)
    rms_normalized = np.array(rms_normalized)

    try:
        popt, _ = curve_fit(
            attenuation_model,
            distances,
            rms_normalized,
            p0=[a_init, b_init],
            bounds=([0.01, 0.001], [10.0, 5.0]),
            maxfev=10000
        )
        a, b = popt
        if a < 0.3:
            print(f"  curve_fit a={a:.3f} too low, using defaults")
            return 1.5, 0.5
        print(f"Calibration done: a={a:.3f}, b={b:.3f}")
        return float(a), float(b)
    except Exception as e:
        print(f"Calibration failed: {e}")
        return 1.5, 0.5


def estimate_position_2d(rms_values, mic_positions, a, b, grid_step=5):
    """
    2D 위치 추정 (PAT)
    삼각형 내부를 격자 탐색 → Loss 최소 위치 반환
    """
    def loss(pos):
        pos = np.array(pos)
        I0_list = []
        for i, rms in enumerate(rms_values):
            r = max(np.linalg.norm(pos - mic_positions[i]), 1.0)
            I0 = rms * (r**a + b)
            I0_list.append(I0)

        total = 0.0
        n = len(I0_list)
        for i in range(n):
            for j in range(i+1, n):
                total += (I0_list[i] - I0_list[j])**2
        return total

    # 삼각형 내부 격자 탐색
    A, B, C = mic_positions[0], mic_positions[1], mic_positions[2]
    x_min = min(A[0], B[0], C[0])
    x_max = max(A[0], B[0], C[0])
    y_min = min(A[1], B[1], C[1])
    y_max = max(A[1], B[1], C[1])

    best_loss = float('inf')
    best_pos  = np.array([(x_min+x_max)/2, (y_min+y_max)/2])

    for x in np.arange(x_min, x_max+grid_step, grid_step):
        for y in np.arange(y_min, y_max+grid_step, grid_step):
            p = np.array([x, y])
            if is_inside_triangle(p, A, B, C):
                l = loss(p)
                if l < best_loss:
                    best_loss = l
                    best_pos  = p.copy()

    # Nelder-Mead로 정밀화
    result = minimize(loss, best_pos, method='Nelder-Mead',
                      options={'xatol': 0.5, 'fatol': 1e-8, 'maxiter': 500})
    return result.x


def is_inside_triangle(p, A, B, C):
    """점 p가 삼각형 ABC 내부에 있는지 확인"""
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1 = sign(p, A, B)
    d2 = sign(p, B, C)
    d3 = sign(p, C, A)
    has_neg = (d1<0) or (d2<0) or (d3<0)
    has_pos = (d1>0) or (d2>0) or (d3>0)
    return not (has_neg and has_pos)


# ──────────────────────────────────────────
# 메인 GUI
# ──────────────────────────────────────────
class LocationApp2D:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Touch Location Estimator (Triangle Array)")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("1100x750")

        self.buf1 = deque(maxlen=SAMPLE_RATE * 3)
        self.buf2 = deque(maxlen=SAMPLE_RATE * 3)
        self.buf3 = deque(maxlen=SAMPLE_RATE * 3)

        # PAT 파라미터
        self.a = a_param
        self.b = b_param
        self.calibrated = False

        # 노이즈
        self.noise1 = 0.0
        self.noise2 = 0.0
        self.noise3 = 0.0
        self.noise_measured = False

        # 캘리브레이션
        self.calib_idx   = 0
        self.calib_count = 0
        self.calib_pos   = []
        self.calib_rms   = []
        self.calib_tmp   = []
        self.in_calib    = False

        # 위치
        self.estimated_pos = np.array([85.0, 50.0])
        self.last_update   = 0

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
        top = tk.Frame(self.root, bg='#0f3460', pady=8)
        top.pack(fill='x', padx=10, pady=(10,0))

        self.conn_label = tk.Label(top, text="ESP32: Waiting...",
                                    font=('Helvetica', 10),
                                    bg='#0f3460', fg='#f39c12')
        self.conn_label.pack(side='left', padx=10)

        self.calib_label = tk.Label(top, text="Calibration needed",
                                     font=('Helvetica', 10, 'bold'),
                                     bg='#0f3460', fg='#e74c3c')
        self.calib_label.pack(side='left', padx=10)

        self.noise_label = tk.Label(top, text="Noise: not measured",
                                     font=('Courier', 9),
                                     bg='#0f3460', fg='#aaa')
        self.noise_label.pack(side='left', padx=10)

        self.rms_label = tk.Label(top,
                                   text="RMS1:0.00000  RMS2:0.00000  RMS3:0.00000",
                                   font=('Courier', 9),
                                   bg='#0f3460', fg='#3498db')
        self.rms_label.pack(side='right', padx=10)

        # 메인
        main = tk.Frame(self.root, bg='#1a1a2e')
        main.pack(fill='both', expand=True, padx=10, pady=8)

        # 왼쪽 패널
        left = tk.Frame(main, bg='#16213e', width=220)
        left.pack(side='left', fill='y', padx=(0,8))
        left.pack_propagate(False)

        tk.Label(left, text="Controls",
                 font=('Helvetica', 12, 'bold'),
                 bg='#16213e', fg='white').pack(pady=(12,8))

        # Noise
        s1 = tk.LabelFrame(left, text="STEP 1. Noise",
                            font=('Helvetica', 9, 'bold'),
                            bg='#16213e', fg='#f39c12', padx=6, pady=6)
        s1.pack(fill='x', padx=8, pady=3)
        tk.Button(s1, text="Measure Noise (3s)",
                  font=('Helvetica', 9, 'bold'),
                  bg='#8e44ad', fg='white', relief='flat',
                  cursor='hand2', pady=4,
                  command=self._measure_noise).pack(fill='x')
        self.noise_lbl = tk.Label(s1, text="Not measured",
                                   font=('Courier', 8),
                                   bg='#16213e', fg='#aaa')
        self.noise_lbl.pack()

        # Calibration
        s2 = tk.LabelFrame(left, text="STEP 2. Calibration",
                            font=('Helvetica', 9, 'bold'),
                            bg='#16213e', fg='#3498db', padx=6, pady=6)
        s2.pack(fill='x', padx=8, pady=3)
        self.calib_pos_lbl = tk.Label(s2,
                                       text=f"Tap at {CALIB_POSITIONS[0]} → [S]",
                                       font=('Helvetica', 8),
                                       bg='#16213e', fg='#aaa',
                                       wraplength=180)
        self.calib_pos_lbl.pack()
        tk.Button(s2, text="Start Calibration",
                  font=('Helvetica', 9, 'bold'),
                  bg='#e67e22', fg='white', relief='flat',
                  cursor='hand2', pady=4,
                  command=self._start_calib).pack(fill='x', pady=(4,2))
        self.save_btn = tk.Button(s2, text="Save [S]",
                                   font=('Helvetica', 9, 'bold'),
                                   bg='#27ae60', fg='white', relief='flat',
                                   cursor='hand2', pady=4,
                                   state='disabled',
                                   command=self._save_sample)
        self.save_btn.pack(fill='x')
        self.calib_status = tk.Label(s2, text="0 / 5 done",
                                      font=('Courier', 8),
                                      bg='#16213e', fg='#aaa')
        self.calib_status.pack()

        self.param_label = tk.Label(left,
                                     text=f"a={self.a:.3f}, b={self.b:.3f}",
                                     font=('Courier', 9),
                                     bg='#16213e', fg='#aaa')
        self.param_label.pack(pady=5)

        # 마이크 위치 표시
        mic_frame = tk.LabelFrame(left, text="Mic Positions (mm)",
                                   font=('Helvetica', 9, 'bold'),
                                   bg='#16213e', fg='#2ecc71', padx=6, pady=6)
        mic_frame.pack(fill='x', padx=8, pady=3)
        for i, (pos, label) in enumerate(zip(MIC_POSITIONS,
                                ['MIC1: (170, 0)', 'MIC2: ( 85,115)', 'MIC3: (  0,  0)'])):
            tk.Label(mic_frame,
                     text=label,
                     font=('Courier', 9),
                     bg='#16213e', fg='#2ecc71').pack(anchor='w')

        self.root.bind('<s>', lambda e: self._save_sample())
        self.root.bind('<S>', lambda e: self._save_sample())
        self.root.focus_set()

        # 오른쪽 그래프
        right = tk.Frame(main, bg='#1a1a2e')
        right.pack(side='left', fill='both', expand=True)

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(9, 6), facecolor='#1a1a2e')
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.4, wspace=0.3)

        self.ax_2d  = self.fig.add_subplot(gs[:, 0])  # 왼쪽 전체: 2D 위치
        self.ax_rms  = self.fig.add_subplot(gs[0, 1])  # 오른쪽 위: RMS 바
        self.ax_wave = self.fig.add_subplot(gs[1, 1])  # 오른쪽 아래: 파형

        # 2D 위치 표시
        self._setup_2d_plot()

        # RMS 바
        self.ax_rms.set_facecolor('#0d1117')
        self.ax_rms.set_title('Current RMS', color='#e0e0e0', fontsize=9)
        self.ax_rms.tick_params(colors='#aaa', labelsize=7)
        self.ax_rms.grid(True, axis='y', alpha=0.2)
        self.rms_bars = self.ax_rms.bar(
            ['MIC1', 'MIC2', 'MIC3'],
            [0, 0, 0],
            color=['#00d4ff', '#ff6b9d', '#f39c12'],
            alpha=0.8)
        self.ax_rms.set_ylim(0, 0.01)

        # 파형
        self.ax_wave.set_facecolor('#0d1117')
        self.ax_wave.set_title('Waveform', color='#e0e0e0', fontsize=9)
        self.ax_wave.set_ylim(-0.01, 0.01)
        self.ax_wave.tick_params(colors='#aaa', labelsize=7)
        self.ax_wave.grid(True, alpha=0.2)
        self.ax_wave.set_xlabel('Time (ms)', color='#aaa', fontsize=7)
        DISPLAY = 3000
        x = np.linspace(0, 300, DISPLAY)
        self.wave1, = self.ax_wave.plot(x, np.zeros(DISPLAY),
                                         color='#00d4ff', lw=0.6, label='M1')
        self.wave2, = self.ax_wave.plot(x, np.zeros(DISPLAY),
                                         color='#ff6b9d', lw=0.6, label='M2')
        self.wave3, = self.ax_wave.plot(x, np.zeros(DISPLAY),
                                         color='#f39c12', lw=0.6, label='M3')
        self.ax_wave.legend(loc='upper right', fontsize=7)
        self.DISPLAY = DISPLAY

        canvas = FigureCanvasTkAgg(self.fig, master=right)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas = canvas

    def _setup_2d_plot(self):
        """2D 삼각형 플롯 초기화"""
        ax = self.ax_2d
        ax.set_facecolor('#0d1117')
        ax.set_title('Touch Position (2D)', color='#e0e0e0', fontsize=11)
        ax.set_xlim(-20, 190)
        ax.set_ylim(-20, 135)
        ax.set_xlabel('X (mm)', color='#aaa', fontsize=9)
        ax.set_ylabel('Y (mm)', color='#aaa', fontsize=9)
        ax.tick_params(colors='#aaa', labelsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)

        # 삼각형 영역
        triangle = plt.Polygon(MIC_POSITIONS,
                                fill=True, facecolor='#1e3a5f',
                                edgecolor='#3498db', linewidth=1.5,
                                alpha=0.4)
        ax.add_patch(triangle)

        # 캘리브레이션 위치
        for pos in CALIB_POSITIONS:
            ax.plot(pos[0], pos[1], 'o',
                    color='#f39c12', markersize=6, alpha=0.5)

        # 마이크 위치
        colors = ['#00d4ff', '#ff6b9d', '#2ecc71']
        labels = ['MIC1\n(170,0)', 'MIC2\n(85,115)', 'MIC3\n(0,0)']
        for i, (pos, c, lbl) in enumerate(zip(MIC_POSITIONS, colors, labels)):
            ax.plot(pos[0], pos[1], 's',
                    color=c, markersize=12, zorder=5)
            ax.annotate(lbl, pos,
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=c)

        # 터치 마커
        self.touch_marker, = ax.plot(
            [85], [50], 'o',
            color='#e74c3c', markersize=18,
            alpha=0.85, zorder=10)
        self.pos_text = ax.text(
            85, 50, '(85, 50)',
            color='white', fontsize=8,
            ha='center', va='center',
            fontweight='bold', zorder=11)

    def _measure_noise(self):
        """3초 노이즈 측정"""
        self.noise_lbl.config(text="Measuring... (3s)")
        self.root.update()

        samples = [[], [], []]
        start = time.time()
        while time.time() - start < 3.0:
            for i, buf in enumerate([self.buf1, self.buf2, self.buf3]):
                b = list(buf)
                if len(b) >= 1000:
                    samples[i].append(calc_rms(b[-1000:]))
            time.sleep(0.1)

        if samples[0]:
            self.noise1 = float(np.mean(samples[0]))
            self.noise2 = float(np.mean(samples[1])) if samples[1] else 0
            self.noise3 = float(np.mean(samples[2])) if samples[2] else 0
            self.noise_measured = True
            self.noise_lbl.config(
                text=f"N1:{self.noise1:.5f}\n"
                     f"N2:{self.noise2:.5f}\n"
                     f"N3:{self.noise3:.5f}",
                fg='#2ecc71')
            self.noise_label.config(
                text=f"Noise measured",
                fg='#2ecc71')
            print(f"Noise: MIC1={self.noise1:.5f}, MIC2={self.noise2:.5f}, MIC3={self.noise3:.5f}")

    def _start_calib(self):
        self.calib_idx   = 0
        self.calib_count = 0
        self.calib_pos   = []
        self.calib_rms   = []
        self.calib_tmp   = []
        self.in_calib    = True
        self.save_btn.config(state='normal')
        self._update_calib_status()
        print("Calibration started!")

    def _update_calib_status(self):
        if self.calib_idx >= len(CALIB_POSITIONS):
            return
        pos    = CALIB_POSITIONS[self.calib_idx]
        remain = CALIB_SAMPLES - self.calib_count
        self.calib_pos_lbl.config(
            text=f"Tap at {pos}\n→ [S] ({remain} left)")
        self.calib_status.config(
            text=f"{self.calib_idx} / {len(CALIB_POSITIONS)} done")

    def _save_sample(self):
        if not self.in_calib:
            return

        bufs = [list(self.buf1), list(self.buf2), list(self.buf3)]
        if any(len(b) < 1000 for b in bufs):
            return

        # 최근 2초에서 피크 RMS
        rms_vals = []
        for buf in bufs:
            rms_vals.append(find_peak_rms(buf))

        # 노이즈 제거
        if self.noise_measured:
            rms_vals[0] = max(rms_vals[0] - self.noise1, 0.0)
            rms_vals[1] = max(rms_vals[1] - self.noise2, 0.0)
            rms_vals[2] = max(rms_vals[2] - self.noise3, 0.0)

        if max(rms_vals) < 0.00005:
            self.calib_pos_lbl.config(text="No signal! Tap then [S]")
            return

        self.calib_tmp.append(rms_vals)
        self.calib_count += 1

        pos = CALIB_POSITIONS[self.calib_idx]
        print(f"  [{self.calib_count}/{CALIB_SAMPLES}] pos={pos} "
              f"RMS=[{rms_vals[0]:.5f}, {rms_vals[1]:.5f}, {rms_vals[2]:.5f}]")

        if self.calib_count < CALIB_SAMPLES:
            self._update_calib_status()
            return

        # 10번 모이면 median
        tmp = np.array(self.calib_tmp)
        med = np.median(tmp, axis=0).tolist()
        print(f"  → Median: {[f'{v:.5f}' for v in med]}")

        self.calib_pos.append(pos)
        self.calib_rms.append(med)

        self.calib_tmp   = []
        self.calib_count = 0
        self.calib_idx  += 1

        if self.calib_idx >= len(CALIB_POSITIONS):
            # 캘리브레이션 완료
            self.in_calib = False
            self.save_btn.config(state='disabled')
            self.a, self.b = calibrate(
                self.calib_pos,
                self.calib_rms,
                MIC_POSITIONS
            )
            self.calibrated = True
            self.calib_label.config(text="Calibration Done!", fg='#2ecc71')
            self.param_label.config(text=f"a={self.a:.3f}, b={self.b:.3f}")
            self.calib_status.config(text="Done! Touch anywhere!")
            self.calib_pos_lbl.config(text="Touch the surface!")
        else:
            self._update_calib_status()

    def _update_loop(self):
        try:
            b1 = list(self.buf1)
            b2 = list(self.buf2)
            b3 = list(self.buf3)

            # 연결 상태
            c1 = self.rec1.connected
            c2 = self.rec2.connected
            c3 = self.rec3.connected
            if c1 and c2 and c3:
                self.conn_label.config(text="ESP32 #1 #2: Connected", fg='#2ecc71')
            elif c1 or c2:
                self.conn_label.config(text="ESP32 #1: Connected", fg='#f39c12')
            else:
                self.conn_label.config(text="ESP32: Waiting...", fg='#e74c3c')

            if len(b1) >= WINDOW_SIZE and len(b2) >= WINDOW_SIZE:
                rms1 = calc_rms(b1[-WINDOW_SIZE:])
                rms2 = calc_rms(b2[-WINDOW_SIZE:])
                rms3 = calc_rms(b3[-WINDOW_SIZE:]) if len(b3) >= WINDOW_SIZE else 0.0

                self.rms_label.config(
                    text=f"RMS1:{rms1:.5f}  RMS2:{rms2:.5f}  RMS3:{rms3:.5f}")

                # RMS 바 업데이트
                max_rms = max(rms1, rms2, rms3, 0.001)
                self.rms_bars[0].set_height(rms1)
                self.rms_bars[1].set_height(rms2)
                self.rms_bars[2].set_height(rms3)
                self.ax_rms.set_ylim(0, max_rms * 1.5)

                # 파형 업데이트
                if len(b1) >= self.DISPLAY:
                    d1 = np.array(b1[-self.DISPLAY:])
                    d2 = np.array(b2[-self.DISPLAY:]) if len(b2) >= self.DISPLAY else np.zeros(self.DISPLAY)
                    d3 = np.array(b3[-self.DISPLAY:]) if len(b3) >= self.DISPLAY else np.zeros(self.DISPLAY)
                    self.wave1.set_ydata(d1)
                    self.wave2.set_ydata(d2)
                    self.wave3.set_ydata(d3)
                    peak = max(abs(d1).max(), abs(d2).max(), abs(d3).max(), 0.002)
                    self.ax_wave.set_ylim(-peak*1.3, peak*1.3)

                # 위치 추정
                now = time.time()
                if self.calibrated and now - self.last_update > 0.15:
                    r1 = max(rms1 - self.noise1, 0.0)
                    r2 = max(rms2 - self.noise2, 0.0)
                    r3 = max(rms3 - self.noise3, 0.0)

                    if max(r1, r2, r3) > 0.00005:
                        pos = estimate_position_2d(
                            [r1, r2, r3], MIC_POSITIONS, self.a, self.b)
                        self.estimated_pos = pos
                        self.last_update = now

                # 마커 업데이트
                px, py = self.estimated_pos
                self.touch_marker.set_xdata([px])
                self.touch_marker.set_ydata([py])
                self.pos_text.set_x(px)
                self.pos_text.set_y(py)
                self.pos_text.set_text(f'({px:.0f},{py:.0f})')

            self.canvas.draw_idle()

        except Exception as e:
            print(f"[Error] {e}")

        self.root.after(150, self._update_loop)

    def on_close(self):
        self.rec1.stop()
        self.rec2.stop()
        self.rec3.stop()
        self.root.destroy()


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  2D Touch Location Estimator (Triangle Array)")
    print("=" * 60)
    print(f"  MIC1: {MIC_POSITIONS[0]} mm  → UDP {UDP_PORT_1}")
    print(f"  MIC2: {MIC_POSITIONS[1]} mm  → UDP {UDP_PORT_2}")
    print(f"  MIC3: {MIC_POSITIONS[2]} mm  → UDP {UDP_PORT_3}")
    print()
    print("  Steps:")
    print("  1. [Measure Noise (3s)]")
    print("  2. [Start Calibration] → tap each position → [S] x10")
    print("  3. Touch anywhere → see 2D position!")
    print("=" * 60)

    root = tk.Tk()
    app  = LocationApp2D(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
