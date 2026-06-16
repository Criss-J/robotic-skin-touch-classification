"""
Robotic Skin - Capsule Effect Test (Single MIC)
================================================
MIC1 하나만 사용!
  PHASE A: No Cap (뚜껑 없이)
  PHASE B: With Cap (뚜껑 씌우고)
  → 완전히 동일한 마이크로 비교!

실행:
    python 5_capsule_test.py
"""

import socket
import threading
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from collections import deque

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
UDP_IP      = "0.0.0.0"
UDP_PORT_1  = 12345
SAMPLE_RATE = 10000
WINDOW_SIZE = 5120
DISPLAY_SIZE= 10000
TEST_DISTANCES = [20, 50, 80, 100, 120]

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


def calc_rms(data):
    return float(np.sqrt(np.mean(np.array(data)**2)))


def find_peak_rms(buf, window=1000, stride=200):
    arr = np.array(buf[-SAMPLE_RATE*2:])
    best = 0.0
    for i in range(0, max(1, len(arr) - window), stride):
        r = calc_rms(arr[i:i+window])
        if r > best:
            best = r
    return best


# ──────────────────────────────────────────
# 메인 GUI
# ──────────────────────────────────────────
class SingleMicTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Capsule Effect Test - Single MIC (MIC1 only)")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("1200x820")

        self.buf = deque(maxlen=SAMPLE_RATE * 5)

        self.noise_a   = 0.0
        self.noise_b   = 0.0
        self.results_a = []
        self.results_b = []
        self.freq_a    = {'low': [], 'high': []}
        self.freq_b    = {'low': [], 'high': []}

        self.current_phase = 'A'
        self.dist_idx      = 0

        self.rec = UDPReceiver(self.buf, UDP_PORT_1)
        self.rec.start()

        self._build_ui()
        self._update_loop()

    def _build_ui(self):
        # 상단
        top = tk.Frame(self.root, bg='#0f3460', pady=8)
        top.pack(fill='x', padx=10, pady=(10,0))

        self.conn_label = tk.Label(top, text="ESP32: Waiting...",
                                    font=('Helvetica', 11),
                                    bg='#0f3460', fg='#f39c12')
        self.conn_label.pack(side='left', padx=15)

        tk.Label(top, text="Single MIC Test - Individual differences eliminated!",
                 font=('Helvetica', 11, 'bold'),
                 bg='#0f3460', fg='#2ecc71').pack(side='left', padx=15)

        self.rms_label = tk.Label(top, text="RMS: 0.00000",
                                   font=('Courier', 11),
                                   bg='#0f3460', fg='#aaa')
        self.rms_label.pack(side='right', padx=15)

        # 메인
        main = tk.Frame(self.root, bg='#1a1a2e')
        main.pack(fill='both', expand=True, padx=10, pady=8)

        # 왼쪽 패널
        left = tk.Frame(main, bg='#16213e', width=270)
        left.pack(side='left', fill='y', padx=(0,8))
        left.pack_propagate(False)

        tk.Label(left, text="Test Steps",
                 font=('Helvetica', 13, 'bold'),
                 bg='#16213e', fg='white').pack(pady=(12,6))

        self.phase_label = tk.Label(left, text="PHASE A: No Cap",
                                     font=('Helvetica', 12, 'bold'),
                                     bg='#e74c3c', fg='white', pady=8)
        self.phase_label.pack(fill='x', padx=8, pady=(0,8))

        # STEP 1
        s1 = tk.LabelFrame(left, text="STEP 1. Noise",
                            font=('Helvetica', 9, 'bold'),
                            bg='#16213e', fg='#f39c12', padx=6, pady=6)
        s1.pack(fill='x', padx=8, pady=3)
        tk.Label(s1, text="Do nothing, click button",
                 font=('Helvetica', 8), bg='#16213e', fg='#aaa').pack()
        tk.Button(s1, text="Measure Noise (3s)",
                  font=('Helvetica', 9, 'bold'),
                  bg='#2d3561', fg='white', relief='flat',
                  cursor='hand2', pady=4,
                  command=self._measure_noise).pack(fill='x', pady=(4,0))
        self.noise_lbl = tk.Label(s1, text="Not measured",
                                   font=('Courier', 8),
                                   bg='#16213e', fg='#aaa')
        self.noise_lbl.pack()

        # STEP 2
        s2 = tk.LabelFrame(left, text="STEP 2. RMS by Distance",
                            font=('Helvetica', 9, 'bold'),
                            bg='#16213e', fg='#3498db', padx=6, pady=6)
        s2.pack(fill='x', padx=8, pady=3)
        self.dist_lbl = tk.Label(s2,
                                  text=f"Tap at {TEST_DISTANCES[0]}mm then [S]",
                                  font=('Helvetica', 8), bg='#16213e', fg='#aaa',
                                  wraplength=210)
        self.dist_lbl.pack()
        tk.Button(s2, text="Save [S]",
                  font=('Helvetica', 9, 'bold'),
                  bg='#2d3561', fg='white', relief='flat',
                  cursor='hand2', pady=4,
                  command=self._save_dist).pack(fill='x', pady=(4,0))
        self.dist_status = tk.Label(s2, text="0 / 5 done",
                                     font=('Courier', 8),
                                     bg='#16213e', fg='#aaa')
        self.dist_status.pack()

        # STEP 3
        s3 = tk.LabelFrame(left, text="STEP 3. Frequency",
                            font=('Helvetica', 9, 'bold'),
                            bg='#16213e', fg='#2ecc71', padx=6, pady=6)
        s3.pack(fill='x', padx=8, pady=3)
        tk.Label(s3, text="Low: slow stroke / High: fast scratch",
                 font=('Helvetica', 8), bg='#16213e', fg='#aaa').pack()
        tk.Button(s3, text="Save Low Freq (stroke)",
                  font=('Helvetica', 9, 'bold'),
                  bg='#16a085', fg='white', relief='flat',
                  cursor='hand2', pady=3,
                  command=lambda: self._save_freq('low')).pack(fill='x', pady=(4,2))
        tk.Button(s3, text="Save High Freq (scratch)",
                  font=('Helvetica', 9, 'bold'),
                  bg='#8e44ad', fg='white', relief='flat',
                  cursor='hand2', pady=3,
                  command=lambda: self._save_freq('high')).pack(fill='x', pady=2)
        self.freq_lbl = tk.Label(s3, text="Low: 0  High: 0",
                                  font=('Courier', 8),
                                  bg='#16213e', fg='#aaa')
        self.freq_lbl.pack()

        tk.Frame(left, bg='#2d3561', height=1).pack(fill='x', padx=8, pady=8)

        self.switch_btn = tk.Button(left,
                                     text="→ Done! Switch to PHASE B (With Cap)",
                                     font=('Helvetica', 9, 'bold'),
                                     bg='#f39c12', fg='white', relief='flat',
                                     cursor='hand2', pady=8,
                                     command=self._switch_phase)
        self.switch_btn.pack(fill='x', padx=8, pady=2)

        tk.Button(left, text="Show Final Results",
                  font=('Helvetica', 11, 'bold'),
                  bg='#e74c3c', fg='white', relief='flat',
                  cursor='hand2', pady=8,
                  command=self._show_results).pack(fill='x', padx=8, pady=(4,2))

        self.root.bind('<s>', lambda e: self._save_dist())
        self.root.bind('<S>', lambda e: self._save_dist())
        self.root.focus_set()

        # 오른쪽 그래프
        right = tk.Frame(main, bg='#1a1a2e')
        right.pack(side='left', fill='both', expand=True)

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(8, 6), facecolor='#1a1a2e')
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.4, wspace=0.35)

        self.ax_wave = self.fig.add_subplot(gs[0, :])
        self.ax_wave.set_facecolor('#0d1117')
        self.ax_wave.set_title('MIC1 Real-time Waveform', color='#e0e0e0', fontsize=11)
        self.ax_wave.set_ylim(-0.01, 0.01)
        self.ax_wave.set_xlabel('Time (ms)', color='#aaa', fontsize=8)
        self.ax_wave.tick_params(colors='#aaa', labelsize=8)
        self.ax_wave.grid(True, alpha=0.2)
        x = np.linspace(0, 1000, DISPLAY_SIZE)
        self.line_wave, = self.ax_wave.plot(x, np.zeros(DISPLAY_SIZE),
                                             color='#00d4ff', lw=0.8,
                                             label='Phase A: No Cap')
        self.ax_wave.legend(loc='upper right', fontsize=8)

        self.ax_rms = self.fig.add_subplot(gs[1, 0])
        self.ax_rms.set_facecolor('#0d1117')
        self.ax_rms.set_title('RMS Progress', color='#e0e0e0', fontsize=10)
        self.ax_rms.set_xlabel('Distance (mm)', color='#aaa', fontsize=8)
        self.ax_rms.set_ylabel('RMS', color='#aaa', fontsize=8)
        self.ax_rms.tick_params(colors='#aaa', labelsize=7)
        self.ax_rms.grid(True, alpha=0.2)
        self.line_a, = self.ax_rms.plot([], [], 'o-', color='#e74c3c',
                                         lw=2, label='No Cap')
        self.line_b, = self.ax_rms.plot([], [], 'o-', color='#2ecc71',
                                         lw=2, label='With Cap')
        self.ax_rms.legend(fontsize=8)

        self.ax_spec = self.fig.add_subplot(gs[1, 1])
        self.ax_spec.set_facecolor('#0d1117')
        self.ax_spec.set_title('Spectrogram', color='#e0e0e0', fontsize=10)
        self.ax_spec.set_xlabel('Time (ms)', color='#aaa', fontsize=8)
        self.ax_spec.set_ylabel('Freq (Hz)', color='#aaa', fontsize=8)
        self.ax_spec.tick_params(colors='#aaa', labelsize=7)
        self.spec_img = self.ax_spec.imshow(
            np.zeros((64, 40)), aspect='auto', origin='lower',
            extent=[0,512,0,5000], cmap='inferno', vmin=0, vmax=3)

        canvas = FigureCanvasTkAgg(self.fig, master=right)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas = canvas

    def _switch_phase(self):
        if self.current_phase == 'A':
            if not self.results_a:
                messagebox.showwarning("Warning", "Phase A has no data!")
                return
            self.current_phase = 'B'
            self.dist_idx = 0
            self.phase_label.config(text="PHASE B: With Cap", bg='#2ecc71')
            self.switch_btn.config(text="Phase B in progress...",
                                   state='disabled', bg='#555')
            self.line_wave.set_label('Phase B: With Cap')
            self.line_wave.set_color('#2ecc71')
            self.dist_lbl.config(
                text=f"PUT CAP ON MIC1!\nTap at {TEST_DISTANCES[0]}mm then [S]",
                fg='#f39c12')
            self.dist_status.config(text="0 / 5 done")
            self.noise_lbl.config(text="Not measured")
            messagebox.showinfo("Phase B Start",
                                "Put the bottle cap on MIC1 now!\n\n"
                                "Then:\n"
                                "1. Measure noise\n"
                                "2. Tap at each distance\n"
                                "3. Save stroke / scratch")

    def _measure_noise(self):
        self.noise_lbl.config(text="Measuring... (3s)")
        self.root.update()
        samples = []
        start = time.time()
        while time.time() - start < 3.0:
            b = list(self.buf)
            if len(b) >= 1000:
                samples.append(calc_rms(b[-1000:]))
            time.sleep(0.1)
        if samples:
            noise = float(np.mean(samples))
            if self.current_phase == 'A':
                self.noise_a = noise
            else:
                self.noise_b = noise
            self.noise_lbl.config(text=f"Noise: {noise:.5f}", fg='#2ecc71')
            print(f"[Phase {self.current_phase}] Noise: {noise:.5f}")

    def _save_dist(self):
        if self.dist_idx >= len(TEST_DISTANCES):
            return
        b = list(self.buf)
        if len(b) < 1000:
            return
        rms = find_peak_rms(b)
        if rms < 0.00005:
            self.dist_lbl.config(text="No signal! Tap then [S]")
            return

        dist  = TEST_DISTANCES[self.dist_idx]
        noise = self.noise_a if self.current_phase == 'A' else self.noise_b
        snr   = rms / noise if noise > 0 else 0
        result = {'dist': dist, 'rms': rms, 'snr': snr}

        if self.current_phase == 'A':
            self.results_a.append(result)
            da = [r['dist'] for r in self.results_a]
            ra = [r['rms']  for r in self.results_a]
            self.line_a.set_data(da, ra)
        else:
            self.results_b.append(result)
            db = [r['dist'] for r in self.results_b]
            rb = [r['rms']  for r in self.results_b]
            self.line_b.set_data(db, rb)

        all_rms = [r['rms'] for r in self.results_a + self.results_b]
        if all_rms:
            self.ax_rms.set_xlim(min(TEST_DISTANCES)-5, max(TEST_DISTANCES)+5)
            self.ax_rms.set_ylim(0, max(all_rms)*1.4)

        print(f"[Phase {self.current_phase}] {dist}mm: RMS={rms:.5f}, SNR={snr:.1f}x")
        self.dist_idx += 1
        self.dist_status.config(text=f"{self.dist_idx} / {len(TEST_DISTANCES)} done")

        if self.dist_idx < len(TEST_DISTANCES):
            nd = TEST_DISTANCES[self.dist_idx]
            self.dist_lbl.config(text=f"Tap at {nd}mm then [S]", fg='#aaa')
        else:
            self.dist_lbl.config(text="Distance test done!", fg='#2ecc71')

    def _save_freq(self, freq_type):
        b = list(self.buf)
        if len(b) < WINDOW_SIZE:
            return
        _, _, Sxx = signal.spectrogram(
            np.array(b[-WINDOW_SIZE:]),
            fs=SAMPLE_RATE, nperseg=128, noverlap=64)
        if self.current_phase == 'A':
            self.freq_a[freq_type].append(np.log1p(Sxx[:64]))
        else:
            self.freq_b[freq_type].append(np.log1p(Sxx[:64]))
        nL = len(self.freq_a['low'])  + len(self.freq_b['low'])
        nH = len(self.freq_a['high']) + len(self.freq_b['high'])
        self.freq_lbl.config(text=f"Low: {nL}  High: {nH}")
        print(f"[Phase {self.current_phase}] {freq_type} saved")

    def _show_results(self):
        if not self.results_a or not self.results_b:
            messagebox.showwarning("Warning",
                "Need both Phase A (No Cap) and Phase B (With Cap) data!")
            return

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
        fig.suptitle(
            'Capsule Effect: Same MIC1 - No Cap vs With Cap\n'
            '(Individual MIC differences eliminated)',
            color='white', fontsize=13, fontweight='bold')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        def sort_results(results):
            s = sorted(results, key=lambda r: r['dist'])
            return ([r['dist'] for r in s],
                    [r['rms']  for r in s],
                    [r['snr']  for r in s])

        da, ra, sa = sort_results(self.results_a)
        db, rb, sb = sort_results(self.results_b)

        # 1. RMS 비교
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#0d1117')
        ax1.plot(da, ra, 'o-', color='#e74c3c', lw=2.5, markersize=7, label='No Cap')
        ax1.plot(db, rb, 'o-', color='#2ecc71', lw=2.5, markersize=7, label='With Cap')
        ax1.set_title('RMS by Distance\n(Same MIC1)', color='white', fontsize=11)
        ax1.set_xlabel('Distance (mm)', color='#aaa')
        ax1.set_ylabel('RMS', color='#aaa')
        ax1.tick_params(colors='#aaa')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)

        # 2. SNR 비교
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#0d1117')
        ax2.plot(da, sa, 'o-', color='#e74c3c', lw=2.5, markersize=7, label='No Cap')
        ax2.plot(db, sb, 'o-', color='#2ecc71', lw=2.5, markersize=7, label='With Cap')
        ax2.set_title('SNR by Distance\n(Same MIC1)', color='white', fontsize=11)
        ax2.set_xlabel('Distance (mm)', color='#aaa')
        ax2.set_ylabel('SNR (x)', color='#aaa')
        ax2.tick_params(colors='#aaa')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        # 3. 향상률
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor('#0d1117')
        ra_d = dict(zip(da, ra))
        rb_d = dict(zip(db, rb))
        common = sorted(set(da) & set(db))
        imprv  = [(rb_d[d]/ra_d[d] - 1)*100 for d in common]
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in imprv]
        bars = ax3.bar([f'{d}mm' for d in common], imprv,
                       color=colors, alpha=0.85, width=0.6)
        ax3.axhline(0, color='white', lw=0.8, linestyle='--')
        ax3.set_title('Cap Effect\n(With Cap / No Cap - 1)',
                      color='white', fontsize=11)
        ax3.set_xlabel('Distance', color='#aaa')
        ax3.set_ylabel('Improvement (%)', color='#aaa')
        ax3.tick_params(colors='#aaa')
        ax3.grid(True, axis='y', alpha=0.2)
        for bar, val in zip(bars, imprv):
            yoff = 1 if val >= 0 else -5
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + yoff,
                     f'{val:+.1f}%', ha='center',
                     color='white', fontsize=9, fontweight='bold')

        # 4. 저주파 스펙트로그램
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor('#0d1117')
        if self.freq_a['low']:
            ax4.imshow(np.mean(self.freq_a['low'], axis=0),
                       aspect='auto', origin='lower',
                       extent=[0,512,0,5000], cmap='inferno')
        ax4.set_title('Low Freq (stroke) - No Cap', color='white', fontsize=10)
        ax4.set_xlabel('Time (ms)', color='#aaa', fontsize=8)
        ax4.set_ylabel('Freq (Hz)', color='#aaa', fontsize=8)
        ax4.tick_params(colors='#aaa', labelsize=7)

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor('#0d1117')
        if self.freq_b['low']:
            ax5.imshow(np.mean(self.freq_b['low'], axis=0),
                       aspect='auto', origin='lower',
                       extent=[0,512,0,5000], cmap='inferno')
        ax5.set_title('Low Freq (stroke) - With Cap', color='white', fontsize=10)
        ax5.set_xlabel('Time (ms)', color='#aaa', fontsize=8)
        ax5.set_ylabel('Freq (Hz)', color='#aaa', fontsize=8)
        ax5.tick_params(colors='#aaa', labelsize=7)

        # 5. 주파수 에너지
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor('#0d1117')
        freqs = np.linspace(0, 5000, 64)
        if self.freq_a['low']:
            m = np.mean([np.mean(s, axis=1) for s in self.freq_a['low']], axis=0)
            ax6.plot(freqs, m, '--', color='#e74c3c', lw=1.5, label='No Cap (Low)')
        if self.freq_b['low']:
            m = np.mean([np.mean(s, axis=1) for s in self.freq_b['low']], axis=0)
            ax6.plot(freqs, m, '--', color='#2ecc71', lw=1.5, label='With Cap (Low)')
        if self.freq_a['high']:
            m = np.mean([np.mean(s, axis=1) for s in self.freq_a['high']], axis=0)
            ax6.plot(freqs, m, color='#e74c3c', lw=1.5, label='No Cap (High)')
        if self.freq_b['high']:
            m = np.mean([np.mean(s, axis=1) for s in self.freq_b['high']], axis=0)
            ax6.plot(freqs, m, color='#2ecc71', lw=1.5, label='With Cap (High)')
        ax6.set_title('Frequency Energy Comparison', color='white', fontsize=10)
        ax6.set_xlabel('Frequency (Hz)', color='#aaa', fontsize=8)
        ax6.set_ylabel('Energy', color='#aaa', fontsize=8)
        ax6.tick_params(colors='#aaa', labelsize=7)
        ax6.legend(fontsize=7)
        ax6.grid(True, alpha=0.2)

        # 결과 출력
        print("\n" + "="*65)
        print("  Result Summary (Same MIC1 - Individual diff. eliminated)")
        print("="*65)
        print(f"  Noise A (No Cap):   {self.noise_a:.5f}")
        print(f"  Noise B (With Cap): {self.noise_b:.5f}")
        print()
        print(f"  {'Dist':>6} | {'RMS(NoCap)':>10} | {'RMS(Cap)':>10} | "
              f"{'SNR(NoCap)':>10} | {'SNR(Cap)':>9} | {'Imprv':>8}")
        print(f"  {'-'*72}")
        sa_d = dict(zip(da, sa))
        sb_d = dict(zip(db, sb))
        for d in common:
            imp = (rb_d[d]/ra_d[d] - 1)*100
            print(f"  {d:>4}mm | {ra_d[d]:>10.5f} | {rb_d[d]:>10.5f} | "
                  f"{sa_d[d]:>9.1f}x | {sb_d[d]:>9.1f}x | {imp:>+7.1f}%")
        print("="*65)

        plt.savefig('capsule_test_result.png', dpi=150,
                    bbox_inches='tight', facecolor='#1a1a2e')
        print("  Saved: capsule_test_result.png")
        plt.show()

    def _update_loop(self):
        try:
            b = list(self.buf)
            if self.rec.connected:
                self.conn_label.config(text="ESP32: Connected", fg='#2ecc71')
            else:
                self.conn_label.config(text="ESP32: Waiting...", fg='#f39c12')

            if len(b) >= DISPLAY_SIZE:
                d   = np.array(b[-DISPLAY_SIZE:])
                rms = calc_rms(b[-1000:]) if len(b) >= 1000 else 0
                self.line_wave.set_ydata(d)
                peak = max(abs(d).max(), 0.002)
                self.ax_wave.set_ylim(-peak*1.3, peak*1.3)
                self.rms_label.config(text=f"RMS: {rms:.5f}")

                if len(b) >= WINDOW_SIZE:
                    _, _, Sxx = signal.spectrogram(
                        np.array(b[-WINDOW_SIZE:]),
                        fs=SAMPLE_RATE, nperseg=128, noverlap=64)
                    log_Sxx = np.log1p(Sxx[:64])
                    self.spec_img.set_data(log_Sxx)
                    self.spec_img.set_clim(log_Sxx.min(), log_Sxx.max()+1e-6)

            self.canvas.draw_idle()
        except Exception as e:
            print(f"[Error] {e}")
        self.root.after(100, self._update_loop)

    def on_close(self):
        self.rec.stop()
        self.root.destroy()


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Capsule Effect Test - Single MIC (MIC1 only)")
    print("=" * 60)
    print("  Same MIC1 → individual differences eliminated!")
    print()
    print("  PHASE A (No Cap):")
    print("    1. Measure noise")
    print("    2. Tap at each distance → [S]")
    print("    3. Stroke / Scratch → save")
    print()
    print("  Switch to PHASE B → put cap on MIC1 → repeat")
    print()
    print(f"  Distances: {TEST_DISTANCES} mm")
    print("=" * 60)

    root = tk.Tk()
    app  = SingleMicTestApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
