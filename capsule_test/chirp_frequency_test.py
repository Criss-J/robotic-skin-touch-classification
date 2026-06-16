"""
Chirp 주파수 응답 테스트
========================
논문 방식을 DIY로 재현:
  - Chirp 신호 (20Hz ~ 5kHz) WAV 파일 생성
  - ESP32 마이크 2개로 동시 녹음 (1개: 병뚜껑 없음, 1개: 병뚜껑 있음)
  - FFT로 주파수 응답 비교 → 그래프 저장

사용법:
  STEP 1: python chirp_frequency_test.py --make-chirp
          → chirp_20_5000hz.wav 생성 후 스마트폰/스피커로 재생

  STEP 2: ESP32 연결 후 실시간 녹음
          python chirp_frequency_test.py --record

  STEP 3: 녹음된 데이터로 분석
          python chirp_frequency_test.py --analyze
"""

import argparse
import socket
import threading
import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

# ── 설정 ─────────────────────────────────────────────────────
UDP_IP       = "0.0.0.0"
UDP_PORT_1   = 12345    # MIC1: 병뚜껑 없음
UDP_PORT_2   = 12346    # MIC2: 병뚜껑 있음
SAMPLE_RATE  = 10000    # ESP32 샘플링 레이트 (Hz)

CHIRP_FS     = 44100    # Chirp WAV 파일 샘플레이트
CHIRP_F_START = 20      # 시작 주파수 (Hz)
CHIRP_F_END   = 5000    # 끝 주파수 (Hz)
CHIRP_DURATION = 10     # 스윕 시간 (초)

RECORD_DURATION = 15    # 녹음 시간 (초, Chirp 10초 + 여유 5초)

SAVE_DIR = "chirp_data"
os.makedirs(SAVE_DIR, exist_ok=True)


# ── STEP 1: Chirp WAV 생성 ────────────────────────────────────
def make_chirp():
    print("=" * 50)
    print("  Chirp 신호 생성")
    print("=" * 50)
    print(f"  주파수 범위: {CHIRP_F_START}Hz ~ {CHIRP_F_END}Hz")
    print(f"  스윕 시간:   {CHIRP_DURATION}초")
    print(f"  스케일:      로그(logarithmic)")
    print()

    t = np.linspace(0, CHIRP_DURATION,
                    int(CHIRP_FS * CHIRP_DURATION), endpoint=False)

    # 로그 스케일 Chirp 신호 생성
    sig = chirp(t, f0=CHIRP_F_START, f1=CHIRP_F_END,
                t1=CHIRP_DURATION, method='logarithmic')

    # 앞뒤 0.5초 페이드 인/아웃 (클리핑 방지)
    fade = int(CHIRP_FS * 0.5)
    sig[:fade]  *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)

    sig_int16 = (sig * 32767 * 0.8).astype(np.int16)

    out_path = os.path.join(SAVE_DIR, "chirp_20_5000hz.wav")
    wavfile.write(out_path, CHIRP_FS, sig_int16)

    print(f"  저장 완료: {out_path}")
    print()
    print("  다음 단계:")
    print("  1. chirp_20_5000hz.wav 를 스마트폰/PC로 복사")
    print("  2. 이어폰 or 소형 스피커를 에어메쉬에 밀착")
    print("  3. 파일 재생하면서:")
    print("     python chirp_frequency_test.py --record")
    print()
    print("  ※ MIC1 = 병뚜껑 없음 (UDP 12345)")
    print("  ※ MIC2 = 병뚜껑 있음  (UDP 12346)")


# ── UDP 수신기 ────────────────────────────────────────────────
class UDPReceiver(threading.Thread):
    def __init__(self, port, label):
        super().__init__(daemon=True)
        self.port    = port
        self.label   = label
        self.buf     = []
        self.running = True
        self.connected = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, port))
        self.sock.settimeout(1.0)

    def run(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(8192)
                samples = (np.frombuffer(data, dtype=np.int32)
                           .astype(np.float32) / 2**31)
                self.buf.extend(samples.tolist())
                self.connected = True
            except socket.timeout:
                self.connected = False

    def stop(self):
        self.running = False
        self.sock.close()


# ── STEP 2: 녹음 ─────────────────────────────────────────────
def record():
    print("=" * 50)
    print("  실시간 녹음 시작")
    print("=" * 50)
    print(f"  녹음 시간: {RECORD_DURATION}초")
    print()
    print("  MIC1 (병뚜껑 없음): UDP 12345 대기 중...")
    print("  MIC2 (병뚜껑 있음):  UDP 12346 대기 중...")
    print()

    rec1 = UDPReceiver(UDP_PORT_1, "MIC1")
    rec2 = UDPReceiver(UDP_PORT_2, "MIC2")
    rec1.start()
    rec2.start()

    # 연결 대기
    print("  ESP32 연결 대기 중...", end="", flush=True)
    for _ in range(10):
        if rec1.connected or rec2.connected:
            break
        time.sleep(1)
        print(".", end="", flush=True)
    print()

    if not rec1.connected and not rec2.connected:
        print("  ⚠ ESP32 연결 안 됨! 계속 진행합니다...")
    else:
        print("  ✅ ESP32 연결됨!")

    print()
    print("  ▶ 지금 Chirp 파일 재생하세요!")
    print()

    start = time.time()
    while time.time() - start < RECORD_DURATION:
        elapsed = time.time() - start
        bar = int(elapsed / RECORD_DURATION * 30)
        print(f"\r  녹음 중: [{'#'*bar}{'.'*(30-bar)}] "
              f"{elapsed:.1f}s / {RECORD_DURATION}s  "
              f"MIC1:{len(rec1.buf):5d}샘플 "
              f"MIC2:{len(rec2.buf):5d}샘플", end="", flush=True)
        time.sleep(0.2)

    print()
    rec1.stop()
    rec2.stop()
    time.sleep(0.5)

    # 저장
    arr1 = np.array(rec1.buf, dtype=np.float32)
    arr2 = np.array(rec2.buf, dtype=np.float32)

    np.save(os.path.join(SAVE_DIR, "mic1_nocap.npy"), arr1)
    np.save(os.path.join(SAVE_DIR, "mic2_cap.npy"),   arr2)

    print()
    print(f"  저장 완료!")
    print(f"    MIC1 (병뚜껑 없음): {len(arr1)}샘플 "
          f"({len(arr1)/SAMPLE_RATE:.1f}초)")
    print(f"    MIC2 (병뚜껑 있음):  {len(arr2)}샘플 "
          f"({len(arr2)/SAMPLE_RATE:.1f}초)")
    print()
    print("  다음 단계:")
    print("    python chirp_frequency_test.py --analyze")


# ── STEP 3: 분석 ─────────────────────────────────────────────
def analyze():
    print("=" * 50)
    print("  주파수 응답 분석")
    print("=" * 50)

    p1 = os.path.join(SAVE_DIR, "mic1_nocap.npy")
    p2 = os.path.join(SAVE_DIR, "mic2_cap.npy")

    if not os.path.exists(p1) or not os.path.exists(p2):
        print("  ❌ 녹음 파일 없음! 먼저 --record 실행하세요.")
        return

    mic1 = np.load(p1)
    mic2 = np.load(p2)

    # 길이 맞추기
    n = min(len(mic1), len(mic2))
    mic1 = mic1[:n]
    mic2 = mic2[:n]

    print(f"  MIC1 샘플 수: {len(mic1)} ({len(mic1)/SAMPLE_RATE:.1f}초)")
    print(f"  MIC2 샘플 수: {len(mic2)} ({len(mic2)/SAMPLE_RATE:.1f}초)")

    # Welch PSD (Power Spectral Density)
    nperseg = min(1024, len(mic1) // 4)
    f1, psd1 = welch(mic1, fs=SAMPLE_RATE, nperseg=nperseg)
    f2, psd2 = welch(mic2, fs=SAMPLE_RATE, nperseg=nperseg)

    # RMS 통계
    rms1 = np.sqrt(np.mean(mic1**2))
    rms2 = np.sqrt(np.mean(mic2**2))
    print(f"\n  전체 RMS:")
    print(f"    MIC1 (병뚜껑 없음): {rms1:.5f}")
    print(f"    MIC2 (병뚜껑 있음):  {rms2:.5f}")
    print(f"    향상률: {(rms2/rms1 - 1)*100:+.1f}%")

    # 주파수 대역별 에너지 비교
    bands = [
        ("저주파  (20~200Hz)",    20,   200),
        ("중주파  (200~1kHz)",   200,  1000),
        ("고주파  (1k~5kHz)", 1000,  5000),
    ]
    print("\n  주파수 대역별 RMS 비교:")
    for name, flo, fhi in bands:
        mask1 = (f1 >= flo) & (f1 <= fhi)
        mask2 = (f2 >= flo) & (f2 <= fhi)
        e1 = np.sqrt(np.mean(psd1[mask1])) if mask1.any() else 0
        e2 = np.sqrt(np.mean(psd2[mask2])) if mask2.any() else 0
        diff = (e2/e1 - 1)*100 if e1 > 0 else 0
        print(f"    {name}: 없음={e1:.5f}  있음={e2:.5f}  "
              f"({diff:+.1f}%)")

    # ── 한글 폰트 설정 ───────────────────────────────────────
    import matplotlib.font_manager as fm
    nanum = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(nanum):
        fm.fontManager.addfont(nanum)
        nanum_bold = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
        if os.path.exists(nanum_bold):
            fm.fontManager.addfont(nanum_bold)
        matplotlib.rcParams['font.family'] = 'NanumGothic'
    matplotlib.rcParams['axes.unicode_minus'] = False

    # ── 노이즈 측정 (신호 없는 구간으로 추정) ─────────────────
    # 앞 1초를 노이즈 구간으로 사용
    noise_len = SAMPLE_RATE
    noise1 = mic1[:noise_len]
    noise2 = mic2[:noise_len]
    _, npsd1 = welch(noise1, fs=SAMPLE_RATE, nperseg=min(512, noise_len//2))
    _, npsd2 = welch(noise2, fs=SAMPLE_RATE, nperseg=min(512, noise_len//2))

    # SNR(dB) 계산
    from scipy.interpolate import interp1d
    interp_n1 = interp1d(np.linspace(0, SAMPLE_RATE/2, len(npsd1)),
                         npsd1, fill_value='extrapolate')
    interp_n2 = interp1d(np.linspace(0, SAMPLE_RATE/2, len(npsd2)),
                         npsd2, fill_value='extrapolate')
    noise1_interp = interp_n1(f1)
    noise2_interp = interp_n2(f2)

    eps = 1e-20
    snr1_db = 10 * np.log10((psd1 + eps) / (noise1_interp + eps))
    snr2_db = 10 * np.log10((psd2 + eps) / (noise2_interp + eps))

    # ── MIC 라벨 정의 ────────────────────────────────────────
    L1 = 'MIC1 (캡슐 있음)'     # mic1 = 캡슐 있음
    L2 = 'MIC2 (캡슐 없음)'     # mic2 = 캡슐 없음
    L1_en = 'Encapsulated (MIC1)'
    L2_en = 'Unencapsulated (MIC2)'

    # ── 3x2 그래프 ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('캡슐(병뚜껑) 유무에 따른 마이크 주파수 응답 비교\n'
                 '(MIC1: 캡슐 있음  vs  MIC2: 캡슐 없음)',
                 fontsize=14, fontweight='bold')

    # 1. 파형 비교
    ax1 = axes[0, 0]
    t_arr = np.arange(n) / SAMPLE_RATE
    ax1.plot(t_arr, mic1, color='black',  lw=0.4, alpha=0.8, label=L1)
    ax1.plot(t_arr, mic2, color='gray',   lw=0.4, alpha=0.7, label=L2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('파형 비교')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 주파수 응답 PSD (선형)
    ax2 = axes[0, 1]
    ax2.plot(f1, psd1, color='black', lw=1.2, label=L1)
    ax2.plot(f2, psd2, color='gray',  lw=1.2, label=L2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD')
    ax2.set_title('주파수 응답 (선형 스케일)')
    ax2.set_xlim(0, SAMPLE_RATE // 2)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. SNR(dB) vs 주파수 ← 논문 Fig.7 방식
    ax3 = axes[0, 2]
    mask_f = (f1 >= 10) & (f1 <= SAMPLE_RATE // 2)
    ax3.plot(f2[mask_f], snr2_db[mask_f], color='gray',
             lw=1.5, label=L2_en)
    ax3.plot(f1[mask_f], snr1_db[mask_f], color='black',
             lw=2.0, label=L1_en)
    ax3.axvspan(10, 1000, alpha=0.1, color='gray', label='Pacinian bandwidth')
    ax3.set_xscale('log')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Signal-to-Noise Ratio (dB)')
    ax3.set_title('SNR vs Frequency\n(논문 Fig.7 방식)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(10, SAMPLE_RATE // 2)

    # 4. 주파수 응답 PSD (로그 스케일)
    ax4 = axes[1, 0]
    mask = f1 > 0
    ax4.semilogy(f1[mask], psd1[mask], color='black', lw=1.5, label=L1)
    ax4.semilogy(f2[mask], psd2[mask], color='gray',  lw=1.5, label=L2)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('PSD (log scale)')
    ax4.set_title('주파수 응답 (로그 스케일)')
    ax4.set_xlim(10, SAMPLE_RATE // 2)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.axvspan(10, 200, alpha=0.08, color='green')
    ax4.axvline(200, color='green', lw=0.8, linestyle='--', alpha=0.6)

    # 5. SNR(dB) 차이 (캡슐 있음 - 없음)
    ax5 = axes[1, 1]
    snr_diff = snr1_db[mask_f] - snr2_db[mask_f]
    ax5.fill_between(f1[mask_f], 0, snr_diff,
                     where=snr_diff >= 0, color='#2ecc71', alpha=0.6,
                     label='캡슐 효과 (+)')
    ax5.fill_between(f1[mask_f], 0, snr_diff,
                     where=snr_diff < 0, color='#e74c3c', alpha=0.6,
                     label='캡슐 효과 (-)')
    ax5.axhline(0, color='black', lw=0.8, linestyle='--')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('SNR 차이 (dB)')
    ax5.set_title('SNR 차이\n(캡슐 있음 - 없음)')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, SAMPLE_RATE // 2)

    # 6. 대역별 향상률 바 차트
    ax6 = axes[1, 2]
    band_names = []
    improvements = []
    for name, flo, fhi in bands:
        mask1 = (f1 >= flo) & (f1 <= fhi)
        mask2 = (f2 >= flo) & (f2 <= fhi)
        # mic1=캡슐있음, mic2=캡슐없음 → (캡슐있음/캡슐없음 - 1)
        e1 = np.sqrt(np.mean(psd1[mask1])) if mask1.any() else 1e-10
        e2 = np.sqrt(np.mean(psd2[mask2])) if mask2.any() else 1e-10
        improvements.append((e1/e2 - 1)*100)
        band_names.append(name.strip())

    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax6.bar(band_names, improvements, color=colors, alpha=0.85, width=0.5)
    ax6.axhline(0, color='black', lw=0.8, linestyle='--')
    ax6.set_ylabel('향상률 (%)')
    ax6.set_title('대역별 캡슐 효과\n(캡슐 있음 / 없음 - 1)')
    ax6.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, improvements):
        yoff = 1 if val >= 0 else -5
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + yoff,
                 f'{val:+.1f}%', ha='center',
                 fontsize=11, fontweight='bold',
                 color='#2c3e50')

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, "frequency_response.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    print(f"\n  그래프 저장: {out_path}")
    print("\n  분석 완료!")


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Chirp 주파수 응답 테스트')
    parser.add_argument('--make-chirp', action='store_true',
                        help='Chirp WAV 파일 생성')
    parser.add_argument('--record',     action='store_true',
                        help='마이크 녹음')
    parser.add_argument('--analyze',    action='store_true',
                        help='주파수 응답 분석')
    args = parser.parse_args()

    if args.make_chirp:
        make_chirp()
    elif args.record:
        record()
    elif args.analyze:
        analyze()
    else:
        print("사용법:")
        print("  python chirp_frequency_test.py --make-chirp")
        print("  python chirp_frequency_test.py --record")
        print("  python chirp_frequency_test.py --analyze")
