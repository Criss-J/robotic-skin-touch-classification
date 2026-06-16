# 🖐 Robotic Skin Touch Classification

> ESP32 + INMP441 마이크 + 에어메쉬를 이용한 실시간 터치 인식 시스템

논문 **"Robotic Skin Mimicking Human Skin Layer and Pacinian Corpuscle for Social Interaction"**
*(IEEE/ASME Transactions on Mechatronics, Vol.29, No.4, 2024 — Kyungseo Park, DGIST)*
을 참고하여 저가형 DIY 환경에서 재현한 프로젝트입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 터치 클래스 | idle / scratch / tap |
| 분류 정확도 | **97.5%** (검증 데이터 기준) |
| 마이크 | INMP441 × 3개 (I2S 디지털, 삼각형 배열) |
| MCU | ESP32 × 2개 |
| 전송 방식 | UDP (WiFi) |
| 샘플링 레이트 | 10,000 Hz |
| 윈도우 크기 | 512ms (5,120 샘플) |

---

## 🗂️ 폴더 구조

```
robotic-skin-touch-classification/
│
├── README.md
│
├── touch_classification/           # 터치 종류 분류 (CNN)
│   ├── 1_collect.py                # 데이터 수집 GUI (자동 감지 방식)
│   ├── 2_train.py                  # Dual-input CNN 학습
│   ├── 3_realtime.py               # 실시간 터치 분류
│   └── touch_classifier_3mic.h5    # 학습된 모델 파일
│
├── touch_location/                 # PAT 2D 위치 추정
│   └── 2d_location.py              # 2D PAT 위치 추정 (마이크 3개 삼각형 배열)
│
└── capsule_test/                   # 캡슐 효과 실험
    ├── capsule_test.py             # 거리별 RMS 비교 실험
    └── chirp_frequency_test.py     # 주파수별 SNR 분석 실험 (Chirp 신호)
```

---

## 🔧 하드웨어 구성

### 마이크 삼각형 배열 (3개)

```
     MIC2 (85, 115mm)
          ▲
         / \
        /   \
MIC3 ──────── MIC1
(0,0)        (170,0)
```

| 마이크 | 좌표 | ESP32 | UDP 포트 |
|--------|------|-------|---------|
| MIC1 | (170, 0) mm | ESP32 #1 | 12345 |
| MIC2 | (85, 115) mm | ESP32 #1 | 12346 |
| MIC3 | (0, 0) mm | ESP32 #2 | 12347 |

---

## 🧠 CNN 구조

논문의 Dual-input CNN을 재현. **마이크 3개** 신호로 두 가지 Feature를 추출하여 입력합니다.

| Feature | 설명 | Shape |
|---------|------|-------|
| Intensity Map | 128샘플(6.4ms) 단위 프레임별 RMS × MIC1 + MIC2 + MIC3 | (n_frames, 3, 1) |
| Spectrogram | 3개 중 신호가 가장 강한 마이크의 STFT 결과 (로그 스케일) | (n_frames, 64, 1) |

```
Intensity Map  → Conv2D(32, 5×2) → MaxPool → Conv2D(16, 5×2) → Flatten ─┐
                                                                            ├→ Concat → Dense(256) → Dense(128) → Softmax(3)
Spectrogram    → Conv2D(32, 5×5) → MaxPool → Conv2D(16, 5×5) → Flatten ─┘

출력: [idle, scratch, tap] 확률
```

### 학습 설정

| 항목 | 값 |
|------|----|
| Optimizer | Adam (lr=5e-5) |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping | patience=10 |
| 데이터 증강 | Time Shift (±200ms) + Gaussian Noise × 5배 |

---

## ⚙️ 환경 설정

### 개발 환경

| 항목 | 버전 |
|------|------|
| Python | 3.10 |
| TensorFlow | 2.12.0 |
| NumPy | 1.23.5 |
| SciPy | 1.10.1 |
| scikit-learn | 1.2.2 |
| matplotlib | 3.7.1 |
| tkinter | 기본 내장 (Python 3.10) |
| ESP32 Arduino Core | 2.0.x (by Espressif) |
| Arduino IDE | 2.x |

### Python 패키지 설치

```bash
pip install numpy==1.23.5 scipy==1.10.1 tensorflow==2.12.0 scikit-learn==1.2.2 matplotlib==3.7.1
```

### ESP32 Arduino 라이브러리

```
- ESP32 Board Package (by Espressif) 2.0.x
- WiFi.h, WiFiUdp.h (내장)
- driver/i2s.h (내장)
```

---

## 🚀 실행 방법

### STEP 1: 데이터 수집

```bash
python touch_classification/1_collect.py
```

- ESP32 2개 연결 후 GUI에서 클래스별 데이터 수집
- 자동 감지 방식 → 실시간 추론과 완전히 동일한 조건 유지
- 수집 데이터: `touch_dataset/` 폴더에 `.npy` 형식으로 저장
- 데이터 형식: `(3, 5120)` — 마이크 3개 × 5,120 샘플

### STEP 2: CNN 학습

```bash
python touch_classification/2_train.py
```

- `touch_dataset/` 폴더의 데이터 자동 로드
- 출력: `touch_classifier_3mic.h5`, `training_result_3mic.png`

### STEP 3: 실시간 분류

```bash
python touch_classification/3_realtime.py
```

- ESP32 2개 연결 후 실시간 터치 분류 시작
- GUI로 파형 + 분류 결과 + Class Probability 확인

### 2D PAT 위치 추정

```bash
python touch_location/2d_location.py
```

- 캘리브레이션 5개 지점 탭핑 후 위치 추정 시작

### 캡슐 효과 실험 (거리별 RMS)

```bash
python capsule_test/capsule_test.py
```

- PHASE A: 캡슐 없이 거리별 RMS 측정
- PHASE B: 같은 마이크에 병뚜껑 씌우고 동일 조건 반복

### 캡슐 효과 실험 (주파수 응답 분석)

```bash
# Chirp 신호 생성 (20Hz ~ 5kHz, 10초)
python capsule_test/chirp_frequency_test.py --make-chirp

# 스마트폰으로 chirp WAV 재생하면서 동시 녹음
python capsule_test/chirp_frequency_test.py --record

# 주파수 응답 분석 + 그래프 저장
python capsule_test/chirp_frequency_test.py --analyze
```

---

## 📊 실험 결과 요약

| 실험 | 결과 |
|------|------|
| 터치 분류 (CNN, 3mic) | 검증 정확도 **97.5%** |
| 캡슐 효과 고주파 (1k~5kHz) | RMS **+82.1%** 향상 |
| 캡슐 효과 저주파 (20~200Hz) | RMS **-10.1%** 감소 |
| 2D PAT 위치 추정 | 코드 구현 완료, 감쇠 파라미터 a=0.067 (논문 기대값 ~1.5), 정밀 추정 미달성 |

---

## 🔑 핵심 교훈

> **학습 조건 = 실시간 조건**
>
> 데이터 수집 시 버퍼 슬라이싱 방식이 실시간 추론과 완전히 동일해야 합니다.
> 다를 경우 CNN이 터치 종류가 아닌 신호 위치를 학습하게 되어 오분류가 발생합니다.
> → 자동 감지 후 현재 버퍼 그대로 저장하는 방식으로 해결

---

## 📝 참고 논문

> Min Jin Yang, Kyungseo Park, Won Dong Kim, Jung Kim
> *"Robotic Skin Mimicking Human Skin Layer and Pacinian Corpuscle for Social Interaction"*
> IEEE/ASME Transactions on Mechatronics, Vol.29, No.4, August 2024

---

## 👤 개발자

경북대학교 전자공학부 정양수
GitHub: [Criss-J](https://github.com/Criss-J)
