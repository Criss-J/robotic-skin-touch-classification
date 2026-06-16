"""
Robotic Skin - CNN 학습 스크립트 (3 클래스)
============================================
클래스: idle / scratch / tap

실행:
    python 2_train_cnn.py
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
DATA_DIR   = "touch_dataset"
MODEL_PATH = "touch_classifier_3mic.h5"
RESULT_IMG = "training_result_3mic.png"

SAMPLE_RATE = 10000
WINDOW_SIZE = 5120

TOUCH_CLASSES = ['idle', 'scratch', 'tap']
N_CLASSES     = len(TOUCH_CLASSES)

EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 5e-5

CLASS_WEIGHT = {i: 1.0 for i in range(N_CLASSES)}


# ──────────────────────────────────────────
# Feature Map 생성
# ──────────────────────────────────────────
def make_feature_maps(audio_3ch):
    """
    3채널 마이크 → Feature Map 생성 (논문 방식)
      Intensity Map: 3개 마이크 시간별 RMS → (n_frames, 3)
      Spectrogram:   가장 강한 마이크 STFT  → (n_frames, 64)

    핵심: 신호가 버퍼 어디에 있든 피크 구간을 중심으로 정렬
    → 학습/실시간 조건 완전히 동일하게!
    """
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

    # 가장 강한 마이크로 스펙트로그램 계산
    stds = [mic1.std(), mic2.std(), mic3.std()]
    strongest = [mic1, mic2, mic3][np.argmax(stds)]

    # 피크 구간 찾기 → 신호 중심 정렬
    win = 1024  # 피크 탐색 윈도우
    best_start = 0
    best_rms   = 0.0
    for i in range(0, len(strongest) - win, 128):
        r = np.sqrt(np.mean(strongest[i:i+win]**2))
        if r > best_rms:
            best_rms   = r
            best_start = i

    # 피크 중심으로 WINDOW_SIZE 구간 추출
    center   = best_start + win // 2
    half     = WINDOW_SIZE // 2
    start    = max(0, center - half)
    end      = start + WINDOW_SIZE
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

    return intensity, log_Sxx


# ──────────────────────────────────────────
# 데이터셋 로드
# ──────────────────────────────────────────
def load_dataset():
    X_int, X_spec, y = [], [], []

    print("\n📂 데이터셋 로딩 중...")
    for label_idx, cls in enumerate(TOUCH_CLASSES):
        path  = os.path.join(DATA_DIR, cls)
        if not os.path.exists(path):
            print(f"  ⚠️  [{cls}] 폴더 없음 - 건너뜀")
            continue

        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        for fname in files:
            data = np.load(os.path.join(path, fname))

            if data.ndim == 1:
                data = np.stack([data, data])
            if data.shape[1] < WINDOW_SIZE:
                pad  = WINDOW_SIZE - data.shape[1]
                data = np.pad(data, ((0,0),(0,pad)))
            else:
                data = data[:, :WINDOW_SIZE]

            peak = np.abs(data).max()
            if peak > 0:
                data = data / peak

            intensity, spec = make_feature_maps(data)
            X_int.append(intensity)
            X_spec.append(spec)
            y.append(label_idx)

        print(f"  ✅ [{cls:10s}] {len(files)}개 로드")

    print(f"\n  총 샘플: {len(y)}개")
    X_int  = np.array(X_int)[..., np.newaxis]
    X_spec = np.array(X_spec)[..., np.newaxis]
    y      = np.array(y)
    return X_int, X_spec, y


# ──────────────────────────────────────────
# 원본 데이터 로드 (Time Shift용)
# ──────────────────────────────────────────
def load_raw_dataset():
    raw_list = []
    y_list   = []

    for label_idx, cls in enumerate(TOUCH_CLASSES):
        path  = os.path.join(DATA_DIR, cls)
        if not os.path.exists(path):
            continue
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        for fname in sorted(files):
            data = np.load(os.path.join(path, fname))
            if data.ndim == 1:
                data = np.stack([data, data])
            if data.shape[1] < WINDOW_SIZE:
                data = np.pad(data, ((0,0),(0,WINDOW_SIZE-data.shape[1])))
            else:
                data = data[:, :WINDOW_SIZE]
            raw_list.append(data)
            y_list.append(label_idx)

    return raw_list, np.array(y_list)


# ──────────────────────────────────────────
# Time Shift
# ──────────────────────────────────────────
def time_shift_sample(data_2ch, max_shift=2000):
    shift  = np.random.randint(-max_shift, max_shift)
    result = np.zeros_like(data_2ch)
    if shift > 0:
        result[:, shift:] = data_2ch[:, :-shift]
    elif shift < 0:
        result[:, :shift] = data_2ch[:, -shift:]
    else:
        result = data_2ch.copy()
    return result


# ──────────────────────────────────────────
# 데이터 증강
# ──────────────────────────────────────────
def augment(X_int, X_spec, y, raw_data_list, factor=5):
    print(f"\n🔀 데이터 증강 (×{factor}) - Time Shift + Noise...")

    aug_int  = [X_int]
    aug_spec = [X_spec]
    aug_y    = [y]
    noise    = 0.005

    for _ in range(factor - 1):
        new_int  = []
        new_spec = []

        for raw in raw_data_list:
            shifted = time_shift_sample(raw, max_shift=2000)
            peak = np.abs(shifted).max()
            if peak > 0:
                shifted = shifted / peak

            intensity, spec = make_feature_maps(shifted)
            new_int.append(intensity)
            new_spec.append(spec)

        new_int  = np.array(new_int)[..., np.newaxis]
        new_spec = np.array(new_spec)[..., np.newaxis]
        new_int  = new_int  + np.random.normal(0, noise, new_int.shape)
        new_spec = new_spec + np.random.normal(0, noise, new_spec.shape)

        aug_int.append(new_int)
        aug_spec.append(new_spec)
        aug_y.append(y)

    X_int_aug  = np.concatenate(aug_int,  axis=0)
    X_spec_aug = np.concatenate(aug_spec, axis=0)
    y_aug      = np.concatenate(aug_y,    axis=0)

    idx = np.random.permutation(len(y_aug))
    print(f"  증강 후 총 샘플: {len(y_aug)}개")
    return X_int_aug[idx], X_spec_aug[idx], y_aug[idx]


# ──────────────────────────────────────────
# 모델
# ──────────────────────────────────────────
def build_model(int_shape, spec_shape, n_classes=N_CLASSES):
    inp1 = layers.Input(shape=int_shape, name='intensity')
    x1   = layers.Conv2D(32, (5,2), padding='same', activation='relu')(inp1)
    x1   = layers.MaxPooling2D((2,1))(x1)
    x1   = layers.Conv2D(16, (5,2), padding='same', activation='relu')(x1)
    x1   = layers.MaxPooling2D((2,1))(x1)
    x1   = layers.Flatten()(x1)

    inp2 = layers.Input(shape=spec_shape, name='spectrogram')
    x2   = layers.Conv2D(32, (5,5), padding='same', activation='relu')(inp2)
    x2   = layers.MaxPooling2D(2)(x2)
    x2   = layers.Conv2D(16, (5,5), padding='same', activation='relu')(x2)
    x2   = layers.MaxPooling2D(2)(x2)
    x2   = layers.Flatten()(x2)

    merged = layers.Concatenate()([x1, x2])
    x      = layers.Dense(256, activation='relu')(merged)
    x      = layers.Dropout(0.3)(x)
    x      = layers.Dense(128, activation='relu')(x)
    x      = layers.Dropout(0.2)(x)
    out    = layers.Dense(n_classes, activation='softmax')(x)

    return Model(inputs=[inp1, inp2], outputs=out, name='TouchClassifier')


# ──────────────────────────────────────────
# 결과 시각화
# ──────────────────────────────────────────
def plot_results(history, y_test, y_pred):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 6), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0d1117')
    ax1.plot(history.history['accuracy'],     color='#3498db', lw=2, label='Train Acc')
    ax1.plot(history.history['val_accuracy'], color='#2ecc71', lw=2, label='Val Acc')
    ax1.plot(history.history['loss'],         color='#e74c3c', lw=2, linestyle='--', label='Train Loss')
    ax1.plot(history.history['val_loss'],     color='#f39c12', lw=2, linestyle='--', label='Val Loss')
    ax1.set_title('Training Curve', color='white', fontsize=13)
    ax1.set_xlabel('Epoch', color='#aaa')
    ax1.legend(facecolor='#16213e', edgecolor='#444', labelcolor='white')
    ax1.tick_params(colors='#aaa')
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[1])
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax2.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xticks(range(N_CLASSES))
    ax2.set_yticks(range(N_CLASSES))
    ax2.set_xticklabels(TOUCH_CLASSES, rotation=45, ha='right', color='#ccc', fontsize=12)
    ax2.set_yticklabels(TOUCH_CLASSES, color='#ccc', fontsize=12)
    ax2.set_title('Confusion Matrix', color='white', fontsize=13)
    ax2.set_xlabel('Predicted', color='#aaa')
    ax2.set_ylabel('Actual', color='#aaa')

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm_norm[i, j]
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                     color='white' if val > 0.5 else '#aaa', fontsize=11)

    plt.colorbar(im, ax=ax2)
    plt.savefig(RESULT_IMG, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f"\n📊 결과 이미지 저장: {RESULT_IMG}")
    plt.show()


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  Robotic Skin CNN 학습 (3 클래스)")
    print("  idle / scratch / tap")
    print("=" * 50)

    # 1. 데이터 로드
    X_int, X_spec, y = load_dataset()
    if len(y) == 0:
        print("\n❌ 데이터 없음.")
        exit()

    # 2. 원본 로드
    raw_list, _ = load_raw_dataset()

    # 3. 분할
    indices = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42)
    tr_idx, val_idx = train_test_split(
        tr_idx, test_size=0.1, stratify=y[tr_idx], random_state=42)

    X_int_tr  = X_int[tr_idx];  X_int_val  = X_int[val_idx];  X_int_te  = X_int[te_idx]
    X_spec_tr = X_spec[tr_idx]; X_spec_val = X_spec[val_idx]; X_spec_te = X_spec[te_idx]
    y_tr      = y[tr_idx];      y_val      = y[val_idx];      y_te      = y[te_idx]
    raw_tr    = [raw_list[i] for i in tr_idx]

    # 4. 증강
    X_int_tr, X_spec_tr, y_tr = augment(
        X_int_tr, X_spec_tr, y_tr, raw_tr, factor=5)

    # 5. 원핫 인코딩
    y_tr_cat  = tf.keras.utils.to_categorical(y_tr,  N_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, N_CLASSES)

    # 6. 모델
    model = build_model(X_int.shape[1:], X_spec.shape[1:])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 7. 학습
    print(f"\n🚀 학습 시작 (epochs={EPOCHS}, batch={BATCH_SIZE})")
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                                  save_best_only=True, verbose=1)
    ]

    history = model.fit(
        [X_int_tr, X_spec_tr], y_tr_cat,
        validation_data=([X_int_val, X_spec_val], y_val_cat),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cb_list,
        class_weight=CLASS_WEIGHT
    )

    # 8. 평가
    print("\n📊 테스트 평가...")
    y_pred = np.argmax(model.predict([X_int_te, X_spec_te]), axis=1)
    acc    = np.mean(y_pred == y_te)
    print(f"\n✅ 테스트 정확도: {acc*100:.1f}%")
    print("\n분류 보고서:")
    print(classification_report(y_te, y_pred, target_names=TOUCH_CLASSES))

    plot_results(history, y_te, y_pred)
    print(f"\n✅ 모델 저장 완료: {MODEL_PATH}")
