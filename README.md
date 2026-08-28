# Voice Recognition & Verification — tiếng Việt

Nhận dạng và xác thực người nói (speaker recognition / verification) cho tiếng Việt, cùng
pipeline huấn luyện, đánh giá và các thành phần phụ trợ (chống giả mạo, ASR, phân tích chất lượng audio).

Model tốt nhất: **ECAPA-TDNN fine-tune trên VoxVietnam** — EER **5,63%**, Top-1 **87,81%**,
Top-5 **92,49%** (gallery 146 speaker), so với 7,75% / 84,39% / 90,32% của checkpoint gốc chưa fine-tune.

> Repo này **chỉ chứa phần voice**. Phần đọc khẩu hình (lip-reading / video) không được đưa vào.

## Bắt đầu nhanh — trích embedding từ 1 file audio

```bash
cd deliverables/asv_speechbrain_v1
pip install -r requirements.txt
python embed.py audio.wav
```

```python
from asv_infer import SpeakerVerifier

asv = SpeakerVerifier()
vec = asv.embed("audio.wav")                 # vector 192 chiều, chuẩn hoá L2
ref = asv.enroll(["a1.wav", "a2.wav", "a3.wav"])
r   = asv.verify(ref, "test.wav")            # r.score, r.accepted
```

Package inference chạy độc lập, **chỉ cần `torch` + `torchaudio`** (kiến trúc ECAPA-TDNN
được vendored sẵn, không phải cài `speechbrain`). Xem
[deliverables/asv_speechbrain_v1/README.md](deliverables/asv_speechbrain_v1/README.md)
để biết cách chọn ngưỡng và các giới hạn trước khi triển khai.

**Trọng số model (83 MB) không nằm trong repo** vì git không phù hợp cho file nhị phân lớn.
Xem phần [Lấy model weights](#lấy-model-weights).

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `deliverables/asv_speechbrain_v1/` | **Package inference hoàn chỉnh** — API + ngưỡng + kiến trúc vendored |
| `scripts/finetune_wespeaker/` | Pipeline huấn luyện & đánh giá (65 file): fine-tune, benchmark, augmentation, chia dữ liệu |
| `src/ASV_System/` | Xác thực người nói: đăng ký, so khớp, lưu DB |
| `src/CM_System/` | Chống giả mạo (AASIST) — phát hiện replay / giọng tổng hợp |
| `src/AudioAnalysis_System/` | Chất lượng audio, phát hiện replay, phát hiện giọng AI |
| `src/Challenge_System/` | Sinh câu thử thách, ASR, khớp nội dung đọc |
| `src/Fusion_System/` | Hợp nhất nhiều điểm số thành quyết định cuối |
| `src/sonic_cipher/` | Pipeline tổng, cấu hình, ASR tiếng Việt (Sherpa) |
| `api/` | REST + WebSocket API (FastAPI) |
| `docs/training_logs/` | Log huấn luyện dạng CSV + biểu đồ của mọi lần thử |

## Kết quả

Đo trên **VoxVietnam test split** — speaker không trùng tập train (đã kiểm chứng 0 giao nhau).
Giao thức nhận dạng 1:N: đăng ký bằng 3 utterance (trung bình embedding), phần còn lại làm truy vấn,
cosine similarity. Gallery 146 speaker / 1.838 truy vấn.

### Model gốc, chưa fine-tune (zero-shot)

| Model | Toolkit | Params | Top-1 | Top-5 | EER |
|---|---|---|---|---|---|
| CAM++-LM | WeSpeaker | 7,18M | **85,75%** | 90,42% | 8,41% |
| ResNet34 | WeSpeaker | 6,63M | 85,20% | 90,37% | 7,86% |
| ReDimNet2-B6-LM | WeSpeaker | 12,4M | 84,00% | **90,48%** | 8,67% |
| ECAPA-TDNN | SpeechBrain | 20,8M | 84,39% | 90,32% | **7,75%** |
| ResNet293-LM | WeSpeaker | 28,6M | 84,55% | 89,88% | 8,26% |
| ResNet34-LM | WeSpeaker | 6,63M | 83,13% | 89,72% | 8,34% |
| SimAM_ResNet34 | WeSpeaker | 25,2M | 82,81% | 89,45% | 9,30% |

Cả 7 model chụm trong khoảng Top-5 89,45–90,48% dù số tham số chênh nhau 4,3 lần.
**Model lớn hơn không tốt hơn** trên tiếng Việt.

### Sau khi fine-tune

| Cấu hình | Dữ liệu train | Top-1 | Top-5 | EER | Không chệch? |
|---|---|---|---|---|---|
| **ECAPA-TDNN** | VoxVietnam 954 | 87,81% | 92,49% | **5,63%** | ✅ |
| ECAPA-TDNN | + VoxCeleb1 | 87,54% | 91,95% | 6,00% | ✅ |
| ResNet293 | VoxVietnam 954 | 87,76% | 92,76% | 5,76% | ❌ |
| ResNet34 + CosFace | + VoxCeleb1 | 88,14% | 92,82% | 6,09% | ❌ |
| ResNet34 + CosFace | VoxVietnam 954 | 87,65% | 92,60% | 6,40% | ✅ |
| ReDimNet2 | + VoxCeleb1 | 85,64% | 90,81% | 8,27% | ❌ |

Cột cuối: các lần thử ban đầu chọn checkpoint bằng **chính tập test** rồi báo cáo trên đó
(selection bias, số bị thổi phồng ~0,3–0,5 điểm). Từ v10 trở đi có **validation set riêng**
(75 speaker, tách rời tập test) nên số liệu không chệch.

## Ba phát hiện chính

**1. Thêm dữ liệu tiếng Anh làm model kém đi.** Xác nhận độc lập trên hai họ kiến trúc:

| Kiến trúc | Chỉ VoxVietnam | + VoxCeleb1 (1.211 spk) |
|---|---|---|
| ECAPA-TDNN | **92,49%** / EER 5,63% | 91,95% / EER 6,00% |
| ResNet34 | **92,60%** / EER 6,40% | 91,35% / EER 6,97% |

**2. Thêm speaker cùng miền dữ liệu đã bão hoà.** Đường cong theo số speaker VoxVietnam:
91 → 413 → 814 → 954 cho Top-5 90,04% → 91,84% → 92,55% → 92,55%. Từ 814 lên 954 **không cải thiện gì**.

**3. Thứ hạng benchmark VoxCeleb không chuyển giao sang tiếng Việt.** ReDimNet2, SimAM_ResNet34
và ResNet293 đều mạnh hơn ResNet34 trên VoxCeleb nhưng **kém hơn hoặc ngang** trên VoxVietnam.

## Huấn luyện lại

Pipeline trong `scripts/finetune_wespeaker/`. Cần chuẩn bị dữ liệu (VoxVietnam) và corpus
augmentation (MUSAN, RIRS_NOISES) — xem đường dẫn cấu hình ở đầu mỗi script.

```bash
python scripts/finetune_wespeaker/finetune_speechbrain_ecapa_v1.py   # cấu hình tốt nhất
python scripts/finetune_wespeaker/baseline_all_pretrained.py         # benchmark model gốc
python scripts/finetune_wespeaker/export_training_logs.py            # log -> CSV
python scripts/finetune_wespeaker/plot_training_curves.py            # CSV -> biểu đồ
```

Cấu hình tốt nhất: ECAPA-TDNN khởi tạo từ `speechbrain/spkrec-ecapa-voxceleb`, loss CosFace
(margin 0,25 · scale 32), augmentation MUSAN + RIRS + speed perturb, Adam lr 1e-5 (backbone) /
1e-3 (head), early stopping theo validation.

> Các script hiện còn **đường dẫn tuyệt đối** (`C:/Lily/voiceKYC/...`) ở đầu file — cần sửa
> theo máy của bạn trước khi chạy.

## Lấy model weights

Trọng số fine-tune (`model.pt`, 83 MB) không commit vào git. Cách lấy:

1. Tải từ mục **Releases** của repo (nếu đã đăng), hoặc
2. Huấn luyện lại bằng `scripts/finetune_wespeaker/finetune_speechbrain_ecapa_v1.py`

Đặt file vào `deliverables/asv_speechbrain_v1/model.pt`.

## Giới hạn

- **Chỉ kiểm chứng trên VoxVietnam** (YouTube tiếng Việt). Chưa đo trên audio KYC thật
  qua điện thoại / phòng ồn — hiệu năng thực tế có thể thấp hơn đáng kể.
- Model ASV **không phải anti-spoofing**. Nó chỉ so giọng, không phát hiện phát lại băng ghi
  hay giọng tổng hợp. `src/CM_System` (AASIST) là thành phần riêng cho việc đó.
- Ở ngưỡng bảo mật cao (FAR ≤ 0,1%), **28% người dùng thật bị từ chối**. Cần luồng dự phòng.
- `run_kyc_verification` trong `src/sonic_cipher/pipeline.py` cần module video (không có trong
  repo này). Dùng `run_voice_verification` cho luồng chỉ-giọng.

## Giấy phép & ghi công

- Kiến trúc ECAPA-TDNN và frontend Fbank vendored từ [SpeechBrain](https://github.com/speechbrain/speechbrain) (Apache-2.0)
- Kiến trúc ResNet / CAM++ / ReDimNet2 / SimAM vendored từ [WeSpeaker](https://github.com/wenet-e2e/wespeaker) (Apache-2.0)
- AASIST cho chống giả mạo · Sherpa-ONNX cho ASR tiếng Việt
- Dữ liệu: VoxVietnam, VoxCeleb1, Common Voice, MUSAN, RIRS_NOISES
