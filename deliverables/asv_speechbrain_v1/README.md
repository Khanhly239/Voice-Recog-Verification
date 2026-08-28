# ASV tiếng Việt — ECAPA-TDNN fine-tune trên VoxVietnam

Model nhận dạng/xác thực người nói (Automatic Speaker Verification) cho tiếng Việt.
Fine-tune từ `speechbrain/spkrec-ecapa-voxceleb` trên VoxVietnam.

## Cài đặt

```bash
pip install -r requirements.txt
```

Cần Python >= 3.9. Không cần GPU (CPU chạy được, ~0,3s/utterance), có GPU thì nhanh hơn.

**Chỉ cần `torch` + `torchaudio`** — không phải cài `speechbrain`. Code kiến trúc ECAPA-TDNN
và frontend Fbank được vendored sẵn trong `_sb_vendored.py` (trích từ SpeechBrain, Apache-2.0).

## Dùng nhanh

```python
from asv_infer import SpeakerVerifier

asv = SpeakerVerifier()

# Đăng ký — nên dùng >= 3 mẫu của cùng một người
ref = asv.enroll(["a1.wav", "a2.wav", "a3.wav"])

# Xác thực 1:1
r = asv.verify(ref, "test.wav")
print(r.score, r.accepted)        # 0.8810  True

# Nhận dạng 1:N
db = {"nguyen_van_a": ref, "tran_thi_b": asv.enroll(["b1.wav", "b2.wav", "b3.wav"])}
print(asv.identify("test.wav", db, top_k=5))
```

Thử nhanh từ dòng lệnh:

```bash
python asv_infer.py ref1.wav ref2.wav ref3.wav test.wav
```

Đầu vào nhận: đường dẫn file (mọi định dạng torchaudio đọc được — wav/mp3/flac/m4a),
hoặc `np.ndarray` / `torch.Tensor` (16 kHz mono; int16 tự động quy về [-1, 1]).
Audio khác 16 kHz được resample tự động. Yêu cầu tối thiểu 0,5 giây.

## Chọn ngưỡng — đọc kỹ phần này

`verify()` so cosine similarity với một ngưỡng. Ngưỡng quyết định đánh đổi giữa
**FAR** (chấp nhận sai người lạ — rủi ro an ninh) và **FRR** (từ chối sai người thật — bất tiện).

| Điểm hoạt động | Ngưỡng | FAR | FRR | Dùng khi |
|---|---|---|---|---|
| `EER` | 0,2208 | 5,12% | 5,11% | **Chỉ để báo cáo, KHÔNG triển khai** |
| `FAR<=5%` | 0,2228 | 5,00% | 5,11% | Rủi ro thấp, ưu tiên tiện lợi |
| **`FAR<=1%`** (mặc định) | **0,3602** | **1,00%** | **10,76%** | Cân bằng cho KYC thông thường |
| `FAR<=0.1%` | 0,5488 | 0,10% | 28,12% | Rủi ro cao (giao dịch lớn) |

```python
asv = SpeakerVerifier(operating_point="FAR<=0.1%")   # hoặc đổi lúc chạy:
asv.set_operating_point("FAR<=1%")
```

**Cảnh báo về FRR:** ở mức bảo mật cao `FAR<=0.1%`, **28% người dùng thật bị từ chối**.
Đây là hạn chế thật của model, không phải lỗi cấu hình. Nếu hệ thống cần cả bảo mật cao
lẫn trải nghiệm tốt, phải có luồng dự phòng (thử lại, xác thực kênh khác) chứ không
thể chỉ dựa vào ngưỡng.

**Với `identify()` (1:N):** điểm cao nhất KHÔNG đảm bảo đúng người. Nếu người lạ có thể
xuất hiện (hệ thống mở), phải kiểm tra thêm `score >= asv.threshold` trước khi chấp nhận.

## Hiệu năng đo được

Tập test: **VoxVietnam test split** — speaker hoàn toàn không trùng với tập train (đã kiểm chứng).
Giao thức: đăng ký bằng 3 utterance (trung bình embedding), phần còn lại làm truy vấn, cosine similarity.

| Gallery | Top-1 | Top-5 | EER |
|---|---|---|---|
| 146 speaker / 1.838 truy vấn | 87,81% | 92,49% | **5,63%** |
| 75 speaker / 939 truy vấn | 87,22% | 94,36% | **5,11%** |

Gallery càng nhiều người thì càng khó (nhiều nhiễu hơn) — nên luôn đọc kèm số speaker.

So với chính checkpoint gốc chưa fine-tune (`spkrec-ecapa-voxceleb`), trên gallery 146:

| | Top-1 | Top-5 | EER |
|---|---|---|---|
| Gốc (zero-shot) | 84,39% | 90,32% | 7,75% |
| **Sau fine-tune** | **87,81%** | **92,49%** | **5,63%** |
| Cải thiện | +3,42 đ | +2,17 đ | **−27%** |

Các con số trên được chọn model bằng **tập validation riêng biệt** (75 speaker, tách rời tập
test), nên là ước lượng không chệch.

## Giới hạn cần biết trước khi triển khai

1. **Chỉ kiểm chứng trên VoxVietnam** — dữ liệu YouTube tiếng Việt. Chưa đo trên audio KYC
   thật (điện thoại, phòng ồn, mic kém). Hiệu năng thực tế có thể thấp hơn đáng kể.
2. **Không phải anti-spoofing.** Model chỉ so giọng, **không phát hiện phát lại băng ghi,
   giọng tổng hợp (TTS), hay voice conversion.** Hệ thống KYC thật bắt buộc phải có
   thành phần chống giả mạo riêng.
3. **Chưa đạt mục tiêu ban đầu** của dự án (Top-5 > 95%, Top-1 > 90%) — còn thiếu ~2,5 và ~2,2 điểm.
4. **Utterance ngắn kém chính xác hơn.** Benchmark dùng audio độ dài tự nhiên (trung bình ~7,7s).
   Dưới 2 giây thì độ tin cậy giảm.
5. **Nhạy với chất lượng đăng ký.** Đăng ký bằng 1 mẫu kém hơn rõ rệt so với 3 mẫu.

## Cấu hình huấn luyện (để tái lập)

| | |
|---|---|
| Kiến trúc | ECAPA-TDNN, channels [1024×4, 3072], embedding 192 chiều |
| Khởi tạo | `speechbrain/spkrec-ecapa-voxceleb` (`embedding_model.ckpt`) |
| Frontend | SpeechBrain `Fbank(n_mels=80)` + `InputNormalization(norm_type='sentence', std_norm=False)` |
| Dữ liệu train | VoxVietnam 954 speaker / 19.602 utterance (tiếng Việt) |
| Loss | CosFace (AM-Softmax), margin 0,25, scale 32 |
| Augmentation | MUSAN (nhiễu/nhạc/babble) + RIRS (vọng phòng) + speed perturb 0,9-1,1× |
| Optimizer | Adam, lr backbone 1e-5 / head 1e-3, weight decay 1e-4 |
| Batch | 4 · Epoch tốt nhất: 30/45 (early stopping theo validation) |

**Frontend phải khớp chính xác.** Dùng kaldi fbank (`torchaudio.compliance.kaldi`) thay vì
SpeechBrain Fbank sẽ cho embedding sai hoàn toàn mà không báo lỗi. `asv_infer.py` đã dùng đúng
frontend — đã kiểm chứng embedding khớp bit-exact với pipeline đánh giá (cosine = 1,000000).

## Về code vendored

`_sb_vendored.py` chứa các thành phần trích **nguyên văn** từ SpeechBrain (Apache-2.0):
`ECAPA_TDNN` và các block con, `Fbank`/`STFT`/`Filterbank`, `InputNormalization`, cùng các
lớp nền `Conv1d`/`Linear`/`BatchNorm1d`. Mục đích là bỏ phụ thuộc vào cả thư viện speechbrain.
Các phần không tham gia tính toán (checkpoint hooks, logger, distributed) được stub lại.

Đã kiểm chứng **khớp bit-exact** với thư viện gốc — `max|diff| = 0.000e+00` ở cả ba tầng
(frontend, model, và toàn trình API). Chạy lại kiểm chứng bất cứ lúc nào:

```bash
pip install speechbrain      # chỉ để đối chiếu
python verify_vendored.py
```

Bản quyền phần vendored thuộc SpeechBrain contributors.

Một phát hiện đáng lưu ý khi huấn luyện: **thêm dữ liệu tiếng Anh (VoxCeleb1, 1.211 speaker)
làm model kém đi** (Top-5 92,49% → 91,95%, EER 5,63% → 6,00%). Đã xác nhận độc lập trên cả
kiến trúc WeSpeaker ResNet34. Vì vậy bản phát hành này chỉ train trên tiếng Việt.

## Nội dung thư mục

```
model.pt             83 MB — trọng số fine-tune + metadata
_sb_vendored.py      kiến trúc ECAPA-TDNN + frontend (vendored từ SpeechBrain, Apache-2.0)
asv_infer.py         API inference
thresholds.json      ngưỡng tại các điểm hoạt động
verify_vendored.py   kiểm chứng vendored khớp bit-exact với speechbrain gốc
requirements.txt
README.md
```
