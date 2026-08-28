import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    dbname: str = os.getenv("DB_NAME", "sonic")
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "root")
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))

    def as_dict(self) -> dict:
        return {
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
        }


@dataclass
class ASVConfig:
    # Pretrained WeSpeaker ResNet34 (VoxCeleb) ONNX model. Thay cho SpeechBrain ECAPA-TDNN sau khi
    # so sánh trực tiếp trên VIVOS (65 speaker) và VoxVietnam (146 speaker) cho thấy ResNet34 thắng
    # cả hai: EER VIVOS 2.19%->0.63%, EER VoxVietnam 6.84%->4.76%, Top-1/Top-5 cũng cao hơn.
    # Rỗng = tự tìm Weights/wespeaker_resnet34.onnx (đóng gói sẵn trong package).
    onnx_model_path: str = os.getenv("ASV_ONNX_MODEL_PATH", "").strip()
    embedding_dim: int = 256
    sample_rate: int = 16000
    # Ba mẫu đăng ký phải cùng một giọng: từ chối nếu cosine nhỏ nhất giữa hai mẫu bất kỳ thấp hơn ngưỡng.
    # Benchmark trên 20 người VIVOS VỚI WESPEAKER RESNET34: min-pairwise-cosine của 3-mẫu-cùng-người
    # dao động 0.53-0.83 (TB 0.70), còn 3-mẫu-bị-trộn-1-người-khác dao động 0.06-0.51 (TB 0.24) --
    # tách biệt rõ hơn hẳn so với SpeechBrain (từng phải hạ xuống 0.45 mà vẫn FAR~15%). Với model mới,
    # 0.55 đạt FAR=0%, FRR=5% trên cùng benchmark.
    enrollment_pairwise_cosine_min: float = 0.55
    # Ngưỡng cosine giữa embedding test và embedding đã đăng ký để coi là "verified" (legacy demo,
    # dùng bởi predict_verification() / app.py Streamlit). Suy ra từ điểm EER benchmark VoxVietnam
    # (146 speaker, giao thức sản xuất, WeSpeaker ResNet34) ở thang chuẩn hóa [0,1] = 0.69
    # -> quy đổi sang cosine thô: 2*0.69 - 1 = 0.38.
    verification_cosine_min: float = 0.38


@dataclass
class CMConfig:
    nb_samp: int = 64600


@dataclass
class ChallengeConfig:
    challenge_length: int = 6
    # 20s: challenge mặc định là dãy 6 chữ số, đọc mất ~5s nên 20s đủ rộng, đồng thời
    # thu hẹp cửa sổ để kẻ tấn công không kịp chuẩn bị/dựng bản ghi cho nội dung vừa
    # nhận. Nếu chuyển challenge về dạng câu tự do (dài hơn) thì cần nâng lại giá trị này.
    session_ttl_seconds: int = 20
    whisper_model_size: str = "base"
    whisper_language: str = "vi"
    match_threshold: float = 0.75
    # Câu tự do dễ bị ASR nhận sai từ hơn dãy số đọc rời -> ngưỡng khớp thấp hơn.
    match_threshold_sentence: float = 0.65


@dataclass
class SherpaAsrConfig:
    """Offline Vietnamese ASR (sherpa-onnx Zipformer). Empty model_dir = auto under repo sherpa/."""

    model_dir: str = os.getenv("SHERPA_MODEL_DIR", "").strip()
    num_threads: int = int(os.getenv("SHERPA_NUM_THREADS", "2"))


@dataclass
class AudioQualityConfig:
    min_snr_db: float = 15.0
    max_silence_ratio: float = 0.60
    min_duration_sec: float = 2.0
    max_duration_sec: float = 30.0
    max_clipping_ratio: float = 0.01
    min_sample_rate: int = 16000


@dataclass
class ReplayDetectorConfig:
    high_freq_cutoff_hz: int = 14000
    high_freq_energy_threshold: float = 0.02
    spectral_flatness_threshold: float = 0.15
    score_weights: dict = field(default_factory=lambda: {
        "high_freq": 0.3,
        "spectral_mod": 0.3,
        "cepstral": 0.2,
        "reverb": 0.2,
    })


@dataclass
class AIVoiceDetectorConfig:
    jitter_threshold: float = 0.005
    shimmer_threshold: float = 0.02
    phase_discontinuity_threshold: float = 0.3
    spectral_smoothness_threshold: float = 0.8


@dataclass
class FaceLivenessConfig:
    ear_threshold: float = 0.21
    min_blinks: int = 1
    min_head_movement_deg: float = 3.0
    min_face_ratio: float = 0.05
    detection_confidence: float = 0.5


@dataclass
class LipSyncConfig:
    # "syncnet": model đã học (Video_System.sync_net), khuyến nghị. "correlation": heuristic
    # Pearson correlation cũ (Video_System.lip_sync) -- giữ lại làm fallback qua env var nếu
    # SyncNet gặp vấn đề (thiếu weight, lỗi môi trường) khi triển khai thực tế.
    backend: str = os.getenv("LIPSYNC_BACKEND", "syncnet").strip()
    correlation_threshold: float = 0.3
    fps: int = 25
    hop_length: int = 512


@dataclass
class SyncNetConfig:
    """Audio-visual sync detection bằng SyncNet (Chung & Zisserman, 2016; MIT license,
    joonson/syncnet_python) -- thay cho Pearson correlation thô sơ của LipSyncConfig.
    Phát hiện audio có KHỚP THỜI GIAN với chuyển động môi hay không (chống dub/replay),
    KHÔNG kiểm tra nội dung câu nói (đó là việc của VisemeConfig).
    """

    # Rỗng = tự tìm pretrained_models/syncnet/syncnet_v2.model (tải qua download_model.sh
    # gốc hoặc thủ công từ robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model).
    model_path: str = os.getenv("SYNCNET_MODEL_PATH", "pretrained_models/syncnet/syncnet_v2.model").strip()
    # confidence (median_dist - min_dist) không có cận trên cố định; midpoint dùng để map
    # qua sigmoid ra [0,1]. Benchmark trên 4 video full-face tự quay (điện thoại, 1 người,
    # 4 câu khác nhau) + 12 cặp imposter (audio bị trộn giữa các clip): confidence genuine
    # nằm trong [2.19, 4.51], imposter nằm trong [0.11, 1.24] -- KHÔNG chồng lấn. 1.72 là điểm
    # giữa khoảng trống này. Lưu ý: mẫu nhỏ (1 người, 4 clip) -- cần benchmark thêm với nhiều
    # người/điều kiện quay khác nhau trước khi tin tưởng hoàn toàn cho production.
    confidence_midpoint: float = 1.72


@dataclass
class VisemeConfig:
    """Khớp khẩu hình (viseme) với nội dung challenge text qua DTW.

    Benchmark trên 4 video full-face tự quay (1 người, 4 câu khác nhau) + 12 cặp
    imposter (câu sai): DTW so khớp thẳng chuỗi quan sát THÔ (theo frame) với
    chuỗi kỳ vọng (theo âm tiết) KHÔNG tách biệt được genuine/imposter dù grid-search
    1,250 tổ hợp ngưỡng z-score + chi phí thay thế (gap tốt nhất vẫn âm, mean_diff
    âm -- tệ hơn ngẫu nhiên). Nguyên nhân: chênh lệch độ dài quá lớn (100-300 frame
    quan sát vs 12-24 phần tử kỳ vọng) cho DTW quá nhiều tự do "lách" đường đi rẻ.

    Fix: nén chuỗi quan sát thành các run liên tiếp cùng nhãn (min_run_frames) trước
    khi đưa vào DTW, đồng thời tăng cost_scale để phạt nặng hơn khi sai lớp. Với
    min_run_frames=10, cost_scale=2.0: mean_diff cải thiện từ ~-0.01 lên +0.216 (11/12
    cặp imposter tách biệt hoàn toàn khỏi genuine, còn 1 cặp biên). Vẫn là cải thiện
    trên mẫu nhỏ (1 người, 4 clip) -- cần thêm dữ liệu đa dạng người nói/điều kiện
    quay để xác nhận trước khi tin tưởng hoàn toàn cho production.
    """

    min_frames: int = 5
    # Gộp run < min_run_frames vào run liền kề dài hơn (coi là nhiễu phân loại).
    min_run_frames: int = 10
    # Hệ số nhân _SUBSTITUTION_COST trong DTW -- phạt nặng hơn khi sai lớp khẩu hình.
    cost_scale: float = 2.0

    # --- 3D Mesh depth (bổ sung) ---
    # Dùng toạ độ z của FaceLandmarker (3D mesh, 478 điểm, z_span thực ~0.18 nên là 3D thật).
    # Có HAI cách dùng và benchmark cho kết luận khác nhau:
    #
    # 1) Trộn protrusion (độ chu môi ra trước) vào phân loại ROUNDED để tăng độ phân biệt
    #    khẩu hình: KHÔNG hiệu quả. Benchmark trên 4 video test_videos (genuine vs imposter,
    #    ghép chéo nội dung) sweep trọng số {0.15..0.7} x làm mượt {1,3,5,9 frame}: MỌI cấu hình
    #    3D đều KÉM hơn baseline 2D thuần (separation 2D=+0.217, 3D tốt nhất chỉ +0.068). z
    #    per-frame quá nhiễu cho phân loại rule-based. -> use_depth=False (giữ toggle để không mất
    #    code, nhưng mặc định tắt vì làm hỏng bộ phân loại 2D đang chạy tốt).
    use_depth: bool = False
    depth_protrusion_weight: float = 0.5
    #
    # 2) Biên độ chuyển động môi theo chiều sâu 3D (std tín hiệu protrusion theo thời gian) làm
    #    tín hiệu LIVENESS/chống-mặt-phẳng: CÓ hiệu quả và luôn được tính (không phụ thuộc use_depth).
    #    Video mặt thật đo được 0.043-0.15; bề mặt phẳng (ảnh/màn hình phát lại) sẽ ~0. Ngưỡng 0.02
    #    để coi là "có chuyển động môi 3D thật". Hiện chỉ báo cáo trong details (depth_amplitude,
    #    depth_liveness_ok), CHƯA hard-gate -- cần thu video tấn công mặt-phẳng thật để hiệu chỉnh,
    #    nhất quán với cách xử lý thận trọng của các ngưỡng khác.
    min_depth_amplitude: float = 0.02


@dataclass
class FusionConfig:
    hard_gate_liveness: float = 0.5
    hard_gate_audio_quality: float = 0.3
    # Test trên VIVOS (giả danh dùng đúng nội dung challenge) cho thấy cm/replay/ai_voice chấm
    # gần tối đa cho MỌI giọng người thật, nên chỉ dựa vào trọng số là chưa đủ để chặn giả danh
    # -> cần gate cứng riêng cho ASV (điểm cosine đã chuẩn hóa [0,1]).
    # Hiệu chỉnh lại theo model WeSpeaker ResNet34 (thay SpeechBrain, xem ASVConfig.onnx_model_path):
    # benchmark cùng giao thức trên VoxVietnam (146 speaker, gallery 1:N) cho điểm EER=4.76% tại
    # normalized=0.69 (so với 5.34%/0.725 của SpeechBrain) -- model mới tách bạch tốt hơn nên
    # dùng thẳng 0.69 làm hard gate.
    hard_gate_asv: float = 0.69

    # Thêm viseme_match (0.20) so với nội dung challenge; giảm bớt
    # replay/ai_voice/lipsync/cm để tổng vẫn = 1.0. Chưa hard-gate vì đây
    # là heuristic rule-based mới, chưa benchmark -- tránh rủi ro tăng FRR.
    weights: dict = field(default_factory=lambda: {
        "asv": 0.30,
        "cm": 0.20,
        "replay": 0.10,
        "ai_voice": 0.10,
        "lipsync": 0.10,
        "viseme_match": 0.20,
    })
    final_threshold: float = 0.50


@dataclass
class VoiceFusionConfig:
    """Fusion cho luồng xác thực chỉ-bằng-giọng-nói (không video/lip-sync/face liveness)."""

    hard_gate_audio_quality: float = 0.3
    # Xem giải thích ở FusionConfig.hard_gate_asv — cùng vấn đề, càng quan trọng hơn ở đây vì
    # không còn liveness/lip-sync để bù. Cùng giá trị 0.69 (WeSpeaker ResNet34, EER VoxVietnam).
    hard_gate_asv: float = 0.69

    weights: dict = field(default_factory=lambda: {
        "asv": 0.45,
        "cm": 0.25,
        "replay": 0.15,
        "ai_voice": 0.15,
    })
    final_threshold: float = 0.55


@dataclass
class KYCConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    asv: ASVConfig = field(default_factory=ASVConfig)
    cm: CMConfig = field(default_factory=CMConfig)
    challenge: ChallengeConfig = field(default_factory=ChallengeConfig)
    sherpa: SherpaAsrConfig = field(default_factory=SherpaAsrConfig)
    audio_quality: AudioQualityConfig = field(default_factory=AudioQualityConfig)
    replay: ReplayDetectorConfig = field(default_factory=ReplayDetectorConfig)
    ai_voice: AIVoiceDetectorConfig = field(default_factory=AIVoiceDetectorConfig)
    face_liveness: FaceLivenessConfig = field(default_factory=FaceLivenessConfig)
    lipsync: LipSyncConfig = field(default_factory=LipSyncConfig)
    syncnet: SyncNetConfig = field(default_factory=SyncNetConfig)
    viseme: VisemeConfig = field(default_factory=VisemeConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    voice_fusion: VoiceFusionConfig = field(default_factory=VoiceFusionConfig)
    device: str = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"


config = KYCConfig()
