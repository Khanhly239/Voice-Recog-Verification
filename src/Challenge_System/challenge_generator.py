import random
import logging
from ASV_System.db_utils import create_kyc_session
from sonic_cipher.config import config

logger = logging.getLogger(__name__)

DIGIT_WORDS_VI = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

# Câu tự do để đọc thay cho dãy số. Chọn câu ngắn, đa dạng âm vị tiếng Việt,
# không mang nội dung nhạy cảm/định danh cụ thể.
SENTENCES_VI = [
    "Tôi xin xác nhận đây là giọng nói của tôi.",
    "Hôm nay tôi thực hiện xác thực danh tính bằng giọng nói.",
    "Xin chào, tôi đang đăng nhập vào hệ thống của mình.",
    "Tôi đồng ý xác minh tài khoản này bằng giọng nói thật.",
    "Ba con mèo nhỏ đang chạy nhảy ngoài vườn hoa.",
    "Trời hôm nay nắng đẹp và có gió mát nhẹ nhàng.",
    "Con số bí mật của tôi không chia sẻ với bất kỳ ai.",
    "Việt Nam có nhiều cảnh đẹp thiên nhiên hùng vĩ.",
    "Tôi đang nói để hệ thống nhận diện giọng nói của mình.",
    "Buổi sáng tôi thường uống một cốc cà phê nóng.",
    "Chiếc xe màu đỏ đậu ngay trước cổng nhà tôi.",
    "Cuối tuần này gia đình tôi sẽ đi du lịch biển.",
    "Học sinh đang chăm chỉ ôn bài trước kỳ thi.",
    "Tôi thích đọc sách vào mỗi buổi tối trước khi ngủ.",
    "Cơn mưa rào bất ngờ đổ xuống thành phố lúc chiều.",
    "Đây là câu tôi đọc để xác thực giọng nói trực tiếp.",
    "Ngân hàng yêu cầu tôi xác thực bằng giọng nói hôm nay.",
    "Tôi luôn giữ thông tin cá nhân của mình được bảo mật.",
]


def _generate_digits(length: int | None = None) -> str:
    length = length or config.challenge.challenge_length
    digits = [str(random.randint(0, 9)) for _ in range(length)]
    return " ".join(digits)


def _generate_sentence() -> str:
    return random.choice(SENTENCES_VI)


def generate_challenge(mode: str = "digits", length: int | None = None) -> str:
    """Generate a random challenge for the user to speak aloud.

    mode: "digits" (random digit string), "sentence" (random free sentence),
    or "random" (pick one of the two with equal probability).
    """
    if mode == "digits":
        return _generate_digits(length)
    if mode == "sentence":
        return _generate_sentence()
    if mode == "random":
        return _generate_digits(length) if random.random() < 0.5 else _generate_sentence()
    raise ValueError(f"Unknown challenge mode: {mode}")


def get_digit_words_map() -> dict[str, str]:
    """Return mapping from digit character to Vietnamese word."""
    return DIGIT_WORDS_VI.copy()


def create_challenge_session(username: str, mode: str = "digits") -> dict:
    """Create a new KYC challenge session and persist it in the database."""
    challenge_text = generate_challenge(mode)
    ttl = config.challenge.session_ttl_seconds
    session_id = create_kyc_session(username, challenge_text, ttl)

    logger.info(f"Challenge session created: {session_id} mode={mode} challenge='{challenge_text}'")
    return {
        "session_id": session_id,
        "challenge_text": challenge_text,
        "expires_in_seconds": ttl,
        "challenge_type": "digits" if challenge_text.replace(" ", "").isdigit() else "sentence",
    }
