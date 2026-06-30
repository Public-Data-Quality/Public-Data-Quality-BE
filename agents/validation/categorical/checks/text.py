from __future__ import annotations

import re
from typing import Any


def clean_reason_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""

    lowered = text.lower()
    if any(token in lowered for token in ("lorem", "asdf", "n/a", "unknown")):
        return ""
    if re.search(r"[A-Za-z]{8,}", text):
        return ""

    words = [word for word in re.split(r"\s+", text) if word]
    if len(words) >= 3:
        suspicious_endings = (
            "하세요",
            "해세요",
            "습니다",
            "입니다",
            "적절",
            "일치",
            "통일",
            "권장",
            "권합니다",
        )
        valid_endings = sum(1 for word in words if word.endswith(suspicious_endings))
        if valid_endings == 0:
            return ""

    return text


def is_specific_out_of_domain_reason(reason: str) -> bool:
    cleaned = clean_reason_text(reason)
    if not cleaned:
        return False

    generic_reasons = {
        "다른 기관명",
        "다른 값",
        "상이한 값",
        "형태가 다름",
        "표현이 다름",
        "이름이 다름",
        "기관명이 다름",
        "다른 명칭",
    }
    if cleaned in generic_reasons:
        return False

    specific_markers = (
        "도메인",
        "범주",
        "체계",
        "분류",
        "tax",
        "taxonomy",
        "기관명이 아님",
        "부서명이 아님",
        "경찰서명이 아님",
        "학교명이 아님",
        "주소가 아님",
        "날짜가 아님",
        "숫자가 아님",
        "코드가 아님",
        "서로 다른 유형",
    )
    return any(marker in cleaned for marker in specific_markers)


def is_specific_normalization_reason(reason: str) -> bool:
    cleaned = clean_reason_text(reason)
    if not cleaned:
        return False

    weak_markers = (
        "표준화",
        "통일",
        "일관",
        "띄어쓰기",
        "공백",
        "형식",
        "표기",
        "권장",
        "적절",
    )
    strong_markers = (
        "오타",
        "오기",
        "약어",
        "축약",
        "코드값",
        "허용값",
        "공식",
        "표준명",
        "동일 대상을 다른 방식",
        "동일한 의미",
    )
    return any(marker in cleaned for marker in strong_markers) and not (
        any(marker in cleaned for marker in weak_markers)
        and not any(marker in cleaned for marker in strong_markers)
    )


def is_specific_row_context_reason(reason: str) -> bool:
    cleaned = clean_reason_text(reason)
    if not cleaned:
        return False
    weak_markers = (
        "문맥",
        "일치하지",
        "불일치",
        "다른 컬럼",
        "다른 값",
        "상이",
        "의심",
        "가능",
    )
    strong_markers = (
        "셰필드",
        "외국",
        "해외",
        "광역",
        "시도",
        "지역명",
        "주소",
        "괄호",
        "닫히지",
        "잘림",
        "누락",
        "경기도",
        "전라남도",
        "인천광역시",
        "서울특별시",
        "부산광역시",
        "대구광역시",
        "광주광역시",
        "대전광역시",
        "울산광역시",
        "세종특별자치시",
        "제주특별자치도",
    )
    if any(marker in cleaned for marker in strong_markers):
        return True
    return not any(marker in cleaned for marker in weak_markers)


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def is_numeric_like_value(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text))


def looks_malformed_text_value(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if "�" in text:
        return True
    if re.search(r"[ÃÂêëìíîïðñòóôõöøùúûüýþÿ]{2,}", text):
        return True
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ]{2,}", text):
        return True
    if re.search(r"[가-힣A-Za-z0-9)][ㄱ-ㅎㅏ-ㅣ]$", text):
        return True
    if re.search(r"(~{2,}|/{2,}|※{2,}|□{2,}|[#@$%^*_={}|\\]{3,})", text):
        return True
    if re.search(r"[?!]{2,}", text):
        return True
    if re.search(r"[가-힣A-Za-z0-9][?！!]{1,}$", text):
        return True
    if "학꾜" in text or "주챠장" in text:
        return True
    lowered = text.lower()
    if any(token in lowered for token in ("lorem ipsum", "asdf", "unknown")):
        return True
    if re.search(r"[\U0001F300-\U0010FFFF]", text):
        return True
    if any(token in lowered for token in ("free parking", "open parking")):
        return True
    if re.search(r"\bopen\s*[✅✔☑]", lowered):
        return True
    if "오늘 날씨" in text:
        return True
    if "AI가 생성한 문장" in text or "테스트 데이터입니다" in text:
        return True
    return False


def looks_context_free_replacement_value(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return False

    replacement_phrases = (
        "현장 상황에 따라",
        "주소 확인 불가",
        "명절 기간 개방하지 않음",
        "명절 기간 중 일부",
        "폐쇄 예정 시설",
        "예약자 외 이용 불가",
        "문의 후 이용 바랍니다",
        "유료 전용 주차장으로 변경",
        "주차 가능 여부는",
        "개방하지 않음",
        "이용 불가",
        "확인 불가",
    )
    return any(phrase in text for phrase in replacement_phrases)


def looks_non_name_value(value: str) -> bool:
    return looks_context_free_replacement_value(value)
