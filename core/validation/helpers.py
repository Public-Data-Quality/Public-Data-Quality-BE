from __future__ import annotations

import re
from datetime import datetime

from ..config.constants import VALIDATION_CRITERIA
from ..schema.models import StandardTerm, ValidationFinding

BOOLEAN_ALLOWED_VALUES = {"y", "n", "yes", "no", "true", "false", "0", "1", "예", "아니오", "유", "무"}
DATE_PATTERNS = ("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d", "%Y/%m/%d", "%Y-%m", "%Y%m", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S")
TIME_ORDER_TOKENS = [
    ("시작", "종료"),
    ("개시", "종료"),
    ("접수", "처리"),
    ("등록", "수정"),
    ("생성", "수정"),
    ("발생", "종료"),
    ("출발", "도착"),
]
REFERENCE_PAIR_TOKENS = [
    ("코드", "명"),
    ("코드", "이름"),
    ("아이디", "명"),
    ("아이디", "이름"),
    ("번호", "명"),
]
SPECIAL_CHAR_RE = re.compile(r"[^\w\s\-./,:()가-힣]")
BROKEN_TEXT_RE = re.compile(r"[�]|[ㄱ-ㅎㅏ-ㅣ]{2,}")
PHONE_DIGIT_RE = re.compile(r"^[0-9+\-() ]+$")


def criterion_meta(category_group: str, criterion_name: str) -> tuple[str, str]:
    category = VALIDATION_CRITERIA[category_group]
    return category["label"], category["criteria"][criterion_name]


def build_finding(
    *,
    column_name: str,
    severity: str,
    category_group: str,
    criterion_name: str,
    message: str,
    finding_type: str | None = None,
    rule_id: str | None = None,
    row_indexes: list[int] | None = None,
    related_columns: list[str] | None = None,
    evidence: list[str] | None = None,
) -> ValidationFinding:
    category_label, criterion_description = criterion_meta(category_group, criterion_name)
    resolved_rule_id = rule_id or criterion_name
    resolved_finding_type = finding_type or (
        "manual_review"
        if severity == "info" or resolved_rule_id in {"manual_review_required", "categorical_value_manual_review"}
        else "issue"
    )
    display_label = "수동 검토 필요" if resolved_finding_type == "manual_review" else "오류/이상 탐지"
    return ValidationFinding(
        column_name=column_name,
        severity=severity,
        finding_type=resolved_finding_type,
        display_label=display_label,
        category_group=category_group,
        category_label=category_label,
        criterion_name=criterion_name,
        criterion_description=criterion_description,
        rule_id=resolved_rule_id,
        message=message,
        row_indexes=row_indexes or [],
        related_columns=related_columns or [],
        evidence=evidence or [],
    )


def parse_datetime(value: str) -> datetime | None:
    candidate = value.strip()
    if not candidate:
        return None
    for pattern in DATE_PATTERNS:
        try:
            return datetime.strptime(candidate, pattern)
        except ValueError:
            continue
    return None


def parse_number(value: str) -> float | None:
    try:
        return float(value.replace(",", "").strip())
    except Exception:
        return None


def contains_broken_text(value: str) -> bool:
    return bool(BROKEN_TEXT_RE.search(value))


def has_whitespace_issue(value: str) -> bool:
    return value != value.strip() or bool(re.search(r"\s{2,}", value))


def has_special_char_issue(value: str) -> bool:
    return bool(SPECIAL_CHAR_RE.search(value))


def allowed_values(term: StandardTerm | None) -> list[str]:
    if term is None or not term.allowed_values:
        return []
    raw = term.allowed_values.replace(":", " ").replace("(", " ").replace(")", " ")
    parts = [part.strip() for part in re.split(r"[,/]", raw) if part.strip()]
    cleaned: list[str] = []
    for part in parts:
        token = part.split()[0].strip()
        if token:
            cleaned.append(token)
    return cleaned
