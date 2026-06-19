from __future__ import annotations

import re


def looks_route_name_column(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return any(
        token in name
        for token in ("도로(노선)명", "노선명", "도로명", "도로노선명")
    )


def looks_institution_category_column(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return any(
        token in name
        for token in ("시설구분", "기관구분", "시설유형", "기관유형", "분류", "구분", "유형")
    )


def is_public_private_category_value(value: str) -> bool:
    text = re.sub(r"\s+", "", str(value or "").strip())
    return bool(re.fullmatch(r"(공공|민간)(기관|시설)?", text))


def looks_date_column(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return "date" in column.semantic_tags or any(
        token in name for token in ("일자", "일시", "날짜", "년월", "등록일", "기준일")
    )


def looks_boolean_column(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return "boolean" in column.semantic_tags or any(
        token in name for token in ("여부", "유무", "YN", "Yn", "yn", "Y/N")
    )


def looks_date_value(value: str) -> bool:
    return bool(re.match(r"^\d{4}[-./]?\d{1,2}[-./]?\d{1,2}$", value.strip()))


def is_yn_value(value: str) -> bool:
    return value.strip().upper() in {"Y", "N"}


def allows_local_prefix_truncation(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    excluded_tags = {
        "numeric",
        "count",
        "quantity",
        "amount",
        "rate",
        "width",
        "identifier",
        "code",
        "date",
        "boolean",
        "phone",
        "geo_lat",
        "geo_lon",
        "coordinate_pair",
    }
    if excluded_tags.intersection(set(column.semantic_tags)):
        return False
    if column.inferred_primitive_type in {"numeric", "date", "empty"}:
        return False
    if looks_route_name_column(column):
        return False
    if any(
        token in name
        for token in ("우편번호", "우편", "번호", "코드", "일자", "일시", "날짜", "수용인원")
    ):
        return False
    return "name" in column.semantic_tags or any(
        token in name for token in ("명", "명칭", "내용", "설명", "사유", "비고", "메모")
    )


def allows_local_surface_normalization(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    excluded_tags = {
        "numeric",
        "count",
        "quantity",
        "amount",
        "rate",
        "width",
        "date",
        "boolean",
        "phone",
        "geo_lat",
        "geo_lon",
        "coordinate_pair",
    }
    if excluded_tags.intersection(set(column.semantic_tags)):
        return False
    if column.inferred_primitive_type in {"numeric", "date", "empty"}:
        return False
    if looks_route_name_column(column):
        return False
    if any(token in name for token in ("우편번호", "우편", "일자", "일시", "날짜", "수용인원")):
        return False
    return True
