from __future__ import annotations

DEFAULT_META_CSV_NAME = "행정안전부_공공데이터포털 목록 메타정보_20250531.csv"
DEFAULT_STANDARD_TERMS_CSV_NAME = "행정안전부_공공데이터 공통표준용어_20251101.csv"
VALIDATION_OUTPUT_DIR_NAME = "validation"
QUALITY_DETECTION_RESULTS_CSV_NAME = "quality_detection_results.csv"

NORMALIZATION_SYNONYM_PATCHES = {
    "연락처": "전화번호",
    "데이터기준일자": "자료기준일자",
    "데이터기준일": "자료기준일자",
    "기준일자": "기준일",
    "소재지도로명주소": "도로명주소",
    "소재지지번주소": "지번주소",
    "업소명": "명칭",
    "CCTV설치여부": "설치여부",
    "CCTV설치대수": "설치대수",
}

TAG_RULE_MAP = {
    "date": ["date_domain", "time_sequence_consistency", "precedence_accuracy"],
    "phone": ["number_domain"],
    "geo_lat": ["number_domain", "logical_consistency"],
    "geo_lon": ["number_domain", "logical_consistency"],
    "coordinate_pair": ["logical_consistency", "reference_relation"],
    "address": ["required_value", "whitespace_special_characters"],
    "boolean": ["boolean_domain", "logical_consistency"],
    "numeric": ["number_domain"],
    "count": ["quantity_domain", "logical_consistency", "calculation_formula"],
    "enum": ["code_domain", "reference_relation"],
    "identifier": ["number_domain", "duplicate_data", "reference_relation"],
    "name": ["required_value", "garbled_text", "whitespace_special_characters"],
    "width": ["number_domain"],
    "amount": ["amount_domain", "calculation_formula"],
    "quantity": ["quantity_domain", "calculation_formula"],
    "rate": ["rate_domain", "calculation_formula"],
    "code": ["code_domain", "reference_relation"],
}

VALIDATION_CRITERIA = {
    "relation_consistency": {
        "label": "데이터 관계 정합성",
        "criteria": {
            "time_sequence_consistency": "시간순서 관계를 갖는 컬럼 간의 데이터 오류 측정",
            "precedence_accuracy": "선후관계를 가지는 컬럼 간의 데이터 오류 측정",
            "logical_consistency": "컬럼 간 관계에 따른 특정 컬럼의 논리적 일관성 오류를 측정",
            "calculation_formula": "원천데이터의 계산 등을 통해 저장되는 계산 값이 정확하게 관리되고 있는지를 측정",
            "reference_relation": "참조하는 컬럼과 참조되는 컬럼 사이의 일관성이 유지되는지 측정",
        },
    },
    "completeness": {
        "label": "컬럼 완결성 검증",
        "criteria": {
            "garbled_text": "컬럼명, 데이터 값에 깨진 글자 또는 완성된 한글이 아닌 데이터 오류 측정",
            "whitespace_special_characters": "컬럼명, 데이터 값에 불필요한 공백과 특수문자가 입력된 오류 측정",
            "required_value": "데이터의 특성 상 반드시 입력되어야 하는 값은 누락없이 제공되어야 함",
            "duplicate_data": "DB 내 두 개 이상 테이블(또는 파일)에 존재하는 동일한(중복) 데이터의 값 일치 여부 측정",
        },
    },
    "domain_validity": {
        "label": "컬럼 특성 유효성 검증",
        "criteria": {
            "date_domain": "날짜 데이터를 저장하는 컬럼의 데이터 값이 유효한 범위를 벗어나는 오류를 측정",
            "number_domain": "정해진 규칙 등에 따라 관리되는 번호 도메인 데이터 오류를 측정",
            "boolean_domain": "여부/유무 등 2값 분류 도메인 컬럼의 유효값 범위 이탈 오류를 측정",
            "code_domain": "동일한 의미의 데이터가 표준으로 정의한 코드값으로 일관되게 적용되지 못한 오류를 측정",
            "categorical_semantic_domain": "범주형 문자열 컬럼의 고유값이 동일 도메인 체계와 의미적으로 일관되게 사용되는지를 측정",
            "amount_domain": "숫자로 저장된 금액 도메인 컬럼의 값이 유효한 범위를 벗어나는 오류를 측정",
            "quantity_domain": "숫자로 저장된 수량 도메인 컬럼의 값이 유효한 범위를 벗어나는 오류를 측정",
            "rate_domain": "숫자로 저장된 율 도메인 컬럼의 값이 유효한 범위를 벗어나는 오류를 측정",
        },
    },
}

DEFAULT_COLUMN_ROUTING_CONFIDENCE = 0.4

LLM_FAST_MODEL = "gemma4:e2b"
LLM_STRONG_MODEL = "gemma4:e4b"
LLM_DEFAULT_MODEL = LLM_FAST_MODEL
OLLAMA_DEFAULT_API_URL = "http://127.0.0.1:11434/api/chat"
LLM_REQUEST_TIMEOUT_SECONDS = 120
LLM_STANDARD_TERM_SAMPLE_SIZE = 200
LLM_RESOLUTION_CONFIDENCE = 0.78
LLM_STRONG_FALLBACK_CONFIDENCE = 0.72
LLM_SEMANTIC_PROFILE_TRIGGER_MATCH_TYPES = {"partial", "rule_only", "rag_resolved", "llm_resolved", "unmatched"}
LLM_SEMANTIC_PROFILE_ALWAYS_TRIGGER_TAGS = {"address"}
LLM_SEMANTIC_PROFILE_ALWAYS_TRIGGER_NAME_TOKENS = {
    "주소",
    "소재지",
    "위치",
    "설명",
    "내용",
    "비고",
    "사유",
    "메모",
    "상세",
    "특이사항",
    "조치",
    "민원",
    "안내",
}
LLM_SEMANTIC_PROFILE_SKIP_TAGS = {
    "date",
    "phone",
    "geo_lat",
    "geo_lon",
    "coordinate_pair",
    "boolean",
    "amount",
    "quantity",
    "count",
    "rate",
    "width",
}
LLM_SEMANTIC_PROFILE_AMBIGUOUS_TERMS = {
    "구분",
    "상태",
    "코드",
    "번호",
    "명",
    "명칭",
    "값",
    "내용",
    "유형",
    "종류",
    "정보",
    "데이터",
}

UPLOAD_DATASET_ID_PREFIX = "upload:"
UPLOAD_PROVIDER_NAME = "사용자 업로드"
UPLOAD_PROVIDER_CODE = "UPLOAD"
UPLOAD_DATASET_TYPE = "FILE"
UPLOAD_SERVICE_TYPE = "UPLOAD"
UPLOAD_UPDATE_CYCLE = "user_upload"

PROFILE_SAMPLE_ROW_LIMIT = 1000
PROFILE_SAMPLE_VALUES_LIMIT = 5
PROFILE_DISTINCT_TRACK_LIMIT = 200
PROFILE_TOP_VALUE_LIMIT = 30
PROFILE_TYPE_INFERENCE_THRESHOLD = 0.8

LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT = 0.75

CATEGORICAL_LLM_MIN_DISTINCT = 2
CATEGORICAL_LLM_MAX_DISTINCT = 30
CATEGORICAL_LLM_MIN_REPEAT_COUNT = 2
CATEGORICAL_LLM_CONFIDENCE_THRESHOLD = 0.9
