from __future__ import annotations

import json
import re
from typing import Any

from ..core.config.constants import (
    LLM_DEFAULT_MODEL,
    LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT,
    LLM_STANDARD_TERM_SAMPLE_SIZE,
    TAG_RULE_MAP,
)
from ..core.llm import ChatLLMClient
from ..core.schema.models import ColumnProfile, PipelineState
from ..core.schema.retrieval import resolve_with_rag
from ..core.validation import semantic_profile_llm_reasons
from .base import BaseAgent


class LLMRoutingAgent(BaseAgent):
    name = "rule_router"

    @staticmethod
    def _rule_tags(column: ColumnProfile) -> list[str]:
        tags = set(column.semantic_tags)
        name = f"{column.raw_name} {column.normalized_name}"
        if any(token in name for token in ("일자", "일시", "날짜", "년월", "등록일", "기준일")):
            tags.add("date")
        if any(token in name for token in ("주소", "소재지")):
            tags.add("address")
        if "위도" in name:
            tags.add("geo_lat")
        if "경도" in name:
            tags.add("geo_lon")
        if any(token in name for token in ("여부", "유무", "YN", "Yn", "yn", "Y/N")):
            tags.add("boolean")
        if any(token in name for token in ("구분", "유형", "종류", "상태", "분류")):
            tags.add("enum")
        if any(token in name for token in ("코드",)):
            tags.add("code")
        if any(token in name for token in ("명", "명칭", "이름", "기관명", "시설명", "경찰서명")):
            tags.add("name")
        if any(token in name for token in ("대수", "개수", "건수", "수량", "좌석수", "정원수")):
            tags.add("quantity")
        if any(token in name for token in ("폭", "너비")):
            tags.add("width")
        if any(token in name for token in ("전화", "연락처", "휴대전화")):
            tags.add("phone")
        return sorted(tags)

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        updated: list[ColumnProfile] = []
        rag_count = 0

        for column in state["columns"]:
            rule_ids: list[str] = []
            for tag in self._rule_tags(column):
                rule_ids.extend(TAG_RULE_MAP.get(tag, []))
            column.assigned_rules = list(dict.fromkeys(rule_ids))
            column.standard_match_type = "unmatched"
            column.rag_required = True
            rag_count += 1

            traces.append(
                self.trace(
                    action="route_rules",
                    target=column.raw_name,
                    detail=(
                        f"rules={column.assigned_rules}, "
                        f"confidence={column.routing_confidence:.2f}, rag={column.rag_required}, "
                        f"match_type={column.standard_match_type}, candidates={column.standard_candidates}"
                    ),
                )
            )
            updated.append(column)

        traces.append(self.trace(action="routing_summary", detail=f"rag_required={rag_count}"))
        return {"columns": updated, "agent_traces": traces}


class LLMColumnResolver:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_DEFAULT_MODEL
        self._llm = ChatLLMClient(model_name=self.model_name)

    @property
    def enabled(self) -> bool:
        return self._llm.enabled

    def _client(self):
        return self._llm if self.enabled else None

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any] | None:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def resolve(self, state: PipelineState, column: ColumnProfile) -> dict[str, Any] | None:
        llm = self._client()
        if llm is None:
            return None

        dataset_meta = state["dataset_meta"]
        standard_terms = list(state["standard_terms"].keys())[:LLM_STANDARD_TERM_SAMPLE_SIZE]
        prompt = f"""
You are a public-data schema routing agent.
Return strict JSON with keys:
- normalized_name: string
- semantic_tags: list[string]
- assigned_rules: list[string]
- standard_candidates: list[string]
- reason: string

Dataset:
- name: {dataset_meta.dataset_name}
- provider: {dataset_meta.provider_name}
- keywords: {", ".join(dataset_meta.keywords)}
- format: {dataset_meta.data_format}

Column:
- raw_name: {column.raw_name}
- normalized_name: {column.normalized_name}
- source: {column.source}
- inferred_type: {column.inferred_primitive_type}
- sample_values: {column.sample_values}
- top_values: {column.top_values[:5]}

Instructions:
- Infer semantic_tags and assigned_rules from the column meaning, not from fixed string matching rules.
- standard_candidates should contain the best matching canonical standard terms only.
- If there is no confident standard term match, return an empty list for standard_candidates.
- assigned_rules may be non-empty even when standard_candidates is empty if the column meaning is still clear.
- reason should be a short Korean sentence.

Known standard terms sample:
{json.dumps(standard_terms, ensure_ascii=False)}
"""
        response = llm.invoke_json(prompt, system_prompt="You are a careful public-data schema routing assistant.")
        if response is None:
            return None
        return self._parse_json_content(response.content)


class LLMSemanticProfiler:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_DEFAULT_MODEL
        self._llm = ChatLLMClient(model_name=self.model_name)

    @property
    def enabled(self) -> bool:
        return self._llm.enabled

    def _client(self):
        return self._llm if self.enabled else None

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any] | None:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _is_actionable_analysis_method(method: str) -> bool:
        text = method.strip()
        if not text:
            return False
        measurable_tokens = (
            "비율",
            "분포",
            "개수",
            "고유값",
            "대표값",
            "빈값",
            "결측",
            "길이",
            "범위",
            "최솟값",
            "최댓값",
            "파싱",
            "상위값",
            "중복",
            "좌표",
            "위도",
            "경도",
            "위치",
            "지역별",
            "집중도",
            "기관별",
            "다른 컬럼",
            "관련 컬럼",
            "동시",
            "모순",
            "쌍",
            "형식별",
            "행 수",
            "%",
        )
        vague_phrases = ("일관성을 검토", "확인합니다", "분석합니다", "검토합니다")
        return any(token in text for token in measurable_tokens) and not any(phrase in text for phrase in vague_phrases)

    @staticmethod
    def _fallback_analysis_methods(column: ColumnProfile) -> list[str]:
        methods: list[str] = []
        if "주소" in column.raw_name or "소재지" in column.raw_name:
            methods.append("주소의 시도/시군구 단위 분포와 위도/경도 평균 좌표를 함께 계산해 지역별 데이터 집중도를 봅니다.")
        if "위도" in column.raw_name or "경도" in column.raw_name:
            methods.append("위도와 경도를 쌍으로 묶어 좌표 중심, 좌표 범위, 상위 주소 지역별 평균 좌표를 계산합니다.")
        if any(token in column.raw_name for token in ("관리기관", "운영기관", "제공기관", "소관기관", "관할", "기관명", "경찰서명")):
            methods.append("기관명과 주소/소재지 또는 위도/경도를 결합해 기관별 지역 분포와 상위 기관-지역 조합의 집중도를 계산합니다.")
        if any(token in column.raw_name for token in ("대상시설", "시설명", "명칭", "이름")):
            methods.append("명칭별 관련 숫자 컬럼의 행 수, 평균, 최솟값, 최댓값을 묶어 규모 차이를 계산합니다.")
        if "여부" in column.raw_name:
            methods.append("여부 값별 관련 대수/건수 컬럼의 총합과 평균을 비교해 상태별 규모 차이를 계산합니다.")
        if any(token in column.raw_name for token in ("대수", "건수", "수량", "개수")):
            methods.append("관련 여부 컬럼의 값별로 이 수량 컬럼의 총합, 평균, 양수 비율을 비교합니다.")
        if "시작" in column.raw_name or "종료" in column.raw_name:
            methods.append("시작/종료 날짜 컬럼을 쌍으로 묶어 같은 월 종료 비율과 연도 구간 분포를 계산합니다.")
        return list(dict.fromkeys(methods))[:4]

    def profile(self, state: PipelineState, column: ColumnProfile) -> dict[str, Any] | None:
        llm = self._client()
        if llm is None:
            return None

        dataset_meta = state["dataset_meta"]
        prompt = f"""
You are a semantic profiling agent for Korean public datasets.
Return strict JSON with keys:
- label: short semantic role name in Korean
- description: one sentence in Korean about the business meaning of this column
- analysis_methods: list of 0 to 3 short Korean sentences describing data-driven analyses that combine this column with related columns when possible
- confidence: float between 0 and 1

Rules:
- label must be written in Korean only
- description must be written in Korean only
- analysis_methods must be written in Korean only
- return an empty analysis_methods list if the samples and column name do not support a concrete measurable analysis
- do not use English unless the original column itself is an English acronym that must remain unchanged
- prefer concise public-data terminology
- only explain what this column represents or means in the dataset
- prefer cross-column analyses only when the related column is likely to exist from the current dataset context
- analysis_methods must be data-driven and measurable from the uploaded data, not generic business review advice
- analysis_methods must be domain analyses based on domain column combinations; do not suggest single-column profiling such as value skew, missing ratio, parsing success, distinct count, or generic distribution
- each analysis method should mention an insight-oriented cross-column metric when possible, such as regional concentration from address plus latitude/longitude, installed flag groups versus CCTV count totals/averages, start/end date duration distribution, code/name category share, or total/subtotal contribution
- for address columns, prefer regional distribution, top regions, coordinate center/range, or road-name address versus jibun address coverage share
- for 여부 columns, prefer comparing the flag groups with related count/amount totals, averages, and positive ratios
- avoid phrasing the method as a data quality test whose only output is missing rows, invalid rows, or contradiction rows
- if there is no likely domain column combination, return an empty analysis_methods list
- do not suggest manually splitting address components unless the current task explicitly requires geocoding or parsing
- do not use vague phrases like "일관성을 검토", "확인합니다", or "분석합니다" without a measurable criterion
- do not claim that a problem exists unless it is visible in the provided samples
- do not suggest broad generic items such as simple review, general trend analysis, vague distribution checks, value skew checks, missing-value checks, parsing checks, or distinct-count checks
- do not mention standard term mapping
- output JSON only

Dataset:
- name: {dataset_meta.dataset_name}
- provider: {dataset_meta.provider_name}
- format: {dataset_meta.data_format}

Column:
- raw_name: {column.raw_name}
- normalized_name: {column.normalized_name}
- semantic_tags: {column.semantic_tags}
- standard_candidates: {column.standard_candidates}
- inferred_type: {column.inferred_primitive_type}
- sample_values: {column.sample_values}
- top_values: {column.top_values[:3]}
"""
        response = llm.invoke_json(
            prompt,
            system_prompt=(
                "You are a semantic profiling assistant for Korean public datasets. "
                "Respond with a single JSON object only. No markdown, no explanation, no code fences."
            ),
        )
        if response is None:
            return None
        payload = self._parse_json_content(response.content)
        if payload is None:
            llm.last_error = f"llm_parse_error:{response.content[:200]}"
            return None
        confidence = payload.get("confidence")
        if confidence is None:
            payload["confidence"] = LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT
        if payload.get("label"):
            payload["label"] = str(payload["label"]).strip()
        if payload.get("description"):
            payload["description"] = str(payload["description"]).strip()
        methods = payload.get("analysis_methods") or []
        if isinstance(methods, list):
            payload["analysis_methods"] = [
                str(method).strip()
                for method in methods
                if self._is_actionable_analysis_method(str(method))
            ][:4]
        else:
            payload["analysis_methods"] = []
        if not payload["analysis_methods"]:
            payload["analysis_methods"] = self._fallback_analysis_methods(column)
        return payload


class RAGResolverAgent(BaseAgent):
    name = "rag_resolver"

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        resolved = []

        for column in state["columns"]:
            if not column.rag_required:
                resolved.append(column)
                continue

            column = resolve_with_rag(
                column=column,
                dataset_meta=state["dataset_meta"],
                standard_terms=state["standard_terms"],
                synonym_index=state["synonym_index"],
                example_index=state["example_index"],
            )
            if column.standard_candidates and column.standard_match_type == "unmatched":
                column.standard_match_type = "rag_resolved"

            traces.append(
                self.trace(
                    action="resolve_column",
                    target=column.raw_name,
                    detail=(
                        f"candidates={column.standard_candidates}, "
                        f"evidence={len(column.rag_evidence)}"
                    ),
                )
            )
            resolved.append(column)

        return {"columns": resolved, "agent_traces": traces}


class SemanticProfilingAgent(BaseAgent):
    name = "semantic_profiler"

    def __init__(self, semantic_profiler: LLMSemanticProfiler | None = None):
        self.semantic_profiler = semantic_profiler

    def _llm_debug_detail(self, use_llm: bool, llm_attempted: bool) -> tuple[str, str]:
        if not use_llm or not llm_attempted or self.semantic_profiler is None:
            return "", ""
        client = self.semantic_profiler._llm
        return client.last_error, client.last_response_preview

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        updated = []
        use_llm = bool(state.get("use_llm_agents")) and self.semantic_profiler is not None

        for column in state["columns"]:
            llm_reasons = semantic_profile_llm_reasons(column)
            needs_llm = use_llm
            llm_attempted = False
            column.semantic_profile_llm_needed = needs_llm
            column.semantic_profile_llm_reasons = ["컬럼 의미 설명 생성"] if use_llm else llm_reasons
            if use_llm:
                llm_attempted = True
                profile = self.semantic_profiler.profile(state, column)
                if profile:
                    column.semantic_profile_label = profile.get("label")
                    column.semantic_profile_description = profile.get("description")
                    column.semantic_profile_analysis_methods = profile.get("analysis_methods") or []
                    column.semantic_profile_confidence = profile.get("confidence")
            if not column.semantic_profile_analysis_methods:
                column.semantic_profile_analysis_methods = LLMSemanticProfiler._fallback_analysis_methods(column)
            llm_error, llm_preview = self._llm_debug_detail(use_llm, llm_attempted)
            traces.append(
                self.trace(
                    action="semantic_profile",
                    target=column.raw_name,
                    detail=(
                        f"label={column.semantic_profile_label}, "
                        f"confidence={column.semantic_profile_confidence}, "
                        f"llm_needed={needs_llm}, reasons={llm_reasons}, "
                        f"llm_error={llm_error}, "
                        f"llm_preview={llm_preview}"
                    ),
                )
            )
            updated.append(column)

        return {"columns": updated, "agent_traces": traces}


def should_run_rag(state: PipelineState) -> str:
    if any(column.rag_required for column in state["columns"]):
        return "rag_resolve"
    return "validate"
