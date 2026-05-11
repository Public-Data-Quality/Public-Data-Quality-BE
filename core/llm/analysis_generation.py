from __future__ import annotations

import ast
import json
import re
from typing import Any

from .client import ChatLLMClient


FORBIDDEN_ANALYSIS_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "print",
    "setattr",
    "vars",
}


def generate_analysis_code(payload: dict[str, Any], llm_model: str | None = None) -> tuple[dict[str, str] | None, str]:
    llm = ChatLLMClient(model_name=llm_model)
    if not llm.enabled:
        return None, llm.last_error or "LLM is not configured"

    generated, error = _request_valid_llm_analysis_code(llm, _analysis_code_prompt(payload))
    if generated is None:
        return None, error

    return {
        "title": str(generated.get("title") or "LLM 실행 분석").strip(),
        "code": str(generated.get("code") or "").strip(),
    }, ""


def repair_runtime_analysis_code(payload: dict[str, Any], llm_model: str | None = None) -> tuple[dict[str, str] | None, str]:
    llm = ChatLLMClient(model_name=llm_model)
    if not llm.enabled:
        return None, llm.last_error or "LLM is not configured"

    generated, error = _request_valid_llm_analysis_code(llm, _runtime_repair_prompt(payload))
    if generated is None:
        return None, error

    return {
        "title": str(generated.get("title") or "LLM 실행 분석").strip(),
        "code": str(generated.get("code") or "").strip(),
    }, ""


def generate_analysis_plan(payload: dict[str, Any], llm_model: str | None = None) -> tuple[list[dict[str, str]] | None, str]:
    headers = payload.get("headers") or []
    column_profiles = payload.get("column_profiles") or []
    llm = ChatLLMClient(model_name=llm_model)
    if not llm.enabled:
        return None, llm.last_error or "LLM is not configured"

    response = llm.invoke_json(
        _analysis_plan_prompt(payload),
        system_prompt=(
            "You design concrete Korean public-data analysis items. "
            "Return one strict JSON object only. No markdown."
        ),
    )
    if response is None:
        return None, llm.last_error or "LLM request failed"

    try:
        generated = json.loads(response.content)
    except json.JSONDecodeError:
        return None, "LLM returned invalid JSON"

    return _normalize_analysis_items(generated, headers, column_profiles), ""


def _validate_generated_analysis_code(code: str) -> str:
    if not code or len(code) > 12000:
        return "generated code is empty or too large"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"

    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != 1 or function_defs[0].name != "analyze":
        return "code must define exactly one function named analyze"
    if any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        return "top-level code other than analyze() is not allowed"

    args = [arg.arg for arg in function_defs[0].args.args]
    if args[:4] != ["rows", "headers", "column_name", "method_text"]:
        return "analyze() must accept rows, headers, column_name, method_text"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)):
            return f"{type(node).__name__} is not allowed"
        if isinstance(node, ast.While):
            return "while loops are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute access is not allowed"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "dunder names are not allowed"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_ANALYSIS_CALLS:
                return f"{node.func.id}() is not allowed"
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ANALYSIS_CALLS:
                return f"{node.func.attr}() is not allowed"
    return ""


def _request_valid_llm_analysis_code(llm: ChatLLMClient, prompt: str) -> tuple[dict[str, Any] | None, str]:
    last_error = ""
    current_prompt = prompt
    previous_code = ""

    for _ in range(3):
        response = llm.invoke_json(
            current_prompt,
            system_prompt=(
                "You write small, deterministic Python analysis functions for Pyodide. "
                "Return one strict JSON object only. No markdown."
            ),
        )
        if response is None:
            return None, llm.last_error or "LLM request failed"

        try:
            generated = json.loads(response.content)
        except json.JSONDecodeError:
            last_error = "LLM returned invalid JSON"
            current_prompt = _static_repair_prompt(prompt, response.content[:4000], last_error)
            continue

        code = str(generated.get("code") or "").strip()
        validation_error = _validate_generated_analysis_code(code)
        if not validation_error:
            return generated, ""

        last_error = validation_error
        previous_code = code
        current_prompt = _static_repair_prompt(prompt, previous_code, validation_error)

    return None, f"LLM analysis code rejected after retry: {last_error}"


def _normalize_text(value: Any) -> str:
    return re.sub(r"[\s_\-()[\]{}.,/\\|:;'\"`~!@#$%^&*+=?<>]", "", str(value or "").lower())


def _profile_by_name(column_profiles: list[Any]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for profile in column_profiles:
        if isinstance(profile, dict) and profile.get("name"):
            profiles[str(profile["name"])] = profile
    return profiles


def _profile_role(profile: dict[str, Any] | None) -> str:
    return str((profile or {}).get("role") or "")


def _profile_number(profile: dict[str, Any] | None, key: str) -> float:
    value = (profile or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _method_mentions_any_header(method_text: str, headers: list[str], excluded: str) -> bool:
    return bool(_mentioned_headers(method_text, headers, excluded))


def _mentioned_headers(method_text: str, headers: list[str], excluded: str = "") -> list[str]:
    normalized_method = _normalize_text(method_text)
    mentioned: list[str] = []
    for header in headers:
        if header == excluded:
            continue
        normalized_header = _normalize_text(header)
        if len(normalized_header) >= 2 and normalized_header in normalized_method:
            mentioned.append(header)
    return mentioned


def _looks_like_generic_analysis(title: str, method_text: str) -> bool:
    normalized = _normalize_text(f"{title} {method_text}")
    generic_tokens = [
        "값분포",
        "상위값",
        "최빈값",
        "쏠림",
        "편중",
        "결측",
        "누락",
        "파싱",
        "고유값",
        "중복",
        "품질",
        "오류",
        "이상치",
    ]
    return any(token in normalized for token in generic_tokens)


def _requires_numeric_metric(method_text: str) -> bool:
    normalized = _normalize_text(method_text)
    if _looks_like_flag_ratio(method_text):
        return False
    numeric_tokens = ["합계", "평균", "총량", "총계", "대수", "수량", "금액", "면적", "용량", "정원", "규모"]
    return any(token in normalized for token in numeric_tokens)


def _looks_like_flag_ratio(method_text: str) -> bool:
    normalized = _normalize_text(method_text)
    return ("비율" in normalized or "분포" in normalized) and any(
        token in normalized for token in ["여부", "유무", "상태", "설치", "운영", "가능"]
    )


def _has_supported_numeric_metric(method_text: str, profiles: dict[str, dict[str, Any]]) -> bool:
    normalized_method = _normalize_text(method_text)
    for name, profile in profiles.items():
        normalized_name = _normalize_text(name)
        if normalized_name not in normalized_method:
            continue
        if _profile_role(profile) == "numeric_metric" and _profile_number(profile, "numeric_parse_ratio") >= 0.7:
            return True
    return False


def _valid_plan_item(
    title: str,
    target_column: str,
    method_text: str,
    headers: list[str],
    profiles: dict[str, dict[str, Any]],
) -> bool:
    if _looks_like_generic_analysis(title, method_text):
        return False

    target_profile = profiles.get(target_column)
    target_role = _profile_role(target_profile)
    normalized_method = _normalize_text(method_text)

    if target_role == "identifier_code" and "코드" not in normalized_method:
        return False

    mentioned_related_headers = _mentioned_headers(method_text, headers, target_column)
    if not mentioned_related_headers:
        return False

    if _requires_numeric_metric(method_text) and not _has_supported_numeric_metric(method_text, profiles):
        return False

    if _looks_like_flag_ratio(method_text) and target_role != "flag_status":
        return False

    if target_role in {"category", "organization"}:
        if not any(_profile_role(profiles.get(header)) in {"numeric_metric", "address", "latitude", "longitude", "date", "flag_status"} for header in mentioned_related_headers):
            return False

    return True


def _normalize_analysis_items(payload: Any, headers: list[Any], column_profiles: list[Any]) -> list[dict[str, str]]:
    safe_headers = [str(header) for header in headers]
    header_set = set(safe_headers)
    profiles = _profile_by_name(column_profiles)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return []

    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in payload["items"]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        target_column = str(item.get("target_column") or "").strip()
        method_text = str(item.get("method_text") or "").strip()
        visualization_hint = str(item.get("visualization_hint") or "combo").strip()
        if not target_column or target_column not in header_set or not method_text:
            continue
        if not _valid_plan_item(title, target_column, method_text, safe_headers, profiles):
            continue
        key = (target_column, method_text)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "title": title or "LLM 추천 분석",
                "target_column": target_column,
                "method_text": method_text,
                "visualization_hint": visualization_hint,
            }
        )
    return items[:6]


def _analysis_code_prompt(payload: dict[str, Any]) -> str:
    headers = payload.get("headers") or []
    sample_rows = payload.get("sample_rows") or []
    csv_sample_text = str(payload.get("csv_sample_text") or "")[:20000]
    row_count_estimate = payload.get("row_count_estimate")
    column_name = payload.get("column_name") or ""
    method_text = payload.get("method_text") or ""
    return f"""
Generate Python code that executes a user-requested data analysis for a Korean public dataset.
Return strict JSON with keys:
- title: short Korean title for the analysis result
- code: Python source code only

The code must:
- define exactly one function: analyze(rows, headers, column_name, method_text)
- return a JSON-serializable dict with keys summary, metrics, and visualization
- use rows as a list of dict records and headers as a list of column names
- compute the requested method from the real rows passed at runtime
- infer related columns from headers and method_text when needed
- If method_text explicitly names headers after normalization, use only those named headers plus column_name as analysis columns.
- Do not substitute another category column when method_text already names a related category column.
- For example, if method_text names "위험요인", do not use "시설종류"; if method_text names "관리기관명" and an address column, do not use unrelated facility type columns.
- match related columns robustly by normalizing both headers and method_text:
  remove spaces, underscores, hyphens, parentheses, and punctuation before comparing.
  For example, method text "CCTV 설치 대수" should match headers like "CCTV설치대수", "CCTV_설치_대수", or "cctv 설치대수".
  For example, method text "시설 종류" should match headers like "시설종류", "시설 유형", "시설구분", or "시설분류".
- for category + numeric ratio analyses, group by column_name and choose the related numeric metric column from method_text and headers.
  Sum the numeric metric per group, then calculate each group's ratio against the total metric sum.
- for category + flag/status ratio analyses such as "시설 종류별 CCTV 설치 비율" or a target column ending with "여부":
  do not search for a numeric metric column.
  Treat column_name as the flag/status column, find the related category column from method_text and headers,
  then calculate each category's total rows, positive/installed rows, and positive ratio.
  Initialize every group with all required keys before incrementing or reading them, e.g.
  {{"total": 0, "positive": 0}}; never read group["positive"] before assignment.
  If method_text explicitly names a related category header, that exact header is mandatory.
  If method_text says "시설 종류별" and no exact "시설 종류" header exists, prefer headers containing "시설" plus one of "종류", "유형", "구분", or "분류".
  Positive Korean flag values include "Y", "YES", "TRUE", "1", "O", "유", "있음", "설치", "설치됨", "운영", "가능".
- do not return a successful zero-total result when non-empty numeric metric values exist in a related metric column.
- do not return "관련된 계량형 컬럼이 없습니다" for flag/status ratio analyses; they are count-and-ratio analyses, not numeric-sum analyses.
- do not return "관련된 카테고리 컬럼이 없습니다" before trying normalized category matching such as "시설 종류" -> "시설종류/시설유형/시설구분/시설분류".
- prefer cross-column analysis over analyzing column_name alone
- inspect the provided sample rows and csv_sample_text before choosing related columns, grouping keys, numeric columns, and visualization
- keep metrics chart-friendly when possible:
  - top_values, top_regions, top_organizations, or top_pairs as lists of {{value, count, ratio}}
  - *_distribution as dicts of label to count
  - numeric scalar metrics for totals, averages, min, max, failed counts
- create visualization yourself as a JSON spec in the returned result:
  - bar: {{"type":"bar","title":"...","x_label":"...","y_label":"...","rows":[{{"label":"...","value":10,"ratio":"..."}}, ...]}}
  - donut: {{"type":"donut","title":"...","rows":[{{"label":"...","value":10,"ratio":"..."}}, ...]}} for small categorical shares
  - histogram: {{"type":"histogram","title":"...","bins":[{{"label":"0~10","value":5}}, ...]}} for numeric ranges
  - line: {{"type":"line","title":"...","rows":[{{"label":"2024","value":10}}, ...]}} for year/month/order trends
  - scatter: {{"type":"scatter","title":"...","x_label":"...","y_label":"...","points":[{{"x":127.1,"y":37.4,"label":"..."}}, ...]}}
  - heatmap: {{"type":"heatmap","title":"...","columns":["..."],"rows":[{{"...":"..."}}, ...]}} for matrix-like region/category concentration
  - table: {{"type":"table","title":"...","columns":["..."],"rows":[{{"...":"..."}}, ...]}}
  - stats: {{"type":"stats","title":"...","items":[{{"label":"...","value":"..."}}, ...]}}
  - combo: {{"type":"combo","title":"...","charts":[bar_or_donut_histogram_line_scatter_heatmap_table_or_stats_specs]}}
- every successful analysis must include at least one plot-like chart in visualization: bar, donut, histogram, line, scatter, heatmap, or combo containing one of them.
- do not return only table or only stats as visualization. If a table or stats is useful, wrap it in combo with a plot first.
- choose a domain-fit visualization, not always a bar chart:
  - coordinates/location: scatter plus region bar/table
  - numeric range: histogram plus stats
  - small categorical share: donut plus bar/table
  - time/order/period: line plus stats
  - organization-region or code-name matrix: heatmap/table plus bars
- limit visualization rows/points to at most 30 items
- visualization title, axis labels, summary, and metrics names must describe the actual columns used.
- handle missing columns and non-numeric values gracefully
- avoid KeyError by using dict.get(), setdefault(), defaultdict with a complete factory, or explicit initialization.
- never import modules. Do not write import or from-import statements.
- do not define top-level variables, helper functions, or classes. Put every helper inside analyze().
- never read files, access network, use eval/exec/open/input/print, or use infinite loops
- use only built-ins plus Counter, defaultdict, math, re, statistics, mean, median already available in globals
- output JSON only

Dataset headers:
{json.dumps(headers, ensure_ascii=False)}

Sample rows:
{json.dumps(sample_rows[:30], ensure_ascii=False)}

Raw CSV sample:
{csv_sample_text}

Estimated total CSV rows:
{row_count_estimate}

Target column:
{column_name}

Recommended analysis method:
{method_text}
"""


def _analysis_plan_prompt(payload: dict[str, Any]) -> str:
    headers = payload.get("headers") or []
    sample_rows = payload.get("sample_rows") or []
    column_profiles = payload.get("column_profiles") or []
    relationship_candidates = payload.get("relationship_candidates") or []
    csv_sample_text = str(payload.get("csv_sample_text") or "")[:20000]
    row_count_estimate = payload.get("row_count_estimate")
    return f"""
Generate a short list of concrete, domain-fit analysis items for a Korean public dataset.
Return strict JSON with key items only:
{{
  "items": [
    {{
      "title": "short Korean title",
      "target_column": "one exact column name from headers",
      "method_text": "one Korean sentence describing the measurable analysis",
      "visualization_hint": "bar|donut|histogram|line|scatter|heatmap|table|combo"
    }}
  ]
}}

Rules:
- Inspect column_profiles, relationship_candidates, headers, sample_rows, and csv_sample_text before selecting items.
- Treat column_profiles as the primary evidence about each column:
  role, semantic label/description, parse ratios, distinct count, top values, and numeric range.
- Use relationship_candidates as evidence only, not as commands. Select an item only when the sample data and profiles support it.
- Create 2 to 6 items only when the dataset supports real domain analysis.
- Every target_column must be copied exactly from Dataset headers.
- Prefer cross-column analyses that combine real related columns.
- A method_text must name the real columns or concepts being combined, and must be executable from the provided headers.
- A method_text must explicitly include every column needed for execution by exact header name.
- Do not write vague methods such as "지역별 분포" unless the method_text names the actual address/location/region column from headers.
- Do not write vague methods such as "시설의 개수" unless the method_text names the actual grouping column and counted entity column from headers.
- For category + numeric analyses, set target_column to the category/grouping column, and mention the numeric metric column in method_text.
- For category + flag/status ratio analyses, set target_column to the flag/status column, and mention the category/grouping column in method_text.
- For numeric analyses, use only columns whose column_profiles role is numeric_metric or numeric_parse_ratio is high.
- For flag/status ratio analyses, use count and positive ratio by category; do not require a numeric metric column.
- For category analyses, make sure a plausible category column exists from role/category, role/organization, or top-value evidence.
- Good examples:
  - organization/provider column + address/location column: regional responsibility distribution using both exact header names
  - address/location + latitude + longitude: spatial facility distribution
  - facility/category column + real amount/count/capacity/area column: scale comparison
  - status/flag column + category column: category-level positive/installed count and ratio
  - status/flag column + real count/amount column: status-group scale comparison when a real numeric metric exists
  - start/end date columns: period and yearly flow
  - code column + name column: code-name mapping summary
- Do not create generic profiling items such as value skew, null ratio, parsing success, distinct count, or simple top values.
- Do not create a recommendation if the method_text cannot name at least one related exact header besides target_column.
- Do not use identifier-like numeric columns such as phone number, postal code, serial number, ID, or code as a numeric metric.
- Do not invent CCTV/capacity/amount/coordinate/date analyses unless those concepts are visible in headers/profiles/sample values.
- If no useful domain item exists, return {{"items":[]}}.
- Output JSON only.

Dataset headers:
{json.dumps(headers, ensure_ascii=False)}

Column profiles:
{json.dumps(column_profiles[:80], ensure_ascii=False)}

Relationship candidates:
{json.dumps(relationship_candidates[:30], ensure_ascii=False)}

Sample rows:
{json.dumps(sample_rows[:30], ensure_ascii=False)}

Raw CSV sample:
{csv_sample_text}

Estimated total CSV rows:
{row_count_estimate}
"""


def _static_repair_prompt(original_prompt: str, previous_code: str, validation_error: str) -> str:
    return f"""
The previous Python code failed server validation.
Return a corrected strict JSON object with keys title and code.

Validation error:
{validation_error}

Previous code:
{previous_code}

Important correction rules:
- The code must contain exactly one top-level function named analyze.
- Do not include any import or from-import statements.
- Do not include top-level assignments, helper functions, classes, or executable statements.
- Put all helper logic inside analyze(rows, headers, column_name, method_text).
- Counter, defaultdict, math, re, statistics, mean, and median are already available.
- Return JSON only.

Original task:
{original_prompt}
"""


def _runtime_repair_prompt(payload: dict[str, Any]) -> str:
    return f"""
The previous Python analysis code passed static validation but failed at runtime in Pyodide.
Return a corrected strict JSON object with keys title and code.

Runtime error:
{str(payload.get("runtime_error") or "")[:4000]}

Previous code:
{str(payload.get("previous_code") or "")[:8000]}

Correction rules:
- Preserve the requested analysis intent.
- If the runtime error is KeyError such as KeyError: 'positive', fix all grouped aggregation dictionaries.
  Initialize every group with every key you later read, for example {{"total": 0, "positive": 0}}, or use
  group.get("positive", 0). Do not increment or read a missing key.
- If the runtime error says the analysis is not a numeric-metric analysis, do not search for a numeric column.
  For target columns ending with "여부" or methods like "시설 종류별 CCTV 설치 비율",
  calculate count and positive ratio by related category column instead.
- If the runtime error says the category column was not found, match "시설 종류" to headers such as
  "시설종류", "시설유형", "시설구분", or "시설분류" using normalized string comparison.
- The code must contain exactly one top-level function named analyze.
- Do not include import or from-import statements.
- Do not include top-level assignments, helper functions, classes, or executable statements.
- If you need helper functions such as distance calculations, define them inside analyze() before first use.
- Avoid referencing local variables before assignment.
- Counter, defaultdict, math, re, statistics, mean, and median are already available.
- Return a dict with summary, metrics, and visualization.
- Return JSON only.

Original task:
{_analysis_code_prompt(payload)}
"""
