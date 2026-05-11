from __future__ import annotations

import tempfile
import sys
import types
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

if __package__ in (None, ""):  # pragma: no cover
    package_name = "public_data_quality_be"
    package_dir = Path(__file__).resolve().parent
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__file__ = str(package_dir / "__init__.py")
        package.__path__ = [str(package_dir)]
        sys.modules[package_name] = package
    __package__ = package_name

from .core.llm.analysis_generation import (
    generate_analysis_code,
    generate_analysis_plan,
    repair_runtime_analysis_code,
)
from .service import default_data_paths, run_pipeline


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent.parent / "public-data-quality-fe" / "dist"))

    @app.get("/api/health")
    def health():
        meta_path, standard_path = default_data_paths()
        return jsonify(
            {
                "status": "ok",
                "meta_csv": str(meta_path),
                "standard_terms_csv": str(standard_path),
            }
        )

    @app.post("/api/analyze")
    def analyze():
        if request.content_type and request.content_type.startswith("multipart/form-data"):
            use_llm_agents = request.form.get("use_llm_agents", "false").lower() == "true"
            llm_model = request.form.get("llm_model") or None
            uploaded_file = request.files.get("dataset_file")

            if not uploaded_file:
                return jsonify({"error": "dataset_file is required"}), 400

            try:
                filename = secure_filename(uploaded_file.filename or "uploaded_dataset.csv")
                suffix = Path(filename).suffix or ".csv"
                with tempfile.TemporaryDirectory(prefix="public_data_quality_upload_") as tmp_dir:
                    uploaded_path = Path(tmp_dir) / f"dataset{suffix}"
                    uploaded_file.save(uploaded_path)
                    result = run_pipeline(
                        uploaded_dataset_csv=str(uploaded_path),
                        uploaded_dataset_name=Path(filename).stem,
                        use_llm_agents=use_llm_agents,
                        llm_model=llm_model,
                    )
                    return jsonify(result)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except Exception as exc:  # pragma: no cover
                return jsonify({"error": str(exc)}), 500

        return jsonify({"error": "Use multipart/form-data with dataset_file"}), 400

    @app.post("/api/llm-analysis-code")
    def llm_analysis_code():
        try:
            payload = request.get_json(silent=True) or {}
            headers = payload.get("headers")
            method_text = str(payload.get("method_text") or "").strip()
            column_name = str(payload.get("column_name") or "").strip()
            llm_model = str(payload.get("llm_model") or "").strip() or None

            if not isinstance(headers, list) or not headers:
                return jsonify({"error": "headers are required"}), 400
            if not method_text:
                return jsonify({"error": "method_text is required"}), 400
            if not column_name:
                return jsonify({"error": "column_name is required"}), 400

            generated, generation_error = generate_analysis_code(payload, llm_model)
            if generated is None:
                return jsonify({"error": generation_error}), 422

            return jsonify(generated)
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"LLM analysis code endpoint failed: {exc}"}), 500

    @app.post("/api/llm-analysis-plan")
    def llm_analysis_plan():
        try:
            payload = request.get_json(silent=True) or {}
            headers = payload.get("headers")
            llm_model = str(payload.get("llm_model") or "").strip() or None

            if not isinstance(headers, list) or not headers:
                return jsonify({"error": "headers are required"}), 400

            items, generation_error = generate_analysis_plan(payload, llm_model)
            if items is None:
                return jsonify({"error": generation_error}), 422

            return jsonify({"items": items})
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"LLM analysis plan endpoint failed: {exc}"}), 500

    @app.post("/api/llm-analysis-code/repair-runtime")
    def repair_llm_analysis_code_runtime():
        try:
            payload = request.get_json(silent=True) or {}
            headers = payload.get("headers")
            method_text = str(payload.get("method_text") or "").strip()
            column_name = str(payload.get("column_name") or "").strip()
            previous_code = str(payload.get("previous_code") or "").strip()
            runtime_error = str(payload.get("runtime_error") or "").strip()
            llm_model = str(payload.get("llm_model") or "").strip() or None

            if not isinstance(headers, list) or not headers:
                return jsonify({"error": "headers are required"}), 400
            if not method_text:
                return jsonify({"error": "method_text is required"}), 400
            if not column_name:
                return jsonify({"error": "column_name is required"}), 400
            if not previous_code:
                return jsonify({"error": "previous_code is required"}), 400
            if not runtime_error:
                return jsonify({"error": "runtime_error is required"}), 400

            generated, generation_error = repair_runtime_analysis_code(payload, llm_model)
            if generated is None:
                return jsonify({"error": generation_error}), 422

            return jsonify(generated)
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"LLM runtime repair endpoint failed: {exc}"}), 500

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def frontend(path: str):
        dist_dir = Path(app.static_folder)
        if not dist_dir.exists():
            return (
                "Frontend build not found. Run `npm install --prefix public-data-quality-fe` and "
                "`npm run build --prefix public-data-quality-fe`, or use "
                "`npm run dev --prefix public-data-quality-fe` for development.",
                503,
            )
        if path and (dist_dir / path).exists():
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
