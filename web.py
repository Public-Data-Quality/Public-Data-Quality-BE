from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

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
    app.run(host="0.0.0.0", port=5000, debug=True)
