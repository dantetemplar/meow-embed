import argparse
import importlib
import json
import os
import sys
from typing import Any, Literal

MODEL_CONFIG_ENV = "MEOW_EMBED_MODEL_CONFIG_JSON"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastAPI and load dense/sparse models on startup. You could pass any uvicorn arguments to the command.",
        epilog=(
            "Examples:\n"
            "  meow-embed --SentenceTransformer sergeyzh/BERTA --SparseEncoder opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 --host 0.0.0.0 --port 8067\n"
            '  meow-embed --SparseEncoder opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 {"device":"cuda","max_active_dims":256}\n'
            '  meow-embed --FlagReranker BAAI/bge-reranker-v2-m3 {"use_fp16":true}\n'
            '  meow-embed --CrossEncoder jinaai/jina-reranker-v2-base-multilingual \'{"model_kwargs":{"torch_dtype":"auto"},"trust_remote_code":true}\'\n'
            '  meow-embed --BGEM3FlagModel BAAI/bge-m3 {"use_fp16":true}\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--SentenceTransformer",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a dense model; optional next token can be JSON kwargs for this model.",
    )
    parser.add_argument(
        "--SparseEncoder",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a sparse model; optional next token can be JSON kwargs for this model.",
    )
    parser.add_argument(
        "--FlagReranker",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a FlagEmbedding reranker model; optional next token can be JSON kwargs for this model.",
    )
    parser.add_argument(
        "--CrossEncoder",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a sentence-transformers CrossEncoder reranker; optional next token can be JSON kwargs for this model.",
    )
    parser.add_argument(
        "--BGEM3FlagModel",
        metavar="MODEL_ID [JSON_KWARGS]",
        action="append",
        help="Add a BGE M3 model; optional next token can be JSON kwargs for this model.",
    )
    args, _ = parser.parse_known_args()
    return args


def parse_model_specs(
    argv: list[str],
) -> tuple[
    list[
        tuple[
            Literal["dense", "sparse", "reranker", "crossEncoder", "bgeM3"],
            str,
            dict[str, Any],
        ]
    ],
    list[str],
]:
    models: list[
        tuple[
            Literal["dense", "sparse", "reranker", "crossEncoder", "bgeM3"],
            str,
            dict[str, Any],
        ]
    ] = []
    passthrough: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in (
            "--SentenceTransformer",
            "--SparseEncoder",
            "--FlagReranker",
            "--CrossEncoder",
            "--BGEM3FlagModel",
        ):
            if idx + 1 >= len(argv):
                raise SystemExit(f"Missing value for {token}.")
            model_id = argv[idx + 1]
            if model_id.startswith("--"):
                raise SystemExit(f"Missing value for {token}.")
            model_type: Literal[
                "dense", "sparse", "reranker", "crossEncoder", "bgeM3"
            ]
            if token == "--SentenceTransformer":
                model_type = "dense"
            elif token == "--SparseEncoder":
                model_type = "sparse"
            elif token == "--FlagReranker":
                model_type = "reranker"
            elif token == "--CrossEncoder":
                model_type = "crossEncoder"
            else:
                model_type = "bgeM3"
            kwargs: dict[str, Any] = {}
            next_idx = idx + 2
            if next_idx < len(argv):
                maybe_kwargs = argv[next_idx]
                if not maybe_kwargs.startswith("--"):
                    try:
                        parsed_kwargs = json.loads(maybe_kwargs)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(
                            f"Invalid JSON kwargs for {token} model {model_id!r}: {exc}"
                        ) from exc
                    if not isinstance(parsed_kwargs, dict):
                        raise SystemExit(
                            f"JSON kwargs for {token} model {model_id!r} must decode to an object."
                        )
                    kwargs = parsed_kwargs
                    next_idx += 1
            models.append((model_type, model_id, kwargs))
            idx = next_idx
            continue
        passthrough.append(token)
        idx += 1
    return models, passthrough


def _has_option(argv: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(item == option or item.startswith(prefix) for item in argv)


def _serialize_model_specs(
    model_specs: list[
        tuple[
            Literal["dense", "sparse", "reranker", "crossEncoder", "bgeM3"],
            str,
            dict[str, Any],
        ]
    ],
) -> str:
    return json.dumps(
        [
            {"type": model_type, "model_id": model_id, "kwargs": kwargs}
            for model_type, model_id, kwargs in model_specs
        ]
    )


def _load_model_config_from_env() -> Any:
    from meow_embed import server

    raw = os.environ.get(MODEL_CONFIG_ENV)
    if not raw:
        raise RuntimeError(f"Missing required env var: {MODEL_CONFIG_ENV}")
    items = json.loads(raw)
    if not isinstance(items, list):
        raise RuntimeError(f"Invalid {MODEL_CONFIG_ENV}: expected JSON array.")
    models: list[server.ModelInstanceConfig] = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid {MODEL_CONFIG_ENV}: item must be object.")
        model_type = item.get("type")
        model_id = item.get("model_id")
        kwargs = item.get("kwargs")
        if model_type not in ("dense", "sparse", "reranker", "crossEncoder", "bgeM3"):
            raise RuntimeError(
                f"Invalid model type in {MODEL_CONFIG_ENV}: {model_type!r}"
            )
        if not isinstance(model_id, str):
            raise RuntimeError(f"Invalid model_id in {MODEL_CONFIG_ENV}: {model_id!r}")
        if not isinstance(kwargs, dict):
            raise RuntimeError(f"Invalid kwargs in {MODEL_CONFIG_ENV}: {kwargs!r}")
        models.append(
            server.ModelInstanceConfig(
                type=model_type, model_id=model_id, kwargs=kwargs
            )
        )
    return server.ModelConfig(models=models)


def uvicorn_app_factory() -> Any:
    from meow_embed import server

    return server.build_app(_load_model_config_from_env())


def main() -> None:
    parse_args()
    model_specs, passthrough = parse_model_specs(sys.argv[1:])
    os.environ[MODEL_CONFIG_ENV] = _serialize_model_specs(model_specs)

    uvicorn_args = ["meow_embed.cli:uvicorn_app_factory", "--factory"]
    if not _has_option(passthrough, "--host"):
        uvicorn_args.extend(["--host", "0.0.0.0"])
    if not _has_option(passthrough, "--port"):
        uvicorn_args.extend(["--port", "8067"])
    uvicorn_args.extend(passthrough)

    uvicorn_main = importlib.import_module("uvicorn.main").main

    uvicorn_main(args=uvicorn_args, prog_name="uvicorn", standalone_mode=True)
