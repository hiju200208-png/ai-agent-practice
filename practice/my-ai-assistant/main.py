"""My AI Assistant — 기본은 API를 호출하지 않는 mock 모드입니다."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    measurement_mode: str = "mock"
    fallback_reason: str | None = None


class ChatClient(Protocol):
    def ask(self, question: str) -> Reply: ...


def get_max_output_tokens() -> int:
    """환경변수에서 출력 토큰 상한을 읽고 검사합니다."""
    value = os.getenv("MAX_OUTPUT_TOKENS", "128")

    try:
        tokens = int(value)
    except ValueError as exc:
        raise ValueError(
            "MAX_OUTPUT_TOKENS에는 128처럼 양의 정수를 입력해 주세요."
        ) from exc

    if tokens <= 0:
        raise ValueError("MAX_OUTPUT_TOKENS는 1 이상이어야 합니다.")

    return tokens


class MockClient:
    def ask(self, question: str) -> Reply:
        question = question.strip()

        if not question:
            raise ValueError("질문을 입력해야 합니다.")

        return Reply(
            text=(
                f"[MOCK 예시] 질문을 받았습니다: {question}\n"
                "실제 OpenAI API 응답이 아닙니다."
            ),
            model="mock-fixture",
        )


class OpenAIClient:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY를 설정해야 합니다.")

        if not model:
            raise ValueError("OPENAI_MODEL_NAME을 설정해야 합니다.")

        # .env에서 읽은 출력 토큰 상한을 저장합니다.
        self.max_output_tokens = get_max_output_tokens()

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Live 모드에는 pip install -r requirements.txt가 필요합니다."
            ) from exc

        self.client = OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=0,
        )
        self.model = model

    def ask(self, question: str) -> Reply:
        question = question.strip()

        if not question:
            raise ValueError("질문을 입력해야 합니다.")

        response = self.client.responses.create(
            model=self.model,
            input=question,
            # 고정값 대신 .env에서 읽은 값을 사용합니다.
            max_output_tokens=self.max_output_tokens,
        )

        usage = getattr(response, "usage", None)
        detail = (
            getattr(usage, "input_tokens_details", None)
            if usage is not None
            else None
        )

        return Reply(
            text=response.output_text or "",
            model=getattr(response, "model", self.model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cached_input_tokens=getattr(detail, "cached_tokens", None),
            measurement_mode="live",
        )


class GeminiClient:
    """OpenAI 호출 실패 시 폴백으로도 사용할 수 있습니다."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY를 설정해야 합니다.")

        if not model:
            raise ValueError("GEMINI_MODEL_NAME을 설정해야 합니다.")

        # Gemini도 같은 출력 토큰 설정을 사용합니다.
        self.max_output_tokens = get_max_output_tokens()

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Live 모드에는 pip install -r requirements.txt가 필요합니다."
            ) from exc

        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens
        )
        self.model = model

    def ask(self, question: str) -> Reply:
        question = question.strip()

        if not question:
            raise ValueError("질문을 입력해야 합니다.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=question,
            config=self.config,
        )

        usage = getattr(response, "usage_metadata", None)

        return Reply(
            text=response.text or "",
            model=getattr(response, "model_version", None) or self.model,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            cached_input_tokens=getattr(
                usage, "cached_content_token_count", None
            ),
            measurement_mode="live",
        )


class FallbackClient:
    """1순위 호출 실패 시 2순위 클라이언트로 한 번만 재시도합니다.

    설정 오류(ValueError)는 숨기지 않습니다.
    2순위 클라이언트는 필요할 때만 만듭니다.
    """

    def __init__(self, primary: ChatClient, make_secondary):
        self.primary = primary
        self.make_secondary = make_secondary

    def ask(self, question: str) -> Reply:
        try:
            return self.primary.ask(question)
        except ValueError:
            raise
        except Exception as exc:
            reason = type(exc).__name__

            try:
                secondary = self.make_secondary()
            except ValueError as config_exc:
                raise ValueError(
                    f"1순위 호출 실패({reason}) 후 "
                    f"2순위 설정 오류: {config_exc}"
                ) from exc

            return replace(
                secondary.ask(question),
                fallback_reason=reason,
            )


def make_client(
    mode: str,
    api_key: str | None = None,
    model: str | None = None,
    provider: str = "openai",
) -> ChatClient:
    if mode == "mock":
        return MockClient()

    if mode == "live" and provider == "gemini":
        return GeminiClient(
            api_key if api_key is not None
            else os.getenv("GEMINI_API_KEY", ""),
            model if model is not None
            else os.getenv("GEMINI_MODEL_NAME", ""),
        )

    if mode == "live":
        # OPENAI_MODEL_NAME을 우선 사용하고 기존 이름도 지원합니다.
        openai_model = (
            os.getenv("OPENAI_MODEL_NAME")
            or os.getenv("OPENAI_MODEL", "")
        )

        return OpenAIClient(
            api_key if api_key is not None
            else os.getenv("OPENAI_API_KEY", ""),
            model if model is not None
            else openai_model,
        )

    raise ValueError("mode는 mock 또는 live여야 합니다.")


def load_local_env() -> None:
    """이 파이썬 파일과 같은 폴더에 있는 .env를 읽습니다."""
    env_path = Path(__file__).resolve().parent / ".env"

    if not env_path.is_file():
        return

    supported_keys = {
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_MODEL_NAME",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_MODEL_NAME",
        "GEMINI_API_KEY",
        "GEMINI_MODEL_NAME",
        "MAX_OUTPUT_TOKENS",
    }

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        if (
            not line.strip()
            or line.lstrip().startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split("=", 1)
        key = key.strip()

        if key in supported_keys:
            # 이미 설정된 환경변수가 있으면 그 값을 우선합니다.
            os.environ.setdefault(
                key,
                value.strip().strip('"').strip("'"),
            )


def run(question: str, client: ChatClient) -> Reply:
    return client.ask(question)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="My AI Assistant — OpenAI runtime foundation"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="이번 과정에서 무엇을 만들까요?",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
    )
    parser.add_argument(
        "--fallback",
        choices=["none", "openai", "gemini"],
        default="none",
        help="1순위 호출이 API 오류로 실패하면 사용할 2순위 provider",
    )
    args = parser.parse_args(argv)

    try:
        if args.mode == "live":
            load_local_env()

        client = make_client(args.mode, provider=args.provider)

        if args.mode == "live" and args.fallback != "none":
            client = FallbackClient(
                client,
                lambda: make_client(
                    "live",
                    provider=args.fallback,
                ),
            )

        reply = run(args.question, client)

    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    except Exception as exc:
        # API 키나 요청 내용 등 민감정보는 출력하지 않습니다.
        print(
            f"API_ERROR: {type(exc).__name__} (민감정보 제외)",
            file=sys.stderr,
        )
        return 3

    print(reply.text)
    print(
        f"\n[mode={reply.measurement_mode}; model={reply.model}]",
        file=sys.stderr,
    )

    if reply.fallback_reason:
        print(
            f"[fallback used: primary failed with "
            f"{reply.fallback_reason}]",
            file=sys.stderr,
        )

    if reply.measurement_mode == "live":
        print(
            f"[usage input={reply.input_tokens}, "
            f"output={reply.output_tokens}, "
            f"cached_input={reply.cached_input_tokens}]",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())