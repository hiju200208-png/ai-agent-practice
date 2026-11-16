import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI, APIConnectionError, APIStatusError


def main():
    # 현재 작업 위치와 관계없이 이 파일 옆의 .env를 읽습니다.
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        print("model_info.py와 같은 폴더에 .env 파일을 만들어 주세요.")
        return 1

    settings = dotenv_values(env_path, encoding="utf-8-sig")
    api_key = (settings.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print(".env에 OPENAI_API_KEY=발급받은키 를 작성해 주세요.")
        return 1

    # API 키는 화면에 출력하지 않습니다.
    with OpenAI(api_key=api_key, timeout=30.0, max_retries=0) as client:
        try:
            # 이 키로 조회되는 모델 목록을 가져옵니다.
            models = client.models.list()
            model_names = sorted({model.id for model in models})

        except APIConnectionError:
            print("연결 실패: 인터넷 연결과 방화벽 설정을 확인해 주세요.")
            return 1

        except APIStatusError as error:
            # 오류 응답 원문 대신 상태 코드와 안내만 출력합니다.
            messages = {
                401: "API 키가 올바른지 확인해 주세요.",
                403: "이 키의 모델 목록 조회 권한과 조직/프로젝트 권한을 확인해 주세요.",
                429: "요청 제한 또는 사용 한도를 확인한 뒤 다시 시도해 주세요.",
            }
            print(f"조회 실패 (HTTP {error.status_code})")
            print(
                messages.get(
                    error.status_code,
                    "OpenAI 서비스 상태와 API 설정을 확인해 주세요.",
                )
            )
            return 1

    print(f"\n[이 API 키로 조회된 전체 모델: {len(model_names)}개]")
    for name in model_names:
        print(f"- {name}")

    # 모델 목록에는 용도 필드가 없어 이름으로 임베딩 모델을 구분합니다.
    embedding_names = [
        name for name in model_names if "embedding" in name.lower()
    ]

    print(f"\n[임베딩 모델 (이름 기준 분류): {len(embedding_names)}개]")
    for name in embedding_names:
        print(f"- {name}")

    if not embedding_names:
        print("조회된 목록에 임베딩 모델이 없습니다.")

    print("\n주의: 모델 목록 조회 권한과 실제 모델 호출 권한은 다릅니다.")
    print("목록에 있어도 프로젝트 제한, 결제/한도, 호출 API에 따라 요청이 실패할 수 있습니다.")
    print("이 코드는 목록만 조회하며 실제 생성/임베딩 호출은 하지 않습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())