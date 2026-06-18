from google import genai
import csv
from pathlib import Path
from datetime import datetime
import time
from google.genai import types
from dotenv import load_dotenv

load_dotenv()   

client = genai.Client()

MODEL_NAME = "gemini-2.5-flash"
LOG_FILE = Path("llm_logs.csv")
INPUT_PRICE_PER_MILLION = 0.10
OUTPUT_PRICE_PER_MILLION = 0.40

def estimate_cost(
    input_tokens: int,
    output_tokens: int
) -> float:
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION

    return input_cost + output_cost

def log_request(
    query: str,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    cost: int,
):
    file_exist = LOG_FILE.exists()
    
    with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exist:
            writer.writerow([
                "timestamp",
                "model",
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "query"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            MODEL_NAME,
            latency_ms,
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            round(cost, 8),
            query
        ])
        


def call_api(
    system_prompt: str,
    user_prompt: str,
    user_query: str,
    max_retries: int = 3
) -> str:

    retry_delay = 1

    print("=" * 50)
    print("CALL_API_START")

    for attempt in range(max_retries):

        print(f"ATTEMPT: {attempt + 1}")

        try:

            print("SYSTEM LENGTH:", len(system_prompt))
            print("USER LENGTH:", len(user_prompt))

            start_time = time.time()

            print("BEFORE GENERATE_CONTENT")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2
                )
            )

            print("AFTER GENERATE_CONTENT")

            latency_ms = int(
                (time.time() - start_time) * 1000
            )

            print("LATENCY:", latency_ms)

            usage = response.usage_metadata

            print("USAGE:", usage)

            input_tokens = getattr(
                usage,
                "prompt_token_count",
                0
            )

            output_tokens = getattr(
                usage,
                "candidates_token_count",
                0
            )

            print("INPUT TOKENS:", input_tokens)
            print("OUTPUT TOKENS:", output_tokens)

            cost = estimate_cost(
                input_tokens,
                output_tokens
            )

            log_request(
                query=user_query,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )

            print("CALL_API_DONE")

            return response.text

        except Exception as e:

            print("CALL_API_EXCEPTION")
            print(type(e))
            print(repr(e))
            print(str(e))

            error_message = str(e).lower()

            if (
                "api key" in error_message
                or "permission" in error_message
                or "401" in error_message
            ):
                raise ValueError(
                    "Invalid Gemini API key"
                )

            if (
                "429" in error_message
                or "rate limit" in error_message
                or "quota" in error_message
            ):
                print(
                    f"Retry after {retry_delay}s"
                )

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

            raise

    raise RuntimeError(
        "Gemini request failed after retries"
    )