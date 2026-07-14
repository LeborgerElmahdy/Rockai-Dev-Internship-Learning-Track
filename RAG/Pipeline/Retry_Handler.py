import re
import time
import random
import socket
import httpx
from google import genai
from dotenv import load_dotenv
from google.genai import errors
from dataclasses import dataclass, field

load_dotenv()
client = genai.Client()

@dataclass
class Retry_Config:
    max_attempts: int = 10
    base_delay: float = 1.0
    max_delay: float = 30.0

    retry_status_codes = (429, 500, 503, 504)

# Digs into error logs to see if gemini api attached an exact number of seconds to wait before retry
def _extract_retry_delay(e) -> float | None:
    """Pull retryDelay out of the error's details, if present."""
    try:
        for detail in e.details.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                seconds = re.match(r"([\d.]+)s", detail["retryDelay"])
                if seconds:
                    return float(seconds.group(1))
    except (AttributeError, KeyError, TypeError):
        pass
    return None

def _handle_API_errors(e, attempt, model):
    "Handles google.genai.errors.APIError"
    status_reason = getattr(e, "status", "UNKNOWN")

    if e.code in Retry_Config.retry_status_codes:
        if attempt == Retry_Config.max_attempts:
            print(f"Max attempts ({Retry_Config.max_attempts}) reached. Giving up.")
            raise e

        exp_delay = min(
            random.uniform(0.5, 1.5) * Retry_Config.base_delay * (2 ** (attempt - 1)),
            Retry_Config.max_delay,
        )
        retry_delay = _extract_retry_delay(e) or exp_delay
        print(f"Code {e.code}. Backing off. Waiting {retry_delay:.2f}s...")
        time.sleep(retry_delay)
        return

    if e.code == 404 or status_reason == "NOT_FOUND":
        print(f"Model Not Found [404]: '{model}' does not exist or is deprecated.")
    elif e.code in (401, 403) or status_reason == "PERMISSION_DENIED":
        print(f"Authentication Failed [{e.code}]: Check your API key, project quotas, or permissions.")
    elif e.code == 400 or status_reason == "INVALID_ARGUMENT":
        print(f"Invalid Argument [400]: Malformed structure or prompt content limits exceeded.")
    else:
        print(f"Non-retryable error [{status_reason}] ({e.code}): {e.message}")
    raise e

def _handle_network_error(e, attempt, model):
    "Handles connection-level failures"
    if attempt == Retry_Config.max_attempts:
        print(f"Max attempts reached after network failure: {type(e).__name__}")
        raise e

    delay = min(Retry_Config.base_delay * (2 ** (attempt - 1)), Retry_Config.max_delay)
    print(f"Network error ({type(e).__name__}: {e}). Retrying in {delay:.2f}s...")
    time.sleep(delay)

def _validate_response(response, model):
    """Handles the '200 OK but the payload is wrong' case — Gemini can return
    a successful response with no embeddings, e.g. if content was safety-blocked."""
    if not getattr(response, "embeddings", None):
        raise ValueError(
            f"Empty embeddings returned by '{model}' — "
            f"input may have been blocked by safety filters or was empty after processing."
        )
    
def call_function_with_handling(fn, *args, model, **kwargs):
    attempt = 0
    while True:
        attempt += 1
        try:
            response = fn(*args,model = model, **kwargs)
            _validate_response(response, model)
            return response

        except errors.APIError as e:
            _handle_API_errors(e, attempt, model)

        except (httpx.ConnectError, httpx.TimeoutException, socket.gaierror, ConnectionError) as e:
            _handle_network_error(e, attempt, model)

        except ValueError:
            raise