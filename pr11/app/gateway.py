"""Шлюз: потокове виконання, журнал кожної спроби, кеш, бюджет, повтори."""
import logging
import math
import os
import threading
import time
from typing import Callable
from dotenv import load_dotenv
from . import cache, budget, costs
from .providers.base import Provider, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

load_dotenv()
PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
FALLBACK = os.getenv('LLM_FALLBACK', 'none')
TIMEOUT = float(os.getenv('LLM_TIMEOUT', '20'))
IDLE_TIMEOUT = float(os.getenv('LLM_IDLE_TIMEOUT', '8'))
TOTAL_TIMEOUT = float(os.getenv('LLM_TOTAL_TIMEOUT', '45'))
MAX_ATTEMPTS = int(os.getenv('LLM_MAX_ATTEMPTS', '3'))
BACKOFF_BASE = float(os.getenv('LLM_BACKOFF_BASE', '0.5'))
BACKOFF_MAX = float(os.getenv('LLM_BACKOFF_MAX', '8'))
MAX_CONCURRENT = max(1, int(os.getenv('LLM_MAX_CONCURRENT', '2')))
MODEL = os.getenv('LLM_MODEL')
TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '600'))
_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT)
_log = logging.getLogger('app.gateway')

def settings() -> dict:
    return {'provider': PROVIDER, 'fallback': FALLBACK, 'model': MODEL, 'timeout': TIMEOUT,
            'idle_timeout': IDLE_TIMEOUT, 'total_timeout': TOTAL_TIMEOUT,
            'max_attempts': MAX_ATTEMPTS, 'max_concurrent': MAX_CONCURRENT}

def make_provider(name: str) -> Provider:
    if name == 'mock':
        return MockProvider()
    return OpenAICompatProvider(base_url=os.getenv('LLM_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai/'),
                api_key=os.getenv('LLM_API_KEY') or 'missing-key',
                model=name if name != 'openai' else (MODEL or 'gemini-2.5-flash'), timeout=TIMEOUT)

def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    start = time.perf_counter()
    metadata = request.metadata
    metadata['provider'] = PROVIDER
    fields = {'draft_id': metadata.get('draft_id'), 'request_id': metadata.get('http_request_id'),
              'sample_id': metadata.get('sample_id'), 'client': metadata.get('client'),
              'prompt_version': metadata.get('prompt_version'),
              'input_chars': metadata.get('input_chars'), 'fact_count': metadata.get('fact_count')}
    key = cache.make_key(request, metadata.get('prompt_version', 'unknown'))
    fresh = bool(metadata.get('fresh', False))
    metadata['cache_key'] = key
    if fresh:
        cache.bypass()
        _log.info('cache.bypass', extra={**fields, 'cache_key': key})
    else:
        hit = cache.get(key)
        if hit:
            metadata['attempts'] = 0
            on_status({'attempt': 0, 'provider': hit.provider, 'cached': True})
            on_delta(hit.text)
            _log.info('cache.hit', extra={**fields, 'cache_key': key, 'cached': True,
                                         'cost': 0.0})
            _log.info('llm.summary', extra={**fields, 'cache_key': key, 'cached': True,
                 'attempts': 0, 'provider': hit.provider, 'model': hit.model,
                 'duration_ms': round((time.perf_counter()-start)*1000,2), 'cost': 0.0})
            return hit
        _log.info('cache.miss', extra={**fields, 'cache_key': key})

    attempt_count = 0
    last_error = None
    for provider_name, max_attempts, fallback in [(PROVIDER, MAX_ATTEMPTS, False),
                                                  *(([(FALLBACK, 1, True)] if FALLBACK != 'none' else []))]:
        if is_cancelled():
            raise ProviderError('cancelled', 'Генерацію скасовано.')
        provider = make_provider(provider_name)
        use_request = Request(messages=request.messages, model=(request.model if not fallback else
                      (None if provider_name == 'mock' else provider_name)),
                      max_tokens=request.max_tokens, temperature=request.temperature, metadata=metadata)
        for attempt in range(1, max(1, max_attempts)+1):
            attempt_count += 1
            metadata['attempts'] = attempt_count
            if is_cancelled():
                raise ProviderError('cancelled', 'Генерацію скасовано.')
            if time.perf_counter()-start > TOTAL_TIMEOUT:
                raise ProviderError('timeout', 'Вичерпано загальний час генерації.')
            try:
                budget.check()
            except budget.BudgetExceeded as exc:
                _log.warning('budget.rejected', extra={**fields, 'spent': exc.spent,
                             'limit': exc.limit, 'resets_at': exc.resets_at, 'kind': 'budget'})
                raise ProviderError('budget', str(exc), retryable=False) from None
            on_status({'attempt': attempt_count, 'provider': provider.name,
                       'fallback': fallback, 'detail': 'Генерація відповіді'})
            queue_start = time.perf_counter()
            acquired = _semaphore.acquire(timeout=max(.1, TOTAL_TIMEOUT - (queue_start-start)))
            if not acquired:
                raise ProviderError('timeout', 'Завелике очікування в черзі провайдера.', retryable=True)
            queued_ms = round((time.perf_counter()-queue_start)*1000,2)
            _log.info('llm.started', extra={**fields, 'cache_key': key, 'attempt': attempt_count,
                     'provider': provider.name, 'model': use_request.model or getattr(provider,'model',''),
                     'fallback': fallback, 'queue_ms': queued_ms})
            text = ''
            usage = None
            finish = None
            first_ms = None
            call_start = time.perf_counter()
            error = None
            try:
                chunks = provider.stream(use_request)
                try:
                    for chunk in chunks:
                        elapsed = time.perf_counter()-call_start
                        if is_cancelled():
                            raise ProviderError('cancelled', 'Генерацію скасовано.', retryable=False)
                        if elapsed > TIMEOUT or time.perf_counter()-start > TOTAL_TIMEOUT:
                            raise ProviderError('timeout', 'Час генерації вичерпано.', retryable=True)
                        if chunk.text:
                            if first_ms is None:
                                first_ms = round(elapsed*1000,2)
                            text += chunk.text
                            on_delta(chunk.text)
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if chunk.finish_reason is not None:
                            finish = chunk.finish_reason
                finally:
                    closer = getattr(chunks, 'close', None)
                    if closer:
                        closer()
            except ProviderError as exc:
                error = exc
            except Exception:
                _log.exception('llm.internal', extra={**fields, 'attempt': attempt_count})
                error = ProviderError('unknown', 'Технічна помилка генерації.', retryable=False)
            finally:
                _semaphore.release()
            duration_ms = round((time.perf_counter()-call_start)*1000,2)
            call_model = getattr(provider, 'last_model', None) or getattr(provider,'model',None) or use_request.model
            cost, estimated = costs.estimate(usage, provider=provider.name, model=call_model, text=text)
            # За обірваний частково згенерований потік могли списати токени.
            if usage is not None or text:
                budget.record(cost)
            payload = {**fields, 'cache_key': key, 'attempt': attempt_count,
                       'provider': provider.name, 'model': call_model,
                       'fallback': fallback, 'cached': False,
                       'queue_ms': queued_ms, 'first_token_ms': first_ms,
                       'duration_ms': duration_ms,
                       'status': 'error' if error else 'ok',
                       'kind': error.kind if error else None,
                       'prompt_tokens': (usage or {}).get('prompt_tokens'),
                       'completion_tokens': (usage or {}).get('completion_tokens'),
                       'usage': usage, 'cost': cost, 'cost_estimated': estimated}
            _log.log(logging.WARNING if error else logging.INFO, 'llm.attempt', extra=payload)
            if not error:
                metadata['first_token_s'] = round(first_ms/1000,3) if first_ms is not None else None
                result = Result(text=text, provider=provider.name, model=call_model or '',
                                usage=usage, finish_reason=finish,
                                elapsed=duration_ms/1000,
                                cost=cost, cost_estimated=estimated)
                if not fallback:
                    cache.put(key, result)
                _log.info('llm.summary', extra={**fields, 'cache_key': key, 'attempts': attempt_count,
                     'provider': provider.name, 'model': result.model, 'cached': False,
                     'duration_ms': round((time.perf_counter()-start)*1000,2),
                     'cost': cost, 'fallback': fallback})
                return result
            last_error = error
            # Після часткової streaming-відповіді не починаємо іншу:
            # інакше оператор побачить зліплений з двох спроб текст.
            if error.kind in {'budget','cancelled'} or text or not error.retryable:
                break
            if attempt < max_attempts:
                pause = error.retry_after if error.retry_after is not None else min(BACKOFF_MAX, BACKOFF_BASE * 2**(attempt-1))
                if time.perf_counter()-start+pause >= TOTAL_TIMEOUT:
                    break
                time.sleep(max(0, pause))
        if last_error and (last_error.kind in {'budget','cancelled'} or text):
            break
    _log.error('llm.summary', extra={**fields, 'cache_key': key, 'attempts': attempt_count,
                 'status': 'error', 'kind': last_error.kind if last_error else 'unknown',
                 'duration_ms': round((time.perf_counter()-start)*1000,2)})
    raise last_error or ProviderError('unknown','Не вдалося створити чернетку.')
