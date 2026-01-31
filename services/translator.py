import os
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Global translator instance
_translator = None
_tokenizer = None


def get_translator():
    """Load MarianMT translator (cached)."""
    global _translator, _tokenizer
    if _translator is None:
        from transformers import MarianMTModel, MarianTokenizer

        model_name = "Helsinki-NLP/opus-mt-en-ru"
        logger.info(f"Loading translation model: {model_name}")

        _tokenizer = MarianTokenizer.from_pretrained(model_name)
        _translator = MarianMTModel.from_pretrained(model_name)

        logger.info("Translation model loaded")

    return _translator, _tokenizer


def _preserve_punctuation(original: str, translated: str) -> str:
    """Preserve original punctuation in translation."""
    original = original.strip()
    translated = translated.strip()

    if original.endswith('?') and not translated.endswith('?'):
        translated = translated.rstrip('.!') + '?'
    elif original.endswith('!') and not translated.endswith('!'):
        translated = translated.rstrip('.?') + '!'
    elif original.endswith('...') and not translated.endswith('...'):
        translated = translated.rstrip('.') + '...'

    return translated


def _post_process_translation(text: str) -> str:
    """Post-process translation for better quality."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s\d])', r'\1 \2', text)

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def translate_text(text: str, context: str = None) -> str:
    """
    Translate text from English to Russian.
    Priority: Yandex → LLM → MarianMT
    """
    if not text.strip():
        return ""

    # Try Yandex Translate first
    yandex_key = os.getenv("YANDEX_TRANSLATE_API_KEY")
    yandex_folder = os.getenv("YANDEX_FOLDER_ID")
    if yandex_key and yandex_folder:
        try:
            result = _translate_with_yandex(text, yandex_key, yandex_folder)
            return _post_process_translation(result)
        except Exception as e:
            logger.warning(f"Yandex translation failed: {e}")

    # Try LLM API
    llm_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if llm_key:
        try:
            result = _translate_with_llm(text, llm_key, context)
            return result
        except Exception as e:
            logger.warning(f"LLM translation failed: {e}")

    # Fallback to local MarianMT
    translated = _translate_local(text)
    translated = _post_process_translation(translated)
    translated = _preserve_punctuation(text, translated)

    return translated


def _translate_with_yandex(text: str, api_key: str, folder_id: str) -> str:
    """
    Translate using Yandex Translate API.
    Supports both IAM token and API key authentication.
    Role required: ai.translate.user or yc.ai.translate.execute
    """
    import requests

    url = "https://translate.api.cloud.yandex.net/translate/v2/translate"

    # Detect auth type
    # API keys start with AQVN (short, ~40 chars)
    # IAM tokens start with t1. or are very long (>200 chars)
    if api_key.startswith('t1.') or len(api_key) > 200:
        auth_header = f"Bearer {api_key}"
    else:
        # API key (AQVN...)
        auth_header = f"Api-Key {api_key}"

    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }

    body = {
        "folderId": folder_id,
        "texts": [text],
        "targetLanguageCode": "ru",
        "sourceLanguageCode": "en"
    }

    logger.info(f"Yandex request: folder={folder_id}, auth={auth_header[:20]}..., key_len={len(api_key)}")
    response = requests.post(url, headers=headers, json=body, timeout=30)
    logger.info(f"Yandex response: {response.status_code}")

    if response.status_code == 403:
        try:
            error_detail = response.json().get('message', response.text)
        except:
            error_detail = response.text
        raise PermissionError(f"Yandex API 403: {error_detail}")

    if response.status_code != 200:
        logger.error(f"Yandex API error {response.status_code}: {response.text}")
        response.raise_for_status()

    result = response.json()
    translated = result["translations"][0]["text"]

    logger.info(f"Yandex translated: {text[:50]}... -> {translated[:50]}...")
    return translated


def _translate_local(text: str) -> str:
    """Translate using local MarianMT model."""
    model, tokenizer = get_translator()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs, num_beams=5, early_stopping=True)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)

    return result


def _translate_with_llm(text: str, api_key: str, context: str = None) -> str:
    """Translate using OpenAI-compatible API with retry logic."""
    import requests
    import time

    base_url = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    system_prompt = """Ты профессиональный переводчик с английского на русский.

Правила:
1. Переводи естественно, как говорят носители русского языка
2. Сохраняй эмоциональную окраску и тон оригинала
3. Идиомы переводи на русские аналоги, а не дословно
4. Сохраняй пунктуацию (?, !, ...)
5. Отвечай ТОЛЬКО переводом, без пояснений"""

    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({"role": "user", "content": f"Контекст: {context}"})
        messages.append({"role": "assistant", "content": "Понял контекст."})

    messages.append({"role": "user", "content": text})

    # Retry logic for rate limits
    max_retries = 5
    for attempt in range(max_retries):
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 500
            },
            timeout=30
        )

        if response.status_code == 429:
            wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
            logger.info(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"OpenAI translated: {text[:30]}... -> {result[:30]}...")
        return result

    # If all retries failed, raise the last error
    response.raise_for_status()


def _translate_batch_with_llm(texts: List[str], api_key: str) -> List[str]:
    """Translate multiple texts in a single API call for speed."""
    import requests
    import time
    import json

    if not texts:
        return []

    base_url = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Формируем текст для перевода с номерами строк
    numbered_texts = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))

    system_prompt = """Ты профессиональный переводчик с английского на русский.

Правила:
1. Переводи естественно, как говорят носители русского языка
2. Сохраняй эмоциональную окраску и тон оригинала
3. Идиомы переводи на русские аналоги, а не дословно
4. Сохраняй пунктуацию (?, !, ...)
5. Отвечай ТОЛЬКО переводами в том же формате (номер. перевод)
6. Сохраняй нумерацию строк"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": numbered_texts}
    ]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 4000
                },
                timeout=120
            )

            if response.status_code == 429:
                wait_time = 2 ** attempt
                logger.info(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()

            # Парсим ответ - извлекаем переводы по номерам
            translations = {}
            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Ищем формат "N. текст" или "N) текст"
                match = re.match(r'^(\d+)[.\)]\s*(.+)$', line)
                if match:
                    num = int(match.group(1))
                    text = match.group(2).strip()
                    translations[num] = text

            # Собираем результаты в правильном порядке
            translated = []
            for i in range(len(texts)):
                if (i + 1) in translations:
                    translated.append(translations[i + 1])
                else:
                    # Fallback - оставляем оригинал если перевод не найден
                    logger.warning(f"Translation not found for segment {i + 1}")
                    translated.append(texts[i])

            logger.info(f"Batch translated {len(texts)} segments in one request")
            return translated

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Batch translation attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
            else:
                raise

    return texts  # Fallback


def translate_segments(segments: List[Dict]) -> List[Dict]:
    """Translate all segments from EN to RU using batch processing."""
    logger.info(f"Translating {len(segments)} segments")

    # Извлекаем тексты для перевода
    texts = [seg['text'] for seg in segments]

    # Пробуем batch-перевод через LLM (быстрее)
    llm_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if llm_key:
        try:
            # Разбиваем на батчи по 30 сегментов (чтобы не превысить лимит токенов)
            batch_size = 30
            all_translations = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Translating batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
                batch_translations = _translate_batch_with_llm(batch, llm_key)
                all_translations.extend(batch_translations)

            # Формируем результат
            translated = []
            for i, seg in enumerate(segments):
                translated.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text_en': seg['text'],
                    'text_ru': all_translations[i] if i < len(all_translations) else seg['text']
                })

            logger.info("Translation complete (batch mode)")
            return translated

        except Exception as e:
            logger.warning(f"Batch translation failed, falling back to sequential: {e}")

    # Fallback: последовательный перевод (медленнее, но надежнее)
    translated = []
    for i, seg in enumerate(segments):
        ru_text = translate_text(seg['text'], None)
        translated.append({
            'start': seg['start'],
            'end': seg['end'],
            'text_en': seg['text'],
            'text_ru': ru_text
        })
        logger.debug(f"Segment {i+1}: {seg['text']} -> {ru_text}")

    logger.info("Translation complete (sequential mode)")
    return translated
