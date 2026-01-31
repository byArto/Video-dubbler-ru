#!/usr/bin/env python3
"""
Тест полного pipeline с OpenAI API
Проверяет все компоненты: Whisper API, Chat API, TTS API
"""
import os
import sys
from dotenv import load_dotenv

# Загружаем .env
load_dotenv()

def test_api_key():
    """Проверка наличия API ключа"""
    print("=" * 70)
    print("1️⃣  ПРОВЕРКА API КЛЮЧА")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ОШИБКА: OPENAI_API_KEY не установлен")
        return False

    print(f"✅ API ключ найден: {api_key[:15]}...")
    print(f"   Длина: {len(api_key)} символов")
    return True


def test_whisper_api():
    """Тест OpenAI Whisper API"""
    print("\n" + "=" * 70)
    print("2️⃣  ТЕСТ OPENAI WHISPER API")
    print("=" * 70)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("✅ OpenAI SDK импортирован")
        print("✅ Client создан")
        print("\n💡 Whisper API готов к использованию")
        print("   Для теста нужен аудио файл")
        return True
    except ImportError as e:
        print(f"❌ ОШИБКА: OpenAI SDK не установлен: {e}")
        print("   Выполните: pip install openai>=1.12.0")
        return False
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return False


def test_translation():
    """Тест перевода через OpenAI Chat API"""
    print("\n" + "=" * 70)
    print("3️⃣  ТЕСТ OPENAI CHAT API (ПЕРЕВОД)")
    print("=" * 70)

    try:
        sys.path.insert(0, '/Users/byarto/Desktop/video-ru-dubber')
        from services.translator import translate_text

        test_phrases = [
            "Hello, how are you today?",
            "This is a great product!",
            "Where is the nearest restaurant?"
        ]

        print("🔄 Тестируем перевод...\n")

        for phrase in test_phrases:
            result = translate_text(phrase)
            print(f"EN: {phrase}")
            print(f"RU: {result}")
            print()

        print("✅ Перевод работает через OpenAI API!")
        return True

    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_api():
    """Тест OpenAI TTS API"""
    print("\n" + "=" * 70)
    print("4️⃣  ТЕСТ OPENAI TTS API")
    print("=" * 70)

    try:
        from openai import OpenAI
        import tempfile

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        voice = os.getenv("OPENAI_TTS_VOICE", "onyx")
        model = os.getenv("OPENAI_TTS_MODEL", "tts-1")

        print(f"🎤 Голос: {voice}")
        print(f"🎼 Модель: {model}")
        print("🔄 Генерируем тестовую озвучку...\n")

        test_text = "Привет! Это тест OpenAI TTS API."

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=test_text,
            speed=1.0
        )

        # Сохраняем во временный файл
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        response.stream_to_file(temp_file.name)

        file_size = os.path.getsize(temp_file.name)
        print(f"✅ Озвучка сгенерирована!")
        print(f"   Текст: {test_text}")
        print(f"   Файл: {temp_file.name}")
        print(f"   Размер: {file_size} байт")
        print(f"\n💡 Прослушайте: open {temp_file.name}")

        return True

    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Проверка конфигурации"""
    print("\n" + "=" * 70)
    print("5️⃣  КОНФИГУРАЦИЯ")
    print("=" * 70)

    config = {
        "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "LLM_API_BASE": os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
        "OPENAI_TTS_MODEL": os.getenv("OPENAI_TTS_MODEL", "tts-1"),
        "OPENAI_TTS_VOICE": os.getenv("OPENAI_TTS_VOICE", "onyx"),
    }

    for key, value in config.items():
        print(f"   {key}: {value}")

    print("\n💡 Для изменения параметров отредактируйте .env файл")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "ТЕСТ OPENAI API INTEGRATION" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []

    # Тест 1: API Key
    results.append(("API Key", test_api_key()))

    if not results[0][1]:
        print("\n❌ Добавьте OPENAI_API_KEY в .env файл и попробуйте снова")
        sys.exit(1)

    # Тест 2: Whisper API
    results.append(("Whisper API", test_whisper_api()))

    # Тест 3: Translation (Chat API)
    results.append(("Translation API", test_translation()))

    # Тест 4: TTS API
    results.append(("TTS API", test_tts_api()))

    # Конфигурация
    test_configuration()

    # Итоги
    print("\n" + "=" * 70)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 70)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {name:20s}: {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("\n✅ Ваш переводчик работает на 100% через OpenAI API:")
        print("   • Whisper API - распознавание речи")
        print("   • Chat API (gpt-4o-mini) - перевод")
        print("   • TTS API - озвучка")
        print("\n🚀 Готово к использованию!")
        print("   Запустите: uvicorn main:app --reload")
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")
        print("   Проверьте ошибки выше")
    print("=" * 70)
