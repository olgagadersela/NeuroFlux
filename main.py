# main.py 

import os
import json
import faiss
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import re
import random
from openai import OpenAI

# --- Telegram bot ---
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters
)
import asyncio
import textwrap
import io
from datetime import datetime, timezone

# --- Данные меню ---
from data import (
    INTERESTING_FACTS,
    CLUSTERS,
    PLANETS,
    THEMES,
    FORMATS,
    AUTHOR_MASKS
)


# --- Конфигурация ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("TELEGRAM_TOKEN =", os.environ.get("TELEGRAM_TOKEN"))
print("OPENAI_API_KEY =", os.environ.get("OPENAI_API_KEY"))

if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN не найден. Проверь .env (локально) или Railway Variables.")
if OPENAI_API_KEY:
    print("✅ OPENAI_API_KEY загружен")
else:
    raise ValueError("❌ OPENAI_API_KEY не найден в переменных окружения Railway")

if TELEGRAM_TOKEN is None:
    print("⚠️ TELEGRAM_TOKEN не найден в окружении")
else:
    print("✅ TELEGRAM_TOKEN загружен")

# --- Инициализация ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Логирование ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Константы ---
EMBEDDINGS_DIR = "embeddings"
INDEX_DIR = "faiss_indices"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

DATA_FILES = ["params.md", "postformatdb.md", "clasterDB.md", "author_masks_db.md", "effectdb.md"]

# ==============================================================================
#  story_bot.py
# ==============================================================================

def read_md_file(md_path: str) -> List[Dict]:
    entries, current = [], {}
    try:
        with open(md_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("<!--") and "type:" in line:
                    if "name" in current and "description" in current:
                        entries.append(current.copy())
                    current = {}
                    current["type"] = line.split("type:")[1].split("-->")[0].strip()
                elif line.startswith("# "):
                    current["name"] = line[2:].strip()
                elif line.startswith("<!-- tags:"):
                    tags_value = line.split("<!-- tags:")[1].split("-->")[0].strip()
                    current["tags"] = [tag.strip() for tag in tags_value.split(",")] if tags_value else []
                elif line.startswith("<!-- cluster:"):
                    cluster_value = line.split("<!-- cluster:")[1].split("-->")[0].strip()
                    current["cluster"] = cluster_value
                elif line.startswith("<!-- planet:"):
                    planet_value = line.split("<!-- planet:")[1].split("-->")[0].strip()
                    current["planet"] = planet_value
                elif line.startswith("<!-- "):
                    new_tags = [
                        "narrative_voice", "narrative_arc",
                        "time_mode", "silence_weight", "rupture", "forbidden", "allowed",
                        "visual", "signature", "palette", "final_stroke"
                    ]
                    for tag in new_tags:
                        if line.startswith(f"<!-- {tag}:"):
                            value = line.split(":", 1)[1].strip()
                            if value.endswith("-->"):
                                value = value[:-3].strip()
                            current[tag] = value
                            break
                elif line and not line.startswith("<!--"):
                    if "name" in current:
                        if "description" in current:
                            current["description"] += "\n" + line.strip()
                        else:
                            current["description"] = line.strip()
        if "name" in current and "description" in current:
            entries.append(current.copy())
    except Exception as e:
        logger.error(f"Ошибка при чтении {md_path}: {e}")
    return entries

def get_openai_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Ошибка создания эмбеддинга: {str(e)}")
        raise

def parse_description_sections(description: str) -> dict:
    sections = {}
    pattern = r"^##\s+([^\n]+)\n(.*?)(?=\n##\s|\Z)"
    matches = re.findall(pattern, description, re.DOTALL | re.MULTILINE)
    for title, content in matches:
        key = title.strip().lower().replace(" ", "_")
        sections[key] = content.strip()
    return sections

def build_index(md_file: str):
    try:
        name = Path(md_file).stem
        entries = read_md_file(md_file)
        logger.info(f"Parsed {len(entries)} entries from {md_file}")
        if not entries:
            logger.warning(f"Нет записей для индексации в {md_file}")
            return None, []
        for entry in entries:
            description = entry.get("description", "")
            entry["description_meta"] = parse_description_sections(description)
        texts = [e.get("description", e.get("name", "")) for e in entries]
        vectors = []
        for text in texts:
            try:
                embedding = get_openai_embedding(text)
                if not embedding or len(embedding) == 0:
                    logger.error(f"Пустой эмбеддинг для: {text[:100]}...")
                    return None, []
                vectors.append(embedding)
            except Exception as e:
                logger.error(f"Ошибка эмбеддинга для '{text[:50]}...': {str(e)}")
                return None, []
        if not vectors:
            logger.error("Не удалось получить ни одного эмбеддинга!")
            return None, []
        vectors = np.array(vectors).astype("float32")
        if vectors.shape[0] == 0 or vectors.shape[1] == 0:
            logger.error("Некорректная размерность векторов!")
            return None, []
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, os.path.join(INDEX_DIR, f"{name}.index"))
        with open(os.path.join(INDEX_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        return index, entries
    except Exception as e:
        logger.error(f"Ошибка построения индекса для {md_file}: {str(e)}")
        try:
            index_path = os.path.join(INDEX_DIR, f"{Path(md_file).stem}.index")
            json_path = os.path.join(INDEX_DIR, f"{Path(md_file).stem}.json")
            for fp in [index_path, json_path]:
                if os.path.exists(fp):
                    os.remove(fp)
                    logger.info(f"Удалён битый файл: {fp}")
        except Exception as cleanup_error:
            logger.error(f"Ошибка очистки: {cleanup_error}")
        return None, []

def load_index(md_file: str) -> Tuple[Optional[faiss.Index], List[Dict]]:
    name = Path(md_file).stem
    index_path = os.path.join(INDEX_DIR, f"{name}.index")
    json_path = os.path.join(INDEX_DIR, f"{name}.json")
    if os.path.exists(index_path) and os.path.exists(json_path):
        index = faiss.read_index(index_path)
        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if index.ntotal == 0 or len(entries) == 0:
            logger.warning(f"Индекс {name} пустой! Перестраиваем...")
            return build_index(md_file)
        return index, entries
    else:
        logger.info(f"Индекс для {md_file} не найден, строим новый...")
        return build_index(md_file)

def ensure_all_indices():
    success_count = 0
    for file in DATA_FILES:
        if os.path.exists(file):
            index, entries = load_index(file)
            if index is not None and entries:
                success_count += 1
            else:
                logger.error(f"Не удалось создать индекс для {file}")
    logger.info(f"Успешно загружено/построено {success_count}/{len(DATA_FILES)} индексов")

def find_closest_theme(problem_text: str, md_file="clasterDB.md", threshold=0.6):
    try:
        index, entries = load_index(md_file)
        if not index or not entries:
            logger.warning("[find_closest_theme] Индекс или записи не загружены.")
            return None
        embedding_list = get_openai_embedding(problem_text)
        embedding = np.array([embedding_list]).astype("float32")
        D, I = index.search(embedding, k=1)
        if D[0][0] > threshold:
            logger.info(f"[find_closest_theme] Расстояние {D[0][0]} > порога {threshold}. Тема не найдена.")
            return None
        idx = I[0][0]
        if idx >= len(entries):
            logger.error(f"[find_closest_theme] Недопустимый индекс: {idx} >= {len(entries)}")
            return None
        result = entries[idx]
        logger.debug(f"[find_closest_theme] D: {D}, I: {I}")
        logger.info(f"[find_closest_theme] Найдена тема: {result.get('name')}")
        return result
    except Exception as e:
        logger.error(f"[find_closest_theme] Ошибка поиска темы: {str(e)}")
        return None

def select_parameter(context: str, param_type: str) -> Optional[Dict]:
    try:
        index, db = load_index("params.md")
        filtered_db = [e for e in db if e.get("type") == param_type]
        if not filtered_db:
            logger.warning(f"Нет записей типа '{param_type}' в params.md")
            return None
        if context.strip().lower() in ["", "общая тема"]:
            return random.choice(filtered_db) if filtered_db else None
        query_emb = get_openai_embedding(context)
        texts = [e.get("description", e.get("name", "")) for e in filtered_db]
        if not texts:
            return random.choice(filtered_db) if filtered_db else None
        vectors = np.array([get_openai_embedding(text) for text in texts]).astype("float32")
        temp_index = faiss.IndexFlatL2(vectors.shape[1])
        temp_index.add(vectors)
        D, I = temp_index.search(np.array([query_emb]), 1)
        if I[0][0] != -1:
            return filtered_db[I[0][0]]
        else:
            return random.choice(filtered_db) if filtered_db else None
    except Exception as e:
        logger.error(f"Ошибка выбора параметра {param_type}: {e}")
        try:
            _, db = load_index("params.md")
            filtered_db = [e for e in db if e.get("type") == param_type]
            return random.choice(filtered_db) if filtered_db else None
        except:
            return None

def find_similar_by_tags(target_entry: Dict, db_entries: List[Dict], top_k: int = 3) -> List[Dict]:
    target_tags = set(target_entry.get("tags", []))
    if not target_tags:
        return random.sample(db_entries, min(top_k, len(db_entries)))
    structural_tags = ["narrative_arc", "time_mode", "silence_weight", "rupture"]
    for tag in structural_tags:
        if value := target_entry.get(tag):
            target_tags.add(f"##{tag}##{value}")
    similarities = []
    for entry in db_entries:
        if entry is target_entry:
            continue
        entry_tags = set(entry.get("tags", []))
        if not entry_tags:
            continue
        intersection = target_tags.intersection(entry_tags)
        union = target_tags.union(entry_tags)
        jaccard_sim = len(intersection) / len(union) if union else 0
        similarities.append((entry, jaccard_sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [entry for entry, _ in similarities[:top_k]]

class PromptBuilder:
    def __init__(self, data: Dict):
        self.data = data
    def build(self):
        prompt_parts = []
        # 1. Роль и позиция
        prompt_parts.append(
            "ТВОЯ РОЛЬ И ПОЗИЦИЯ:\n"
            "Ты не просто исполнитель, а автор с собственным голосом. "
            "Твоя задача — создавать тексты, которые не только информируют, "
            "но и вызывают эмоциональный отклик. Ты пишешь не ради красивых слов, "
            "а чтобы сформировать устойчивую связь между текстом и читателем.\n"
            "В мире, где все звучат одинаково, ты — голос, который узнают и запоминают."
        )
        prompt_parts.append(
            f"В центре истории — одна внутренняя проблема (вводи плавно, а не в лоб ). Цель — наглядно, очень подробно и глубоко раскрыть тему {self.data.get('theme', 'не указана')} "
            f"и преодолеть проблему {self.data.get('problem', 'не указана')} через личную трансформацию главного персонажа."
        )
        prompt_parts.append("Всё должно быть построено вокруг проблемы/темы. Всё. Раскладывай тему/проблему по частям, чтобы читатель мог понять смысл")
        prompt_parts.append(
            f"Cоздаем и вписываем согласно прочим параметрам вроде маски автора и темы в повествование профиль главного персонажа (живой человек): опиши пол, задай персонажу возраст от 20 до 50 лет, внешность (опиши различные в том числе отличительные детали), придумай психологический типаж согласно контексту"
            f"и при необходимости архетип личности для более проработанного персонажа."
        )
        # 2. Техники повествования
        prompt_parts.append("\n---\nТехники повествования\n"
            "История — это не только 'что произошло', но и почему это важно.\n"
            "Продумывай сложное многослойное повествование.\n"
            "В зависимости от контекста используй неожиданные повороты решения, чтобы повествование не казалось нудным.\n"
            "Говори с читателем, а не вещай ему.\n"
            "Не упрощай как в голливудском кино, пусть истории и подходы к решению проблем будут практичными и реальными а не глянцевыми и обобщенными.\n"
            "Не используй штампы вроде: 'Она почувствовала', 'Она осознала', 'Он взял блокнот и записал' и т.д.\n"
            "Подробно опиши, как герой преодолевает проблему.\n"
            "Используй примеры из реальной жизни.\n"
            "Задавай значимые вопросы.\n"
            
            "Создавай неожиданные связи между идеями.\n"
            "Признавай сложность темы, но объясняй просто.\n"
            "Избегать повторов ключевых слов и образов, использовать вариации и синонимы.\n"
            "Добавлять уникальные микродетали (звук, запах, текстура, свет), которые создают эффект присутствия.\n"
            "Чередовать длинные образные фразы с короткими и резкими предложениями для рваного ритма.\n"
            "Раскрой научную идею корректировки (паттернов) мышления через : методы, предлагаемые разными школами психологической терапии (выбирай наиболее подходящие), подробно описывай решение и сам путь изменения жизни главного героя на практике в деталях ( НЕ общими фразами вроде он решил пойти рисовать, танцевать, в кафе записать мысли, а более научно обосонованные описания которые отвечают на вопросы Почему именно туда? Что главный герой чувстсвует, ощущает? каким образо это помогает"
            "Учти, что мир разнообращен выбирай как городские мета вроде  кафе так и сельские, ближе к природе - мысли широко и разнообразно"
            "Вности рахнообразие когда предлагаешь терапевтические подходы, не предлагай только рисование если уместны другие подходы"
            "Включать неожиданные конкретные сенсорные детали (например, скрип ткани, вкус металлического воздуха), чтобы текст не был излишне абстрактным.\n"
            "Сохранять баланс: метафоры должны переплетаться с осязаемыми предметами, чтобы читатель не терялся в слишком высокой абстракции.\n"
            "В финале сцены оставлять “эхо” — последнюю деталь или образ, который продолжает жить после прочтения."
        )
        # 3. Маска автора (перемещена вверх)
        prompt_parts.append("\n---\n🖋 МАСКА АВТОРА — отвечает за голос и подачу. Применяй все её элементы в каждом абзаце. Если возникает конфликт с форматом повествования — приоритет за маской.")
        author_mask_type = self.data.get('author_mask')
        if not author_mask_type:
            prompt_parts.append("Маска автора не задана.")
        else:
            try:
                _, author_db = load_index("author_masks_db.md")
                mask_entry = next((e for e in author_db if e.get("name") == author_mask_type), None)
                if mask_entry is None:
                    prompt_parts.append(f"Маска '{author_mask_type}' не найдена в базе.")
                else:
                    prompt_parts.append("\nРАСШИФРОВКА МАСКИ АВТОРА:")
                    description_meta = mask_entry.get("description_meta", {})
                    def safe_get_mask(key):
                        return description_meta.get(key, None)
                    for key, label in [
                        ("lexicon", "Лексикон"),
                        ("tempo", "Темп"),
                        ("pov", "POV"),
                        ("techniques", "Приёмы"),
                        ("sensory_palette", "Сенсорная палитра"),
                        ("signature_phrase", "Сигнатурная фраза (всегда в финале)")
                    ]:
                        val = safe_get_mask(key)
                        if val:
                            prompt_parts.append(f"{label}: {val}")
            except Exception as e:
                prompt_parts.append(f"Ошибка загрузки маски автора: {e}")
        # 4. Формат
        prompt_parts.append("\n---\n🎭 ФОРМАТ — отвечает за структуру повествования, которая должна быть таковой, чтобы достигнуть заданной форматом Цели, но не за суть проблемы. Показывает как надо подавать, но не что надо подавать")
        format_type = self.data.get('format_type')
        if not format_type:
            prompt_parts.append("Формат повествования не задан.")
        else:
            try:
                _, format_db = load_index("postformatdb.md")
                format_entry = next((e for e in format_db if e.get("name") == format_type), None)
                if format_entry is None:
                    prompt_parts.append(f"Формат повествования '{format_type}' не найден в базе.")
                else:
                    prompt_parts.append("\n---\nРАСШИФРОВКА ФОРМАТА ПОВЕСТВОВАНИЯ:")
                    def safe_get(key):
                        return format_entry.get(key, None)
                    narrative_voice = safe_get("narrative_voice")
                    if narrative_voice:
                        prompt_parts.append(f"Голос повествования: {narrative_voice}")
                    narrative_arc = safe_get("narrative_arc")
                    if narrative_arc:
                        prompt_parts.append(f"Форма дуги: {narrative_arc} — история не линейна, возвращается с новым уровнем.")
                    time_mode = safe_get("time_mode")
                    if time_mode:
                        prompt_parts.append(f"Режим времени: {time_mode} — время здесь не течёт, а дышит.")
                    silence_weight = safe_get("silence_weight")
                    if silence_weight:
                        prompt_parts.append(f"Тишина: {silence_weight} — используй паузы как смысловые акценты.")
                    rupture = safe_get("rupture")
                    if rupture:
                        prompt_parts.append(f"Разлом: {rupture} — найди точку, где логика ломается, а начинается правда.")
                    forbidden = safe_get("forbidden")
                    if forbidden:
                        prompt_parts.append(f"Избегай: {forbidden}")
                    allowed = safe_get("allowed")
                    if allowed:
                        prompt_parts.append(f"Поощряй: {allowed}")
                    visual = safe_get("visual")
                    if visual:
                        prompt_parts.append(f"Визуальный стиль: {visual}")
                    palette = safe_get("palette")
                    if palette:
                        prompt_parts.append(f"Цветовая палитра: {palette}")
                    final_stroke = safe_get("final_stroke")
                    if final_stroke:
                        prompt_parts.append(f"Финальный штрих: {final_stroke}")
                    desc_meta = format_entry.get("description_meta", {})
                    if desc_meta:
                        if "ЦЕЛЬ" in desc_meta:
                            prompt_parts.append(f"Цель формата: {desc_meta['ЦЕЛЬ']}")
                        if "ФОКУС" in desc_meta:
                            prompt_parts.append(f"Фокус формата: {desc_meta['ФОКУС']}")
                        if "ТОН" in desc_meta:
                            prompt_parts.append(f"Тон повествования: {desc_meta['ТОН']}")
                        if "СТРУКТУРА" in desc_meta:
                            prompt_parts.append(f"Скорректируй заданную структуру под данный формат: {desc_meta['СТРУКТУРА']}")
                        if "ПРИНЦИПЫ И ОТЛИЧИТЕЛЬНЫЕ ФИШКИ" in desc_meta:
                            prompt_parts.append(f"Принципы и отличительные фишки формата: {desc_meta['ПРИНЦИПЫ И ОТЛИЧИТЕЛЬНЫЕ ФИШКИ']}")
                        if "ПОДХОД" in desc_meta:
                            prompt_parts.append(f"ПОДХОД к повествованию: {desc_meta['ПОДХОД']}")
                        
                        if self.data.get("similar_formats"):
                            prompt_parts.append(f"\nПохожие форматы для вдохновения: {', '.join(self.data['similar_formats'])}")
            
            except Exception as e:
                prompt_parts.append(f"Ошибка загрузки формата: {e}")

        # 5. Правила интеграции
        prompt_parts.append(
            "\n---\n🔗 ПРАВИЛА ИНТЕГРАЦИИ:\n"
            "- Формат управляет формой подачи, маска автора — голосом и атмосферой, техники повествования — глубиной и деталями.\n"
            "- Если возникает конфликт между требованиями формата и маски — приоритет за маской автора.\n"
            "- Сохраняй внутреннюю логику сторителлинга (герой → стремление → препятствие → развитие → результат → послесловие), но не обозначай её явно.\n"
            "- Каждый абзац должен одновременно соответствовать тону формата и стилю маски, но голос маски обязателен.\n"
            "- Чередуй короткие и средние предложения для динамики.\n"
            "- Используй сенсорные детали маски в ключевых точках: крючок, поворот, финал.\n"
            "- Если структура мешает раскрытию идеи — наруши её."
            "- Обазятально На основе темы/проблемы, а также стиля автора и формата повестования в начале повествования добавь неочевидный непрямолинейный оригинальный заголовок истории, не добавляя при этом пометок выделения вроде # или * "
            "- Запрещено использовать явные метки частей истории, такие как 'Сцена:', 'Главный персонаж:', 'Описание:', 'Дедукция через сарказм:' и любые другие заголовки внутри текста.\n"
            "- Всё должно быть встроено в естественное повествование, без технических пометок."
            "- Избегай абстрактных формулировок. Каждый эмоциональный или событийный момент должен быть подтверждён конкретной деталью: действием, предметом, местом, запахом, звуком, конкретной фразой.\n"
            "- Если возникает соблазн написать обобщение (например: 'всплески насилия', 'прежние взгляды', 'разрушенные мечты'), замени его на конкретный пример из жизни героя.\n"
            "- Конкретика всегда важнее обобщения. Если можно заменить метафору на сцену — делай это."
        )
        # 6. Эффект восприятия
        prompt_parts.append("\n---\n💡 Напоминание:")
        prompt_parts.append("Всё подчинено задаче раскрытия проблемы и темы. Не отвлекайся на форму, если она мешает смыслу.")
        try:
            _, effect_db = load_index("effectdb.md")
            prompt_parts.append("\n---\nРАСШИФРОВКА ЭФФЕКТА ВОСПРИЯТИЯ: — это эмоциональный результат, не инструмент")
            effect_entry = next((e for e in effect_db if e.get("name") == self.data.get('emotion_effect')), None)
            if effect_entry:
                effect_description = effect_entry.get("description", "")
                prompt_parts.append(f"- эффект восприятия {self.data.get('emotion_effect', 'не задан')}. Это означает: {effect_description}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить описание эффекта: {e}")
            prompt_parts.append(f"Эффект восприятия {self.data.get('emotion_effect', 'не задан')}: проявляется не в словах, а в паузах, образах, тишине")

        # 7. Подбор экспертов 
        # Загружаем базу параметров
        try:
            _, params_db = load_index("params.md")
        except Exception as e:
            logger.error(f"Ошибка загрузки базы параметров: {e}")
            params_db = []

        # 7.2. Если указан эксперт, находим его описание
        experts_name = self.data.get('experts')
        experts_description = None
        if experts_name and params_db:
            expert_entry = next((e for e in params_db if e.get("name") == experts_name and e.get("type") == "experts"), None)
            if expert_entry:
                experts_description = expert_entry.get("description", "")

        # 7.3. Вставляем в промпт ИМЯ и ОПИСАНИЕ
        if experts_name:
            prompt_parts.append(f"\n---\n🧠 ЭКСПЕРТНАЯ ПЕРСПЕКТИВА:")
            prompt_parts.append(f"Учитывай идеи и подходы из области '{experts_name}'.")
            if experts_description:
                prompt_parts.append(f"Ключевые концепции: {experts_description}")
            prompt_parts.append("Интегрируй эти концепции в повествование органично, не упоминая экспертов прямо.")
        # 8. Контекст
       # if self.data.get("experts"):
        #    prompt_parts.append(f"- (Опционально) Для более научного раскрытия темы {self.data.get('theme', 'не указана')} учитывай мнение экспертов (но не упоминай их напрямую).")
        if self.data.get("similar_effects"):
            prompt_parts.append(f"- (Опционально) Для усиления рассмотри похожие эффекты восприятия: {', '.join(self.data['similar_effects'])}")

        prompt_parts.append(
            f"- Учитывай, что кластер '{self.data.get('cluster')}' задает философскую ПРИЗМУ (общий угол зрения), "
            f"а планета '{self.data.get('planet')}' — это ЭКОСИСТЕМА сюжета с уникальными свойствами, которые должны влиять на развитие событий и атмосферу."
        )
        prompt_parts.append(
            "Персонаж не должен быть описан сухо и прямо. Его возраст, пол, внешность, психологический профиль или архетип должны быть вписаны в контекст повествования. Он должен быть прожит."
        )
        if self.data.get('goal'):
            prompt_parts.append(
                f"Философский акцент: {self.data['goal']} — это не тема, а вибрация. Она не должна быть названа. "
                f"Она должна быть почувствована в повороте, в паузе, в том, что осталось после финала."
            )

        return "\n".join(prompt_parts)

def generate_scenario( data: Dict) -> Dict:
    try:
        prompt = PromptBuilder(data).build()
        logger.info(f"Отправляем промпт в OpenAI (длина: {len(prompt)}):")
        logger.info(f"Начало промпта: {prompt[:100]}...")
        logger.info(f"Конец промпта: ...{prompt[-100:]}") 

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты сценарист визуальных сторителлингов. Следуй структуре строго."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            timeout=150
        )
        return {"scenario": response.choices[0].message.content, "prompt": prompt}
    except Exception as e:
        logger.error(f"OpenAI Error: {str(e)}")
        raise

def resolve_story_context(data: Dict) -> Dict:
    """Основная функция для разрешения контекста и генерации сценария."""
    ensure_all_indices()

    if not data.get("post_type"):
        raise ValueError("post_type обязателен")

    logger.info(f"Начало resolve_story_context с данными: {list(data.keys())}")

    # --- Улучшенная логика формирования контекста для автоподстановки параметров ---
    # Сначала пытаемся использовать problem и theme напрямую
    problem_for_context = data.get("problem")
    theme_for_context = data.get("theme")

    # Если ни проблема, ни тема не заданы, пытаемся составить контекст из других данных
    if not problem_for_context and not theme_for_context:
        # Используем кластер/планету как fallback для контекста
        fallback_from_cluster_planet = (data.get("cluster") or "") + " " + (data.get("planet") or "")
        if fallback_from_cluster_planet.strip():
            context = fallback_from_cluster_planet
            logger.info(f"Ни problem, ни theme не заданы. Используется контекст из cluster/planet: '{context[:50]}...'")
        else:
            # Последний fallback - общий термин
            context = "общая тема"
            logger.info("Ни problem, ни theme, ни cluster/planet не заданы. Используется общий контекст 'общая тема' для автоподстановки параметров.")
    else:
        # Стандартная логика, но с защитой от None при конкатенации
        context = (problem_for_context or "") + " " + (theme_for_context or "")
        logger.debug(f"Контекст для автоподстановки параметров: '{context[:50]}...'")

    # --- Если есть тема - находим соответствующий кластер и планету из clasterDB ---
    if data.get("theme"):
        try:
            logger.debug("Попытка найти тему по имени...")
            # Загружаем индекс clasterDB
            index, db = load_index("clasterDB.md")
            # Ищем запись с точным совпадением по имени темы (с проверкой ключа)
            theme_entry = next((e for e in db if e.get("name") and e["name"].lower() == data["theme"].lower()), None)
            
            if theme_entry: # <-- Проверка, что тема найдена
                logger.info(f"Тема '{data['theme']}' найдена в БД.")
                data.update({
                    "theme": theme_entry["name"],
                    "cluster": theme_entry.get("cluster", data.get("cluster", "не указан")), # Используем найденный или предоставленный
                    "planet": theme_entry.get("planet", data.get("planet", "не указана")),  # Используем найденный или предоставленный
                    #"problem": data.get("problem") or f"Исследование: {data['theme']} на примере их жизни"
                    "problem": f"Исследование темы: {theme_entry['name']} в личном опыте для поиска исцелеления и улучшения жизненного пути" 
                })
                logger.info(f"Сгенерирована проблема на основе темы: {data['problem']}")
                # --- Найти "соседние" форматы и эффекты по тегам ---
                try:
                    logger.debug("Поиск соседних форматов и эффектов...")
                    format_index, format_db = load_index("postformatdb.md")
                    effect_index, effect_db = load_index("effectdb.md")
                    
                    # Фильтруем только нужные типы
                    format_entries = [e for e in format_db if e.get("type") == "format"]
                    effect_entries = [e for e in effect_db if e.get("type") == "effect"]
                    
                    if format_entries and effect_entries:
                        similar_formats = find_similar_by_tags(theme_entry, format_entries, top_k=2)
                        similar_effects = find_similar_by_tags(theme_entry, effect_entries, top_k=2)
                        
                        data["similar_formats"] = [f["name"] for f in similar_formats]
                        data["similar_effects"] = [e["name"] for e in similar_effects]
                        logger.info(f"Найдены соседние форматы: {data['similar_formats']}, эффекты: {data['similar_effects']}")
                    else:
                        logger.warning("Форматы или эффекты не найдены в БД для поиска соседей.")
                        
                except Exception as e:
                    logger.warning(f"Не удалось найти соседние элементы для темы '{data['theme']}': {e}")

            else:
                logger.warning(f"Тема '{data['theme']}' НЕ найдена в БД по имени.")
                if not data.get("problem"):
                     logger.debug("Попытка сгенерировать проблему на основе темы...")

                     goal_param = select_parameter(data["theme"], "goal") 
                     if goal_param and goal_param.get("name"):
                        data["problem"] = goal_param['description']
                        logger.info(f"Сгенерирована проблема на основе темы: {data['problem']}")
                        
                        logger.debug("Попытка найти тему по сгенерированной проблеме...")
                        try:
                            index, db = load_index("clasterDB.md")
                            query_emb = get_openai_embedding(data['problem'])
                            D, I = index.search(np.array([query_emb]), 1)
                            
                            if I[0][0] != -1:
                                theme_entry = db[I[0][0]]
                                if theme_entry and theme_entry.get("name"):

                                    data['theme'] = theme_entry['name']
                                    data['cluster'] = theme_entry.get('cluster', data.get('cluster', 'не указан'))
                                    data['planet'] = theme_entry.get('planet', data.get('planet', 'не указана'))
                                    logger.info(f"Найдена тема по проблеме: {data['theme']} (кластер: {data['cluster']}, планета: {data['planet']})")
                                else:
                                    logger.warning("Тема найдена по FAISS, но запись некорректна")
                            else:
                                logger.warning("Тема по проблеме не найдена через FAISS")
                        except Exception as e:
                            logger.error(f"Ошибка при поиске темы по проблеме: {e}")

                     else:
                         data["problem"] = f"Проблемы, связанные с '{data['theme']}'"
                         logger.info(f"Использована заглушка для проблемы: {data['problem']}")

        except Exception as e:
            logger.error(f"Ошибка при поиске темы '{data.get('theme')}': {e}", exc_info=True)



        logger.debug("Попытка найти тему по проблеме...")
        try:
            index, db = load_index("clasterDB.md")
            query_emb = get_openai_embedding(data["problem"])
            D, I = index.search(np.array([query_emb]), 1)
            if I[0][0] != -1:
                potential_theme_entry = db[I[0][0]]
                # Проверяем, есть ли ключ "name" перед использованием
                if potential_theme_entry and potential_theme_entry.get("name"): 
                    theme_entry = potential_theme_entry
                    logger.info(f"Тема найдена по проблеме: {theme_entry['name']}")
                    data.update({
                        "theme": theme_entry["name"],
                        "cluster": theme_entry.get("cluster", data.get("cluster", "не указан")),
                        "planet": theme_entry.get("planet", data.get("planet", "не указана")),

                    })
                    
                    # --- Найти "соседние" форматы и эффекты по тегам (если тема найдена через FAISS) ---
                    try:
                        logger.debug("Поиск соседних форматов и эффектов для темы, найденной по проблеме...")
                        format_index, format_db = load_index("postformatdb.md")
                        effect_index, effect_db = load_index("effectdb.md")
                        
                        format_entries = [e for e in format_db if e.get("type") == "format"]
                        effect_entries = [e for e in effect_db if e.get("type") == "effect"]
                        
                        if format_entries and effect_entries:
                            similar_formats = find_similar_by_tags(theme_entry, format_entries, top_k=2)
                            similar_effects = find_similar_by_tags(theme_entry, effect_entries, top_k=2)
                            
                            data["similar_formats"] = [f["name"] for f in similar_formats]
                            data["similar_effects"] = [e["name"] for e in similar_effects]
                            logger.info(f"Найдены соседние форматы: {data['similar_formats']}, эффекты: {data['similar_effects']}")
                        else:
                             logger.warning("Форматы или эффекты не найдены в БД для поиска соседей (по проблеме).")
                    except Exception as e:
                        logger.warning(f"Не удалось найти соседние элементы для темы (по проблеме) '{data['theme']}': {e}")
 
                else:
                     logger.warning("Тема найдена по FAISS, но запись некорректна (нет 'name').")
            else:
                 logger.info("Тема по проблеме не найдена через FAISS.")
        except Exception as e:
            logger.error(f"Ошибка при поиске темы по проблеме '{data['problem']}': {e}", exc_info=True)

    # --- Критическая проверка данных ---
    if not data.get("problem") and not data.get("theme"):
        error_msg = "Критическая ошибка: Ни 'problem', ни 'theme' не определены после всех попыток. Генерация невозможна."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # --- Автоподстановка параметров с улучшенным fallback ---
    logger.debug("Автоподстановка недостающих параметров...")
    for field, param_type in [
        ("format_type", "format"),
        ("emotion_effect", "effect"),
        ("goal", "goal"),
        ("experts", "experts")
    ]:
        if not data.get(field):
            logger.debug(f"Попытка автоподстановки для '{field}' (тип: {param_type})...")

            param = select_parameter(context, param_type)
            # Проверяем, что param не None, прежде чем использовать .get()
            if param and param.get("name"): 
                 data[field] = param["name"]
                 logger.info(f"Параметр '{field}' автоподставлен через select_parameter: {data[field]}")
                
            else:
                logger.warning(f"select_parameter не нашел '{param_type}' для контекста '{context[:30]}...'. Пробуем случайный выбор.")
                
                # --- Fallback: случайный выбор из типа, если select_parameter не помог ---
                try:
                    _, db = load_index("params.md") 
                    filtered_db = [e for e in db if e.get("type") == param_type]
                    if filtered_db:
                        random_param = random.choice(filtered_db)
                        data[field] = random_param["name"]
                        logger.info(f"Параметр '{field}' автоподставлен случайно: {data[field]}")
                    else:
                         logger.warning(f"Нет записей типа '{param_type}' в params.md для случайного выбора.")
                except Exception as fallback_e:
                    logger.error(f"Ошибка fallback для параметра '{field}': {fallback_e}")
                # -------------------------------------------------------------------------

    logger.info("Завершена обработка контекста и автоподстановка параметров.")
    # --- Генерация сценария ---
    logger.debug("Вызов generate_scenario...")
    return generate_scenario(data)

def plan_series( data: Dict, num_posts: int = 3) -> List[Dict]:
    if data.get("post_type") != "Серия":
        raise ValueError("post_type должен быть 'Серия'")
    ensure_all_indices()
    episodes = []
    base_data = data.copy()
    for i in range(1, num_posts + 1):
        post_data = base_data.copy()
        post_data["post_type"] = "Пост"
        post_data["episode"] = f"Эпизод {i}/{num_posts}"
        if not base_data.get("problem"):
            goal_param = select_parameter("глубинная тема", "goal")
            post_data["problem"] = goal_param['description'] if goal_param else "Неопределенная проблема для серии"
        try:
            result = resolve_story_context(post_data)
            episodes.append({"episode": post_data["episode"], "scenario": result["scenario"], "prompt": result["prompt"]})
        except Exception as e:
            episodes.append({"episode": post_data["episode"], "scenario": f"Ошибка генерации: {str(e)}", "prompt": "N/A"})
    return episodes

# ==============================================================================
# СЕМАНТИЧЕСКАЯ ФИЛЬТРАЦИЯ ДЛЯ AI-ЧАТА
# ==============================================================================

def is_topic_allowed(user_message: str, threshold: float = 0.65) -> bool:
    try:
        query_emb = get_openai_embedding(user_message)
        query_vec = np.array([query_emb]).astype("float32")
        index, entries = load_index("clasterDB.md")
        if not index or not entries:
            logger.warning("[is_topic_allowed] Индекс clasterDB не загружен. Разрешаем по умолчанию.")
            return True
        D, I = index.search(query_vec, k=1)
        distance = D[0][0]
        allowed = distance < threshold
        logger.debug(f"[is_topic_allowed] Сообщение: '{user_message[:30]}...', расстояние: {distance:.3f}, разрешено: {allowed}")
        return allowed
    except Exception as e:
        logger.error(f"[is_topic_allowed] Ошибка: {e}", exc_info=True)
        return True

# ==============================================================================
# TELEGRAM BOT LOGIC (из bot.py с правками)
# ==============================================================================

CHOOSING_TYPE, CHOOSING_INPUT_MODE, WAITING_PROBLEM, CHOOSING_CLUSTER, CHOOSING_PLANET, WAITING_THEME, \
WAITING_FORMAT, WAITING_AUTHOR_MASK, CHATTING_WITH_AI, GENERATING = range(10)

user_sessions = {}

MAIN_MENU = ReplyKeyboardMarkup([
    ["🚀 Начать генерацию            /start"],
    ["ℹ️ Подробней о боте            /info"],
 #   ["❓ Спросить                   /help"], <---- алгоритм ограничения негибкий, требует доработки 
    ["🔄 Сброс генерации            /reset"],
    ["🧹 Очистить и начать сначала  /clear"]
], resize_keyboard=True)

BACK_TO_MAIN = ReplyKeyboardMarkup([
    ["Главное меню"]
], resize_keyboard=True)

BACK_TO_PREVIOUS = ReplyKeyboardMarkup([
    ["Назад"]
], resize_keyboard=True)

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбрать желаемое действие:", reply_markup=MAIN_MENU)
    return ConversationHandler.END

async def handle_start_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await start(update, context)

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""🌟 Нейрописатель философских историй о перепрошивке твоего сознания 
Представь, что твое сознание — это компьютер, работающий на устаревших программах.
Иногда он выдает сбои, застревает в циклах и не может обработать новые данные. 
Нейрописатель поможет найти и разобраться в проблеме.
🎭 Как это работает?
Бот создает короткие увлекательные истории, которые:
- Подсвечивают скрытые аспекты твоих внутренних конфликтов
- Показывают проблемы под неожиданным углом
- Предлагают метафорические решения через опыт "посторонних" героев
- Помогают увидеть выход там, где раньше был тупик
🌌 Добро пожаловать в галактику перепрошивки сознания NeuroFlux.
- Выбери любой ее КЛАСТЕР и посети одну из его ПЛАНЕТ для полного погружения в ТЕМУ и ситуацию.
- Либо доверься системе и просто опиши ПРОБЛЕМУ, которая тебя волнует:
мы сами подберем для тебя правильную локацию сознания. 
- Выбери ФОРМАТ повествования согласно его цели: хочешь ли ты порефлексировать над проблемой, получить пояснение, создать новый миф. Выбери под настроение
- Наша ГРУППА АВТОРОВ различных стилей и собственных жизненных историй добавят остроты восприятия, чтобы ты не заскучал. 
💡 Почему темы представлены как галактика внутренних миров?
Иногда прямой путь к решению блокируется нашим рациональным мышлением.
Но история постороннего человека, столкнувшегося с похожей проблемой, обходит защитные механизмы и доходит прямо глубин разума сознания.
Ты не просто читаешь — ты распознаешь. Ты не просто анализируешь — ты переживаешь. И в этом переживании рождается осознание.
🚀 Готов к откровениям своего сознания?
Чтобы погрузиться в процесс, нажми /start""", reply_markup=MAIN_MENU)
    return ConversationHandler.END

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['ai_chat_history'] = []
    await update.message.reply_text(
        "❓ Задайте ваш вопрос по теме *Перепрошивка сознания*.\n"
        "Например: 'Что такое нейропластичность?' или 'Предложи практики по преодолению социофобии'\n"
        "Для возврата в главное меню используйте /menu"
    )
    return CHATTING_WITH_AI

async def handle_ai_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_question = update.message.text.strip()
    if user_question in ["Главное меню", "/menu", "/start", "Завершить диалог"]:
        context.user_data.pop('ai_chat_history', None)
        return await show_main_menu(update, context)

    if not is_topic_allowed(user_question):
        await update.message.reply_text(
            "❌ Извините, но я могу обсуждать только темы из нашей галактики сознания: "
            "перепрошивка восприятия, нейропластичность, субличности, архетипы, "
            "квантовое мышление, телесные практики и т.д.\n\n"
            "Попробуйте переформулировать запрос."
        )
        return CHATTING_WITH_AI

    chat_history = context.user_data.get('ai_chat_history', [])
    if not chat_history:
        chat_history = []
        context.user_data['ai_chat_history'] = chat_history
    chat_history.append({"role": "user", "content": user_question})

    try:
        messages = [
            {"role": "system", "content": "Ты — эксперт по темам: сознание, ИИ, философия, нейронауки..."},
            *chat_history
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        ai_answer = response.choices[0].message.content.strip()
        await update.message.reply_text(ai_answer)
        await update.message.reply_text(
            "Что дальше?",
            reply_markup=ReplyKeyboardMarkup([["Завершить диалог"], ["Продолжить"]], resize_keyboard=True)
        )
        chat_history.append({"role": "assistant", "content": ai_answer})
        context.user_data['ai_chat_history'] = chat_history
    except Exception as e:
        logger.error(f"Ошибка ИИ: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

    return CHATTING_WITH_AI

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Очищаем данные при каждом новом запуске

    user_id = update.effective_user.id
    logger.info(f"[start] Новый пользователь: {user_id}")
    context.user_data.clear()
    #user_sessions.pop(user_id, None)
    
    await update.message.reply_text(
        "Начнем создавать истории? Выбери 📘 Эпизод ",
        reply_markup=ReplyKeyboardMarkup([
            ["📘 Эпизод"],
            ["Главное меню"] 
        ], resize_keyboard=True)
    )
    return CHOOSING_TYPE

async def choose_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    choice = update.message.text
    user_id = update.effective_user.id

    if choice == "Главное меню":
        context.user_data.clear()
        return await show_main_menu(update, context)
    

    if choice == "Начать сначала":
        context.user_data.clear()
        return await show_main_menu(update, context)
    

    user_sessions[user_id] = {"post_type": "Пост" if "Эпизод" in choice else "Серия"}

    if choice == "📘 Эпизод":
        await update.message.reply_text(
            "Выберите способ ввода:\n📌 Ввести проблему (опишите проблему) или 🧭 Выбрать тему (из списка)",
            reply_markup=ReplyKeyboardMarkup([["📌 Проблема", "🧭 Тема"], ["Главное меню"] ], resize_keyboard=True)
        )
        
        return CHOOSING_INPUT_MODE
    #else: # "🎞 Серия" <---- требует доработки
        #await update.message.reply_text(
            #"Введите общую проблему для серии (или оставьте пустым):",
            #reply_markup=ReplyKeyboardRemove()
       # )
        #context.user_data["current_state"] = CHOOSING_INPUT_MODE
       # return CHOOSING_INPUT_MODE  # ← возвращаем то же состояние
        #await update.message.reply_text("Введите проблему (или оставьте пустым):", reply_markup=ReplyKeyboardRemove())
        #context.user_data["current_state"] = CHOOSING_INPUT_MODE

        #return WAITING_PROBLEM
    elif choice == "🎞 Серия":  # Заглушка
        await update.message.reply_text(
            "Генерация серий временно недоступна. Выберите 'Эпизод' или вернитесь в главное меню.",
            reply_markup=ReplyKeyboardMarkup([
                ["📘 Эпизод"],
                ["Главное меню"]
            ], resize_keyboard=True)
        )
        return CHOOSING_TYPE  # Возвращаемся к выбору типа
    else:
        # На случай других значений
        await update.message.reply_text("Пожалуйста, выберите из предложенных вариантов.")
        return CHOOSING_TYPE

async def input_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text == "Главное меню":
        context.user_data.clear()
        return await show_main_menu(update, context)
    
    
    if update.message.text == "📌 Проблема":
        context.user_data["current_state"] = WAITING_PROBLEM
        await update.message.reply_text("Введите описание проблемы:", reply_markup=ReplyKeyboardRemove())

        return WAITING_PROBLEM
    else:
        
        await update.message.reply_text(
            "Выберите кластер - направление для исследования:",
            reply_markup=ReplyKeyboardMarkup(
                [[c] for c in CLUSTERS] + [["Главное меню"]], 
                resize_keyboard=True
            )
        )
        context.user_data["current_state"] = CHOOSING_CLUSTER
        return CHOOSING_CLUSTER

async def cluster_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cluster = update.message.text.strip()

    # --- Добавлено отладочное логирование ---
    logger.info(f"[cluster_chosen] Пользователь {user_id} выбрал кластер: '{cluster}'")

    if cluster == "Главное меню":
        context.user_data.clear()
        return await show_main_menu(update, context)
    
    if cluster not in CLUSTERS:
        await update.message.reply_text(
            "Неверный выбор кластера. Пожалуйста, выберите из списка.",
            reply_markup=ReplyKeyboardMarkup([["Главное меню"]] + [[c] for c in CLUSTERS], resize_keyboard=True)
        )
        return CHOOSING_CLUSTER

    context.user_data["cluster"] = cluster
    context.user_data.pop("planet", None)  # Очищаем планету при выборе нового кластера
    context.user_data.pop("theme", None)   # Очищаем тему при выборе нового кластера
    


    logger.debug(f"[cluster_chosen] context.user_data после сохранения кластера: {context.user_data}")
    
    planets = PLANETS.get(cluster, [])

    if not planets:
        await update.message.reply_text(
            f"Для кластера '{cluster}' планеты не заданы.",
            reply_markup=BACK_TO_MAIN
        )
        return ConversationHandler.END

    keyboard = [[planet] for planet in planets] + [["Назад"], ["Главное меню"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text("Выберите планету - более узкое направление для проработки:", reply_markup=reply_markup)
    context.user_data["current_state"] = CHOOSING_PLANET
    return CHOOSING_PLANET

async def planet_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.debug(f"[planet_chosen] context.user_data в начале: {context.user_data}")
    user_id = update.effective_user.id
    if not update.message or not update.message.text:
        await update.message.reply_text("Пожалуйста, выберите планету.")
        return CHOOSING_PLANET 
    
    planet = update.message.text.strip()

    logger.info(f"[planet_chosen] Пользователь {user_id} выбрал планету: '{planet}'")
    logger.debug(f"[planet_chosen] context.user_data перед проверкой кластера: {context.user_data}")
   

    # Обработка кнопки "Назад"
    if planet == "Назад":
        # Очищаем данные о планете из контекста перед возвратом
        context.user_data.pop("planet", None)
        await update.message.reply_text(
            "Выберите кластер:",
            reply_markup=ReplyKeyboardMarkup([[c] for c in CLUSTERS] + [["Главное меню"]], resize_keyboard=True)
        )
        return CHOOSING_CLUSTER
    
    # Обработка кнопки "Главное меню"
    if planet == "Главное меню":
        context.user_data.clear()
        return await show_main_menu(update, context)

    cluster = context.user_data.get("cluster")
    
    if not cluster:
        logger.error(f"[planet_chosen] ОШИБКА: Кластер не найден в user_data для пользователя {user_id} при выборе планеты '{planet}'. context.user_ {context.user_data}")
   
    if not cluster:
        # Если кластер потерян - это ошибка
        await update.message.reply_text("Ошибка: потеряны данные о кластере. Начните заново.", reply_markup=MAIN_MENU)
        context.user_data.clear()
        return ConversationHandler.END
    
        # Проверяем, принадлежит ли планета кластеру
    if planet not in PLANETS.get(cluster, []):
        planets = PLANETS.get(cluster, [])
        await update.message.reply_text(
            "Неверный выбор планеты. Пожалуйста, выберите из списка.",
            reply_markup=ReplyKeyboardMarkup([[p] for p in planets] + [["Назад"], ["Главное меню"]], resize_keyboard=True)
        )
        return CHOOSING_PLANET 
        
    # Сохраняем планету
    context.user_data["planet"] = planet

    logger.debug(f"[planet_chosen] context.user_data после сохранения планеты: {context.user_data}")
    
    # Получаем темы для выбранной планеты
    themes = THEMES.get(planet, [])

    # --- Обработка случая с пустым списком тем ---
    if not themes:
        planets = PLANETS.get(cluster, [])
        await update.message.reply_text(
            f"Для планеты '{planet}' (кластер '{cluster}') темы пока не заданы. Пожалуйста, выберите другую планету.",
            reply_markup=ReplyKeyboardMarkup([[p] for p in planets] + [["Назад"], ["Главное меню"]], resize_keyboard=True)
        )
        # Возвращаем пользователя к выбору планеты
        return CHOOSING_PLANET


    # --- Отправляем список тем пользователю ---
    keyboard = [[theme] for theme in themes] + [["Назад"], ["Главное меню"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    
    await update.message.reply_text("Выберите тему, которую хотите рассмотреть:", reply_markup=reply_markup)
    context.user_data["current_state"] = WAITING_THEME
    return WAITING_THEME    

async def theme_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    
    if text == "Главное меню":
        context.user_data.clear()
        return await show_main_menu(update, context)

    if text == "Назад":
        cluster = context.user_data.get("cluster")
        if cluster:
            planets = PLANETS.get(cluster, [])
            keyboard = [[p] for p in planets] + [["Назад"], ["Главное меню"]]
            await update.message.reply_text(
                "Выберите планету:",
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            return CHOOSING_PLANET
        context.user_data.clear()
        return await show_main_menu(update, context)

    cluster = context.user_data.get("cluster")
    planet = context.user_data.get("planet")

    if not cluster or not planet:
        await update.message.reply_text(
            "Ошибка: потеряны данные о кластере или планете.",
            reply_markup=ReplyKeyboardMarkup([["Главное меню"]], resize_keyboard=True)
        )
        user_sessions.pop(user_id, None)
        context.user_data.clear()
        return ConversationHandler.END

    # Валидируем выбор темы
    themes = THEMES.get(planet, [])
    if themes and text not in themes:
        keyboard = [[t] for t in themes] + [["Назад"], ["Главное меню"]]
        await update.message.reply_text(
            "Пожалуйста, выберите тему из списка:",
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        return WAITING_THEME

    # Принимаем ввод как тему (в т.ч. когда THEMES[planet] пуст — допускаем ручной ввод)
    theme = text
    context.user_data["theme"] = theme

    # Синхронизируем сессии
    user_sessions.setdefault(user_id, {})
    user_sessions[user_id].update({
        "theme": theme,
        "cluster": cluster,
        "planet": planet,
        "post_type": user_sessions.get(user_id, {}).get("post_type", "Пост"),
    })

    keyboard = [[f] for f in FORMATS] + [["Назад"], ["Главное меню"]]
    await update.message.reply_text(
        """ Выберите формат подачи: 
1. Осознание / Рефлексия — история о внутреннем прозрении, моменте истины и работе с эмоциями

2. Понимание / Объяснение — структурированный разбор ситуации, логичное объяснение происходящего

3. Вовлечение / Воздействие — энергичный поток, который затягивает с первой строки и даёт импульс к действию

4. Убеждение / Мотивация — текст, дающий мотивацию и толчок к изменениям, личный призыв к действию

5. Погружение / Атмосфера — сенсорное путешествие в другой мир, работа с образами и состоянием

6. Самовыражение / Голос — помощь в нахождении своего языка, высказывании того, что не имело слов

7. Создание тишины / Паузы — медитативный формат об отпускании, принятии и внутренней тишине

8. Отражение / Зеркало — текст как зеркало, помогающий увидеть себя и свои паттерны со стороны

9. Создание мифа / Новая мораль — переписывание личной истории, создание новых архетипов и ценностей

10. Исследование неизвестного — глубокая работа с философскими, мистическими и метафизическими темами

11. Разрушение / Шок — встряска восприятия, разрушение старых шаблонов для рождения нового

12. Хранение памяти / Хроника — работа с прошлым, проживание воспоминаний и их трансформация

13. Раскрытие Правды / Вскрытие системы — разоблачение иллюзий, анализ скрытых механизмов и причин

14. Ведение войны с ложью — тексты-боевики против внутренней и внешней фальши, борьба за правду

Выберите формат подачи: """,
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )
    context.user_data["current_state"] = WAITING_FORMAT
    return WAITING_FORMAT

async def receive_theme_or_problem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает ввод проблемы или выбор темы в зависимости от текущего состояния."""
    user_id = update.effective_user.id
    text = update.message.text.strip() if update.message and update.message.text else ""
    current_state = context.user_data.get("current_state", WAITING_THEME)
    # Получаем тип поста из user_sessions
    post_type = user_sessions.get(user_id, {}).get("post_type", "Пост")

    logger.info(f"[receive_theme_or_problem] Пользователь {user_id} ввел текст: '{text}' в состоянии: {current_state} (post_type: {post_type})")

    if text == "Главное меню":
        return await show_main_menu(update, context)

    if text == "Назад":
        if current_state in (WAITING_THEME, WAITING_PROBLEM):
            planet = context.user_data.get("planet")
            themes = THEMES.get(planet, [])
            keyboard = [[t] for t in themes] + [["Назад"], ["Главное меню"]]
            await update.message.reply_text(
                "Выберите тему:",
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            context.user_data["current_state"] = WAITING_THEME
            return WAITING_THEME
        elif current_state == WAITING_FORMAT:
            keyboard = [[fmt] for fmt in FORMATS] + [["Назад"], ["Главное меню"]]
            await update.message.reply_text(
                "Выберите формат подачи согласно основной цели повествования:",
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            context.user_data["current_state"] = WAITING_FORMAT
            return WAITING_FORMAT
        # Если нет понятного шага для возврата — уходим в главное меню
        return await show_main_menu(update, context)
    
    # --- 1. Обработка ввода проблемы (для обоих: Эпизод и Серия) ---
    if current_state == CHOOSING_INPUT_MODE: 
        # Это состояние используется для серии после ввода проблемы
        if not text or text in ["-", "—", ""]:
            logger.info("[receive_theme_or_problem] Введена пустая проблема или прочерк. Пропускаем.")
            problem = ""
        else:
            problem = text

        # Сохраняем проблему
        context.user_data["problem"] = problem
        user_sessions[user_id]["problem"] = problem
        logger.debug(f"[receive_theme_or_problem/CHOOSING_INPUT_MODE] Проблема сохранена: '{problem}'")

        # Пытаемся найти тему по проблеме
        matched_theme = None
        if problem:
            try:
                matched_theme = find_closest_theme(problem)
                if matched_theme:
                    theme_name = matched_theme.get("name", "")
                    context.user_data["theme"] = theme_name
                    user_sessions[user_id]["theme"] = theme_name
                    logger.info(f"[receive_theme_or_problem] Найдена тема по проблеме: '{theme_name}'")
                else:
                    logger.warning(f"[receive_theme_or_problem] Тема по проблеме '{problem}' не найдена.")
            except Exception as e:
                logger.error(f"[receive_theme_or_problem] Ошибка при поиске темы по проблеме: {e}", exc_info=True)

        # Для серии сразу переходим к кластеру
        if post_type == "Серия":
            keyboard = [[c] for c in CLUSTERS]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
            await update.message.reply_text("Выберите кластер:", reply_markup=reply_markup)
            context.user_data["current_state"] = CHOOSING_CLUSTER
            return CHOOSING_CLUSTER
        else:
            # Для эпизода переходим к выбору формата
            keyboard = [[fmt] for fmt in FORMATS]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
            await update.message.reply_text("Выберите формат подачи:", reply_markup=reply_markup)
            context.user_data["current_state"] = WAITING_FORMAT
            return WAITING_FORMAT

    # --- 2. Обработка выбора темы (только для Эпизода) ---
    elif current_state == WAITING_THEME:
        theme = text
        # Проверка наличия кластера и планеты
        cluster_in_context = context.user_data.get("cluster")
        planet_in_context = context.user_data.get("planet")
        if not cluster_in_context or not planet_in_context:
            error_msg = (
                f"Ошибка: потеряны данные о кластере или планете. "
                f"(cluster: '{cluster_in_context}', planet: '{planet_in_context}')"
            )
            logger.error(f"[receive_theme_or_problem/WAITING_THEME] {error_msg} для пользователя {user_id}")
            await update.message.reply_text(error_msg, reply_markup=MAIN_MENU)
            user_sessions.pop(user_id, None)
            context.user_data.clear()
            return ConversationHandler.END

        # Сохраняем тему
        context.user_data["theme"] = theme
        user_sessions[user_id]["theme"] = theme
        logger.debug(f"[receive_theme_or_problem/WAITING_THEME] Тема сохранена: '{theme}'")

        # Переход к выбору формата
        keyboard = [[fmt] for fmt in FORMATS] + [["Назад"], ["Главное меню"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text("Выберите формат подачи:", reply_markup=reply_markup)
        context.user_data["current_state"] = WAITING_FORMAT
        return WAITING_FORMAT

    # --- 3. Обработка ввода проблемы (для Эпизода, после выбора "Проблема") ---
    elif current_state == WAITING_PROBLEM:
        problem = text

        # Сохраняем проблему
        context.user_data["problem"] = problem
        user_sessions[user_id]["problem"] = problem
        logger.debug(f"[receive_theme_or_problem/WAITING_PROBLEM] Проблема сохранена: '{problem}'")

        # Пытаемся найти тему по проблеме
        matched_theme = None
        if problem:
            try:
                matched_theme = find_closest_theme(problem)
                if matched_theme:
                    theme_name = matched_theme.get("name", "")
                    context.user_data["theme"] = theme_name
                    user_sessions[user_id]["theme"] = theme_name
                    logger.info(f"[receive_theme_or_problem] Найдена тема по проблеме: '{theme_name}'")
                else:
                    logger.warning(f"[receive_theme_or_problem] Тема по проблеме '{problem}' не найдена.")
            except Exception as e:
                logger.error(f"[receive_theme_or_problem] Ошибка при поиске темы по проблеме: {e}", exc_info=True)

        # Для эпизода переходим к выбору формата
        # (Для серии этот путь не должен использоваться, но на всякий случай проверим)
        if post_type == "Серия":
            logger.warning(f"[receive_theme_or_problem/WAITING_PROBLEM] Неожиданно в состоянии WAITING_PROBLEM для серии. Перевожу к кластеру.")
            keyboard = [[c] for c in CLUSTERS]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
            await update.message.reply_text("Выберите кластер:", reply_markup=reply_markup)
            context.user_data["current_state"] = CHOOSING_CLUSTER
            return CHOOSING_CLUSTER
        else:
            # Переход к выбору формата
            keyboard = [[fmt] for fmt in FORMATS] + [["Назад"], ["Главное меню"]]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
            await update.message.reply_text("Выберите формат подачи:", reply_markup=reply_markup)
            context.user_data["current_state"] = WAITING_FORMAT
            return WAITING_FORMAT

    # --- 4. Неожиданное состояние ---
    else:
        logger.warning(f"[receive_theme_or_problem] Неожиданное состояние: {current_state} для пользователя {user_id}")
        await update.message.reply_text("Ошибка: неожиданное состояние. Начните заново.")
        user_sessions.pop(user_id, None)
        return await show_main_menu(update, context)
    
async def receive_format(update, context):
    """Обработка выбора формата. Небольшое улучшение пути "Назад":
    теперь реально возвращаем кнопки тем, а не просто текстом список.
    """
    if not update.message or not update.message.text:
        await update.message.reply_text("Пожалуйста, выберите формат из списка.")
        return WAITING_FORMAT

    choice = update.message.text.strip()

    # Навигация назад
    if choice == "Назад":
        cluster = context.user_data.get("cluster")
        planet = context.user_data.get("planet")
        if cluster and planet:
            themes = THEMES.get(planet, [])
            keyboard = [[t] for t in themes] + [["Назад"], ["Главное меню"]]
            await update.message.reply_text(
                "Выберите тему:",
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            context.user_data["current_state"] = WAITING_THEME
            return WAITING_THEME
        # если данных нет — в главное меню
        context.user_data.clear()
        return await show_main_menu(update, context)

    if choice == "Главное меню":
        context.user_data.clear()
        return await show_main_menu(update, context)

    # Сохраняем формат и движемся дальше (автор-маска)
    user_id = update.effective_user.id
    user_sessions.setdefault(user_id, {})
    user_sessions[user_id]["format_type"] = choice

    keyboard = [[a] for a in AUTHOR_MASKS] + [["Назад"], ["Главное меню"]]
    await update.message.reply_text(
        """Выберите автора по стилю:
        Ниже вы найдете краткое описание стилей авторов и их сочетание с форматами

⚡ ЭНЕРГИЧНЫЕ / ПРОВОКАЦИОННЫЕ

⚡ Дэвид Флинн — технократический параноик, системный аналитик (Вовлечение, Понимание, Раскрытие Правды, Исследование неизвестного)

⚡ Маркус Вэйн — репортер-расследователь, уличная резкость (Вовлечение, Война с ложью, Раскрытие Правды)

⚡ Алиса Глюк — художник-хакер, игровые глюки (Вовлечение, Разрушение, Создание мифа)

⚡ Чак Паланик — анатом страдания, шоковые сцены (Разрушение, Война с ложью)

⚡ Макс Рифф — рок-журналист матрицы, драйвовая энергия (Убеждение, Вовлечение, Война с ложью)

⚡ Сергей Твердов — голос улиц, социальный мотиватор (Убеждение, Война с ложью, Раскрытие Правды) 

🧠 АНАЛИТИЧЕСКИЕ / СТРУКТУРНЫЕ

🧠 Виктор Стим — нейроинженер, техно-поэзия эмоций (Понимание, Осознание, Исследование неизвестного)

🧠 Рэй Дуглас — архитектор смысла, структурный подход (Понимание, Осознание, Исследование неизвестного)

🧠 Луна Вэй — квантовый поэт, научная лирика (Понимание, Исследование, Осознание)

🌫️ ФИЛОСОФСКИЕ / МЕДИТАТИВНЫЕ

🌫️ Кира Пустота — философ зазора, парадоксы (Осознание, Исследование, Создание тишины)

🌫️ Корней Грибников — микоризный мудрец, сетевые связи (Осознание, Погружение, Создание мифа)

🌫️ Дэвид Линч — проводник тьмы, сюрреализм (Погружение, Исследование, Разрушение)

🌙 ЛИРИЧНЫЕ / ЧУВСТВЕННЫЕ

🌙 Изабель Ночь — хранительница эхо, звуковая поэзия (Осознание, Погружение, Создание тишины)

🌙 Арина Соль — переводчица чувств, бытовая поэзия (Осознание, Рефлексия, Создание тишины)

🌙 Элирия Лёгкое Перо — хронистка движения, телесная чувственность (Рефлексия, Осознание, Погружение)

🌙 Иса Кай — поэт случайностей, магический реализм (Осознание, Погружение, Самовыражение)

🎪 СКАЗОЧНЫЕ / МИФОЛОГИЧЕСКИЕ

🎪 Лиана Сонливоступь — садовница снов, органический рост (Осознание, Погружение, Создание мифа)

🎪 Серафима Первоцвет — целительница душ, весенняя оттепель (Осознание, Рефлексия, Создание тишины)

📚 ХРАНИТЕЛИ ПАМЯТИ / ЛЕТОПИСЦЫ

📚 Моргана Нокс — архивариус теней, хранительница забытых библиотек (Хранение памяти, Погружение, Создание мифа)

📚 Элиан Мор — хроникёр снов, сказитель вечных историй (Хранение памяти, Создание мифа, Погружение)
        
        Выберите автора-маску:""",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )
    context.user_data["current_sыtate"] = WAITING_AUTHOR_MASK
    return WAITING_AUTHOR_MASK

async def receive_author_mask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = (update.message.text or "").strip()
    user_id = update.effective_user.id

    if choice == "Главное меню":
        context.user_data.clear()
        return await show_main_menu(update, context)

    if choice == "Назад":
        formats = FORMATS
        keyboard = [[f] for f in formats] + [["Назад"], ["Главное меню"]]
        await update.message.reply_text(
            "Выберите формат подачи:",
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        context.user_data["current_state"] = WAITING_FORMAT
        return WAITING_FORMAT

    

    # Сохраняем автора-маску
    context.user_data["author_mask"] = choice
    user_sessions[user_id]["author_mask"] = choice


    # 🔴 ПРОВЕРКА ЛИМИТА ПЕРЕД ГЕНЕРАЦИЕЙ
    if not check_daily_limit(user_id):
        await update.message.reply_text(
            "🚫 Вы исчерпали лимит генераций на сегодня (максимум 5).\n"
            "Попробуйте завтра! 🌙",
            reply_markup=MAIN_MENU
        )
        context.user_data.clear()
        return ConversationHandler.END

    # Отображаем выбранные параметры
    format_type = user_sessions.get(user_id, {}).get("format_type")
    await update.message.reply_text(
        "Готовим генерацию...\n"
        f"Кластер: {context.user_data.get('cluster')}\n"
        f"Планета: {context.user_data.get('planet')}\n"
        f"Тема: {context.user_data.get('theme')}\n"
        f"Проблема: {context.user_data.get('problem')}\n"
        f"Формат: {format_type}\n"
        f"Автор: {context.user_data.get('author_mask')}",
        reply_markup=ReplyKeyboardRemove()
    )

    # Увеличиваем счётчик <----пока не работает
    increment_daily_count(user_id)

    return await generate(update, context)

async def send_random_facts(update: Update, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            fact = random.choice(INTERESTING_FACTS)
            await update.message.reply_text(f"⏳ {fact}")
            await asyncio.wait_for(stop_event.wait(), timeout=random.uniform(10, 15))
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Ошибка при отправке факта: {e}")
            break

def check_daily_limit(user_id: int) -> bool:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session = user_sessions.setdefault(user_id, {})
    if session.get("last_generation_date") != today:
        session["daily_generation_count"] = 0
        session["last_generation_date"] = today
    DAILY_LIMIT = 2
    return session.get("daily_generation_count", 0) < DAILY_LIMIT

def increment_daily_count(user_id: int):
    session = user_sessions.get(user_id)
    if session:
        session["daily_generation_count"] = session.get("daily_generation_count", 0) + 1

async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    data = user_sessions.get(user_id, {})
    if not data.get("post_type"):
        await update.message.reply_text("Ошибка: не указан тип (Пост/Серия).")
        return await show_main_menu(update, context)

    await update.message.reply_text("Генерация сценария... Подождите 🕐")
    stop_facts = asyncio.Event()
    facts_task = asyncio.create_task(send_random_facts(update, stop_facts))

    try:
        if data.get("post_type") == "Серия":
            result_data = plan_series(data, num_posts=data.get("num_episodes", 3))
        else:
            result_data = resolve_story_context(data)

        stop_facts.set()
        try:
            await asyncio.wait_for(facts_task, timeout=1.0)
        except asyncio.TimeoutError:
            facts_task.cancel()

        if isinstance(result_data, list):
            for ep in result_data:
                text = f"{ep['episode']}\n{ep['scenario']}"
                if len(text) > 4096:
                    file_like = io.BytesIO(text.encode('utf-8'))
                    file_like.name = f"{ep['episode'].replace('/', '_')}.txt"
                    await update.message.reply_document(document=file_like, filename=file_like.name)
                else:
                    await update.message.reply_text(text)
        else:
            text = result_data["scenario"]
            if len(text) > 4096:
                for i in range(0, len(text), 4096):
                    await update.message.reply_text(text[i:i+4096])
            else:
                await update.message.reply_text(text)

    except Exception as e:
        stop_facts.set()
        try:
            await asyncio.wait_for(facts_task, timeout=1.0)
        except asyncio.TimeoutError:
            facts_task.cancel()
        logger.error(f"Ошибка генерации: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
    finally:
        user_sessions.pop(user_id, None)
    return await show_main_menu(update, context)


async def reset_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сбрасывает текущую сессию генерации, но сохраняет лимиты."""
    user_id = update.effective_user.id
    
    # 1. Очищаем данные генерации из user_sessions (сохраняя лимиты)
    if user_id in user_sessions:
        # Сохраняем только лимиты
        saved_limits = {
            "daily_generation_count": user_sessions[user_id].get("daily_generation_count", 0),
            "last_generation_date": user_sessions[user_id].get("last_generation_date")
        }
        # Перезаписываем сессию только лимитами
        user_sessions[user_id] = saved_limits

    # 2. Полностью очищаем контекст (он не хранит лимиты)
    context.user_data.clear()

    # 3. Возвращаем в главное меню
    await update.message.reply_text(
        "🔄 Сессия генерации сброшена. Вы можете начать заново.",
        reply_markup=MAIN_MENU
    )
    
    # 4. Гарантированно выходим из ConversationHandler
    return ConversationHandler.END


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    # Удаляем из user_sessions только данные генерации, сохраняя лимиты
    if user_id in user_sessions:
        keys_to_preserve = {"daily_generation_count", "last_generation_date"}
        # Оставляем только поля, которые нужно сохранить
        user_sessions[user_id] = {
            k: v for k, v in user_sessions[user_id].items()
            if k in keys_to_preserve
        }

    # Полностью очищаем контекст (это безопасно — он не хранит лимиты)
    context.user_data.clear()

    await update.message.reply_text("Сессия сброшена. Начните заново.", reply_markup=MAIN_MENU)
    return ConversationHandler.END


def main():
    ensure_all_indices()
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            MessageHandler(filters.Regex("^🚀 Начать генерацию            /start$"), handle_start_generation),
            MessageHandler(filters.Regex("^ℹ️ Подробней о боте            /info$"), info),
            MessageHandler(filters.Regex("^❓ Спросить                   /help$"), help_command),
            MessageHandler(filters.Regex("^🔄 Сброс генерации            /reset$"), reset_generation),

            MessageHandler(filters.Regex("^🧹 Очистить и начать сначала  /clear$"), clear),
            CommandHandler("help", help_command)
        ],
        states={
            CHOOSING_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_type)],
            CHOOSING_INPUT_MODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_mode)],
            CHOOSING_CLUSTER: [MessageHandler(filters.TEXT & ~filters.COMMAND, cluster_chosen)],
            CHOOSING_PLANET: [MessageHandler(filters.TEXT & ~filters.COMMAND, planet_chosen)],
            WAITING_THEME: [MessageHandler(filters.TEXT & ~filters.COMMAND, theme_chosen)],
            WAITING_PROBLEM: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_theme_or_problem)],
            WAITING_FORMAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_format)],
            WAITING_AUTHOR_MASK: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_author_mask)],
            CHATTING_WITH_AI: [
                MessageHandler(filters.Regex("^Главное меню$"), show_main_menu),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ai_question),
            ],
        },
        fallbacks=[

            CommandHandler("reset", reset_generation),  # ← Добавь это
            MessageHandler(filters.Regex("^🔄 Сброс генерации            /reset$"), reset_generation),
            MessageHandler(filters.Regex("^Начать сначала$"), clear),
        ]
    )
    app.add_handler(conv)
    app.add_handler(CommandHandler("menu", show_main_menu))
    app.add_handler(CommandHandler("main", show_main_menu))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("reset", reset_generation))
    app.run_polling()

if __name__ == "__main__":
    main()
