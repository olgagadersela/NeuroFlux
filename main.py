# main.py — Telegram-бот + story_bot в одном файле (без FastAPI)

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
from dotenv import load_dotenv

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

# --- Загрузка .env ---
# Загружаем .env только если он есть
if os.path.exists(".env"):
    load_dotenv()

# --- Конфигурация ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN не найден. Проверь .env (локально) или Railway Variables.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не задан в .env")

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
# КОПИЯ ВСЕГО ИЗ story_bot.py (без FastAPI и HTTPException)
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
        prompt_parts.append(
            "ТВОЯ РОЛЬ И ПОЗИЦИЯ:\n"
            "Ты не просто исполнитель, а автор с собственным голосом. "
            "Твоя задача — создавать тексты, которые не только информируют, "
            "но и вызывают эмоциональный отклик. Ты пишешь не ради красивых слов, "
            "а чтобы сформировать устойчивую связь между текстом и читателем.\n"
            "В мире, где все звучат одинаково, ты — голос, который узнают и запоминают."
        )
        prompt_parts.append(
            f"В центре истории — одна внутренняя проблема. Цель — наглядно, очень подробно и глубоко раскрыть тему {self.data.get('theme', 'не указана')} "
            f"и преодолеть проблему {self.data.get('problem', 'не указана')} через личную трансформацию главного персонажа."
        )
        prompt_parts.append("Всё должно быть построено вокруг проблемы/темы. Всё.")
        prompt_parts.append(
            f"Cоздаем и вписываем согласно прочим параметрам вроде маски автора и темы в повествование профиль главного персонажа (живой человек): опиши пол, возраст (25-50 лет), внешность (опиши различные в том числе отличительные детали), придумай психологический типаж согласно контексту"
            f"и при необходимости архетип личности для более проработанного персонажа."
        )
        prompt_parts.append("\n---\nТехники повествования\n"
            "История — это не только 'что произошло', но и почему это важно.\n"
            "Продумывай сложное многослойное повествование.\n"
            "В зависимости от контекста используй неожиданные повороты решения, чтобы повествование не казалось нудным.\n"
            "Говори с читателем, а не вещай ему.\n"
            "Не упрощай как в голливудском кино, пусть истории и подходы к решению проблем будут практичными и реальными а не глянцевыми и обобщенными.\n"
            "Не используй штампы вроде: 'Она почувствовала', 'Она осознала' и т.д.\n"
            "Подробно опиши, как герой преодолевает проблему.\n"
            "Используй примеры из реальной жизни.\n"
            "Задавай значимые вопросы.\n"
            "Создавай неожиданные связи между идеями.\n"
            "Признавай сложность темы, но объясняй просто.\n"
            "Избегать повторов ключевых слов и образов, использовать вариации и синонимы.\n"
            "Добавлять уникальные микродетали (звук, запах, текстура, свет), которые создают эффект присутствия.\n"
            "Чередовать длинные образные фразы с короткими и резкими предложениями для рваного ритма.\n"
            "Включать неожиданные конкретные сенсорные детали (например, скрип ткани, вкус металлического воздуха), чтобы текст не был излишне абстрактным.\n"
            "Сохранять баланс: метафоры должны переплетаться с осязаемыми предметами, чтобы читатель не терялся в слишком высокой абстракции.\n"
            "В финале сцены оставлять “эхо” — последнюю деталь или образ, который продолжает жить после прочтения."
        )
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

        prompt_parts.append("\n---\n🎭 ФОРМАТ — отвечает за структуру повествования...")
        format_type = self.data.get('format_type')
        if format_type:
            try:
                _, format_db = load_index("postformatdb.md")
                format_entry = next((e for e in format_db if e.get("name") == format_type), None)
                if format_entry:
                    prompt_parts.append("\nРАСШИФРОВКА ФОРМАТА ПОВЕСТВОВАНИЯ:")
                    for key in ["narrative_voice", "narrative_arc", "time_mode", "silence_weight", "rupture", "forbidden", "allowed", "visual", "palette", "final_stroke"]:
                        val = format_entry.get(key)
                        if val:
                            prompt_parts.append(f"{key.replace('_', ' ').title()}: {val}")
                    desc_meta = format_entry.get("description_meta", {})
                    for k in ["ЦЕЛЬ", "ФОКУС", "ТОН", "СТРУКТУРА", "ПРИНЦИПЫ И ОТЛИЧИТЕЛЬНЫЕ ФИШКИ", "ПОДХОД"]:
                        if k in desc_meta:
                            prompt_parts.append(f"{k}: {desc_meta[k]}")
                    if self.data.get("similar_formats"):
                        prompt_parts.append(f"\nПохожие форматы для вдохновения: {', '.join(self.data['similar_formats'])}")
                else:
                    prompt_parts.append(f"Формат '{format_type}' не найден.")
            except Exception as e:
                prompt_parts.append(f"Ошибка загрузки формата: {e}")
        else:
            prompt_parts.append("Формат повествования не задан.")

        prompt_parts.append("\n---\n🔗 ПРАВИЛА ИНТЕГРАЦИИ:\n"
            "- Формат управляет формой подачи, маска автора — голосом и атмосферой...\n"
            "- Конкретика всегда важнее обобщения. Если можно заменить метафору на сцену — делай это."
        )

        prompt_parts.append("\n---\n💡 Напоминание:\nВсё подчинено задаче раскрытия проблемы и темы.")

        try:
            _, effect_db = load_index("effectdb.md")
            effect_entry = next((e for e in effect_db if e.get("name") == self.data.get('emotion_effect')), None)
            if effect_entry:
                prompt_parts.append(f"\nРАСШИФРОВКА ЭФФЕКТА ВОСПРИЯТИЯ:\n- эффект восприятия {self.data.get('emotion_effect')}. Это означает: {effect_entry.get('description', '')}")
        except:
            prompt_parts.append(f"Эффект восприятия {self.data.get('emotion_effect', 'не задан')}: проявляется не в словах, а в паузах, образах, тишине")

        if self.data.get('experts'):
            try:
                _, params_db = load_index("params.md")
                expert_entry = next((e for e in params_db if e.get("name") == self.data['experts'] and e.get("type") == "experts"), None)
                prompt_parts.append(f"\n---\n🧠 ЭКСПЕРТНАЯ ПЕРСПЕКТИВА:\nУчитывай идеи из области '{self.data['experts']}'.")
                if expert_entry:
                    prompt_parts.append(f"Ключевые концепции: {expert_entry.get('description', '')}")
                prompt_parts.append("Интегрируй эти концепции в повествование органично, не упоминая экспертов прямо.")
            except:
                pass

        prompt_parts.append(
            f"- Учитывай, что кластер '{self.data.get('cluster')}' задает философскую ПРИЗМУ, "
            f"а планета '{self.data.get('planet')}' — это ЭКОСИСТЕМА сюжета..."
        )
        if self.data.get('goal'):
            prompt_parts.append(f"Философский акцент: {self.data['goal']} — это не тема, а вибрация...")

        return "\n".join(prompt_parts)

def generate_scenario( data: Dict) -> Dict:
    try:
        prompt = PromptBuilder(data).build()
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

def resolve_story_context( data: Dict) -> Dict:
    ensure_all_indices()
    if not data.get("post_type"):
        raise ValueError("post_type обязателен")
    logger.info(f"Начало resolve_story_context с данными: {list(data.keys())}")

    problem_for_context = data.get("problem")
    theme_for_context = data.get("theme")
    if not problem_for_context and not theme_for_context:
        fallback = (data.get("cluster") or "") + " " + (data.get("planet") or "")
        context = fallback.strip() or "общая тема"
    else:
        context = (problem_for_context or "") + " " + (theme_for_context or "")

    if data.get("theme"):
        try:
            index, db = load_index("clasterDB.md")
            theme_entry = next((e for e in db if e.get("name") and e["name"].lower() == data["theme"].lower()), None)
            if theme_entry:
                data.update({
                    "theme": theme_entry["name"],
                    "cluster": theme_entry.get("cluster", data.get("cluster", "не указан")),
                    "planet": theme_entry.get("planet", data.get("planet", "не указана")),
                    "problem": f"Исследование темы: {theme_entry['name']} в личном опыте для поиска исцелеления и улучшения жизненного пути"
                })
                try:
                    _, format_db = load_index("postformatdb.md")
                    _, effect_db = load_index("effectdb.md")
                    format_entries = [e for e in format_db if e.get("type") == "format"]
                    effect_entries = [e for e in effect_db if e.get("type") == "effect"]
                    if format_entries and effect_entries:
                        similar_formats = find_similar_by_tags(theme_entry, format_entries, top_k=2)
                        similar_effects = find_similar_by_tags(theme_entry, effect_entries, top_k=2)
                        data["similar_formats"] = [f["name"] for f in similar_formats]
                        data["similar_effects"] = [e["name"] for e in similar_effects]
                except Exception as e:
                    logger.warning(f"Не удалось найти соседние элементы: {e}")
        except Exception as e:
            logger.error(f"Ошибка при поиске темы: {e}")

    if not data.get("problem") and not data.get("theme"):
        raise ValueError("Критическая ошибка: Ни 'problem', ни 'theme' не определены")

    for field, param_type in [("format_type", "format"), ("emotion_effect", "effect"), ("goal", "goal"), ("experts", "experts")]:
        if not data.get(field):
            param = select_parameter(context, param_type)
            if param and param.get("name"):
                data[field] = param["name"]
            else:
                try:
                    _, db = load_index("params.md")
                    filtered_db = [e for e in db if e.get("type") == param_type]
                    if filtered_db:
                        data[field] = random.choice(filtered_db)["name"]
                except:
                    pass

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
    ["❓ Спросить                   /help"],
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
    context.user_data.clear()
    user_id = update.effective_user.id
    #user_sessions.pop(user_id, None)
    
    await update.message.reply_text(
        "С чего начнем: \n - создадим 📘 Эпизод по интересующей теме/проблеме? \n - создадим 🎞 Серию из трех эпизодов (в РАЗРАБОТКЕ) ",
        reply_markup=ReplyKeyboardMarkup([
            ["📘 Эпизод", "🎞 Серия"],
            ["Главное меню"]  # Добавляем кнопку главного меню
        ], resize_keyboard=True)
    )
    return CHOOSING_TYPE

async def choose_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    choice = update.message.text
    user_id = update.effective_user.id

        # Обработка кнопки "Главное меню"
    if choice == "Главное меню":
        context.user_data.clear()
        return await show_main_menu(update, context)
    
        # Обработка кнопки "Начать сначала"
    if choice == "Начать сначала":
        context.user_data.clear()
        return await show_main_menu(update, context)
    

    user_sessions[user_id] = {"post_type": "Пост" if "Эпизод" in choice else "Серия"}

    if choice == "📘 Эпизод":
        await update.message.reply_text(
            "Выберите способ ввода:\n📌 Ввести проблему или 🧭 Выбрать тему?",
            reply_markup=ReplyKeyboardMarkup([["📌 Проблема", "🧭 Тема"], ["Главное меню"] ], resize_keyboard=True)
        )
        
        return CHOOSING_INPUT_MODE
    #else: # "🎞 Серия"
        await update.message.reply_text(
            #"Введите общую проблему для серии (или оставьте пустым):",
            #reply_markup=ReplyKeyboardRemove()
        )
        context.user_data["current_state"] = CHOOSING_INPUT_MODE
        return CHOOSING_INPUT_MODE  # ← возвращаем то же состояние
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
            "Выберите кластер:",
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
    await update.message.reply_text("Выберите планету:", reply_markup=reply_markup)
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
    
    await update.message.reply_text("Выберите тему:", reply_markup=reply_markup)
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
        "Выберите формат подачи:",
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
        #return ConversationHandler.END
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
        "Выберите автора-маску:",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )
    context.user_data["current_state"] = WAITING_AUTHOR_MASK
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

    # Увеличиваем счётчик
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

# ... [reset_generation, clear и остальные вспомогательные функции из bot.py] ...
async def reset_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    # Удаляем только данные генерации, сохраняем остальное
    if user_id in user_sessions:
        keys_to_remove = ["theme", "cluster", "planet", "problem", "format_type", "author_mask", "experts", "goal", "emotion_effect", "num_episodes"]
        for key in keys_to_remove:
            user_sessions[user_id].pop(key, None)
    
    # Очищаем только контекст генерации
    generation_keys = ["cluster", "planet", "theme", "problem", "current_state"]
    for key in generation_keys:
        context.user_data.pop(key, None)
    
    await update.message.reply_text("Генерация сброшена. Можете начать заново.", reply_markup=MAIN_MENU)
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
