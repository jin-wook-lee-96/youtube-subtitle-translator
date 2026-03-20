"""
YouTube 영문 자막 추출 + 한국어 번역 → .md 다운로드
Streamlit UI 버전 | OpenAI gpt-4o-mini 사용
실행: streamlit run app.py
"""

import re
import os
import urllib.request
from datetime import datetime

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
from youtube_transcript_api import YouTubeTranscriptApi

OPENAI_MODELS = [
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


# ── 핵심 함수들 ──────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("YouTube video ID를 찾을 수 없습니다.")


def get_video_title(video_id: str) -> str:
    try:
        req = urllib.request.Request(
            f"https://www.youtube.com/watch?v={video_id}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8")
        m = re.search(r"<title>(.*?) - YouTube</title>", html)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return video_id


def get_transcript(video_id: str) -> tuple[str, list, str]:
    api = YouTubeTranscriptApi()
    tlist = api.list(video_id)
    try:
        t = tlist.find_manually_created_transcript(["en", "en-US", "en-GB"])
    except NoTranscriptFound:
        try:
            t = tlist.find_generated_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            keys = list(tlist._manually_created_transcripts.keys()) or \
                   list(tlist._generated_transcripts.keys())
            t = tlist.find_transcript(keys).translate("en")

    fetched = t.fetch()
    entries = [
        {
            "text": e.text if hasattr(e, "text") else e["text"],
            "start": e.start if hasattr(e, "start") else e["start"],
        }
        for e in fetched
    ]
    full_text = " ".join(e["text"] for e in entries)
    return full_text, entries, t.language


def format_seconds(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def entries_to_paragraphs(entries: list, chunk: int = 8) -> list[tuple[str, str]]:
    """chunk 개 문장씩 묶어 (타임스탬프, 본문) 리스트로 반환"""
    result = []
    for i in range(0, len(entries), chunk):
        group = entries[i: i + chunk]
        ts = format_seconds(group[0]["start"])
        body = " ".join(e["text"].replace("\n", " ") for e in group)
        result.append((ts, body))
    return result


def align_korean_to_timestamps(korean_text: str, eng_paragraphs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """한국어 텍스트를 영문 문단 수(N)에 맞게 균등 분할 후 타임스탬프 매핑"""
    n = len(eng_paragraphs)
    sentences = re.split(r'(?<=[.!?。])\s+', korean_text.strip())
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return [(ts, korean_text) for ts, _ in eng_paragraphs[:1]]

    # n개 그룹으로 균등 분배
    total = len(sentences)
    result = []
    for i in range(n):
        start = (i * total) // n
        end = ((i + 1) * total) // n
        chunk = " ".join(sentences[start:end]).strip()
        ts = eng_paragraphs[i][0]
        result.append((ts, chunk if chunk else ""))

    return result


def translate_to_korean(client: OpenAI, english_text: str, placeholder) -> str:
    """gpt-5.4-nano 스트리밍 번역 → placeholder 안에 실시간 출력"""
    korean = ""
    with client.chat.completions.create(
        model=OPENAI_MODEL,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional English-to-Korean translator. "
                    "Translate the given English transcript naturally and accurately into Korean. "
                    "Use clear paragraph breaks between topic changes. "
                    "Output ONLY the Korean translation, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Translate the following English transcript to Korean:\n\n{english_text}",
            },
        ],
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            korean += delta
            placeholder.markdown(
                f"<div style='color:#aaa;font-size:0.85rem;line-height:1.7'>{korean}▌</div>",
                unsafe_allow_html=True,
            )
    return korean


def render_paragraph_block(label: str, caption: str, paragraphs: list, timestamp: bool = False):
    st.markdown(f"### {label}")
    st.caption(caption)
    with st.container(height=520):
        for i, item in enumerate(paragraphs):
            if timestamp:
                ts, body = item
                st.markdown(
                    f"<span style='color:#888;font-size:0.72rem;font-family:monospace'>[{ts}]</span><br>"
                    f"<span style='font-size:0.95rem;line-height:1.8'>{body}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='font-size:0.95rem;line-height:1.9'>{item}</span>",
                    unsafe_allow_html=True,
                )
            if i < len(paragraphs) - 1:
                st.divider()


def build_markdown(title: str, url: str, english: str, korean: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# {title}

- **YouTube**: {url}
- **생성일**: {now}

---

## 영문 스크립트 (English Transcript)

{english}

---

## 한국어 번역 (Korean Translation)

{korean}
"""


# ── Streamlit UI ─────────────────────────────────────────────

st.set_page_config(page_title="YouTube 자막 번역기", page_icon="🎬", layout="wide")
st.title("🎬 YouTube 영문 자막 → 한국어 번역")

# API 키 — Streamlit Cloud secrets 우선, 없으면 .env 폴백
api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

# 사이드바: 상태 정보만 표시
with st.sidebar:
    st.header("⚙️ 설정")
    if api_key:
        st.success("API 키 로드됨 ✅")
    else:
        st.error("API 키 없음 ❌\n\n`.env` 파일에 `OPENAI_API_KEY`를 입력하세요.")
    st.markdown("---")
    OPENAI_MODEL = st.selectbox("모델 선택", OPENAI_MODELS, index=3)
    st.markdown("**자막**: YouTube Transcript API")

st.caption(f"영어 자막을 자동으로 가져와 {OPENAI_MODEL}로 한국어 번역 후 .md 파일로 다운로드합니다.")

# 메인 입력
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
run = st.button("▶ 번역 시작", type="primary", disabled=not (url and api_key))

if run:
    client = OpenAI(api_key=api_key)

    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("영상 정보 가져오는 중..."):
        title = get_video_title(video_id)
    st.subheader(f"📺 {title}")

    with st.spinner("영문 자막 추출 중..."):
        try:
            english_text, entries, lang = get_transcript(video_id)
        except TranscriptsDisabled:
            st.error("이 영상은 자막이 비활성화되어 있습니다.")
            st.stop()
        except NoTranscriptFound:
            st.error("영어 자막을 찾을 수 없습니다.")
            st.stop()
        except Exception as e:
            st.error(f"자막 오류: {e}")
            st.stop()

    eng_paragraphs = entries_to_paragraphs(entries, chunk=8)
    word_count = len(english_text.split())

    col1, col2 = st.columns(2)

    with col1:
        render_paragraph_block(
            "📄 영문 스크립트",
            f"언어: `{lang}` · 단어: {word_count:,} · 문단: {len(eng_paragraphs)}개",
            eng_paragraphs,
            timestamp=True,
        )

    with col2:
        st.markdown("### 🇰🇷 한국어 번역")
        status = st.caption(f"{OPENAI_MODEL} 번역 중... (스트리밍)")
        col2_placeholder = st.empty()

        korean_text = translate_to_korean(client, english_text, col2_placeholder)

        # 번역 완료 → placeholder 교체 (블럭 하나만)
        col2_placeholder.empty()
        kor_paragraphs = align_korean_to_timestamps(korean_text, eng_paragraphs)
        status.caption(f"문단: {len(kor_paragraphs)}개 · 영문과 동일 타임스탬프")
        with col2_placeholder.container(height=520):
            for i, (ts, body) in enumerate(kor_paragraphs):
                st.markdown(
                    f"<span style='color:#888;font-size:0.72rem;font-family:monospace'>[{ts}]</span><br>"
                    f"<span style='font-size:0.95rem;line-height:1.8'>{body}</span>",
                    unsafe_allow_html=True,
                )
                if i < len(kor_paragraphs) - 1:
                    st.divider()

    # .md 다운로드
    st.markdown("---")
    md_content = build_markdown(title, url, english_text, korean_text)
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:50].strip()
    filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    st.download_button(
        label="⬇️ .md 파일 다운로드",
        data=md_content.encode("utf-8"),
        file_name=filename,
        mime="text/markdown",
        type="primary",
    )
    st.success(f"완료! `{filename}` 다운로드 준비됨.")
