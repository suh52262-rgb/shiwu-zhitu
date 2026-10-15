"""文本分块（Phase 4）。

按段落/句子边界贪心合并，控制每块长度并带少量重叠，
保证语义完整性的同时适配 embedding 模型的输入长度上限。
"""

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；.!?\n])")


def chunk_text(text: str, max_chars: int = 400, overlap: int = 60) -> list[str]:
    """把整页文本切成若干块。

    策略：先按句号/换行切成句子，再贪心合并到 max_chars 以内；
    相邻块之间保留 overlap 字符重叠，避免关键句被切断后检索不到。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        # 单句超长时硬切（如连续公式段落）
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = sentence[:overlap]
            piece, sentence = sentence[:max_chars], sentence[max_chars - overlap :]
            chunks.append(piece)
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current}{sentence}"
        else:
            chunks.append(current)
            current = (current[-overlap:] if overlap else "") + sentence
    if current:
        chunks.append(current)
    return chunks
