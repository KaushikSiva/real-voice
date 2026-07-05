from __future__ import annotations

import re


class SentenceChunker:
    def __init__(self, min_chars: int = 80) -> None:
        self.min_chars = min_chars
        self._buffer = ""

    def push(self, text: str) -> list[str]:
        self._buffer += text.replace("\n", " ")
        return self._drain(keep_tail=True)

    def flush(self) -> list[str]:
        return self._drain(keep_tail=False)

    def _drain(self, keep_tail: bool) -> list[str]:
        chunks: list[str] = []
        while True:
            match = re.search(r"([.!?])(\s+|$)", self._buffer)
            if not match:
                break

            end = match.end()
            candidate = self._buffer[:end].strip()
            remaining = self._buffer[end:].lstrip()

            if keep_tail and len(candidate) < self.min_chars:
                if not remaining:
                    break
                next_match = re.search(r"([.!?])(\s+|$)", remaining)
                if not next_match:
                    break
                end += next_match.end()
                candidate = self._buffer[:end].strip()
                remaining = self._buffer[end:].lstrip()

            chunks.append(_squash_spaces(candidate))
            self._buffer = remaining

        if not keep_tail and self._buffer.strip():
            chunks.append(_squash_spaces(self._buffer.strip()))
            self._buffer = ""

        return chunks


def _squash_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
