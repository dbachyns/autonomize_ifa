"""Vision reader backed by the OpenAI API (structured outputs)."""

from __future__ import annotations

import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from form_agent.config import settings
from form_agent.forms import get_plugin, identify
from form_agent.ingest import doc_id_for, load_pages
from form_agent.readers.base import Reader, register
from form_agent.readers.llm_common import (
    PROMPT_VERSION,
    UNIVERSAL_PROMPT,
    LLMUniversal,
    cost_usd,
    encode_image,
    to_universal,
    typed_prompt,
)
from form_agent.schema import FormRecord, RunInfo


@register("openai")
class OpenAIReader(Reader):
    def __init__(self, model: str | None = None, max_side: int = 2048):
        self.model = model or settings.openai_model
        self.max_side = max_side
        self.client = OpenAI()
        self._tokens = [0, 0]

    def read(self, image_path: Path) -> FormRecord:
        start = time.monotonic()
        self._tokens = [0, 0]
        pages = load_pages(image_path)
        content = []
        for i, page in enumerate(pages, 1):
            if len(pages) > 1:
                content.append({"type": "text", "text": f"Page {i}:"})
            content.append({"type": "image_url", "image_url": {"url": encode_image(page, self.max_side), "detail": "high"}})

        universal = to_universal(self._parse(content, UNIVERSAL_PROMPT, LLMUniversal), len(pages))
        form_type, confidence = identify(universal)

        typed = None
        if plugin := get_plugin(form_type):
            prompt = typed_prompt(plugin.title, plugin.extraction_hints())
            typed = self._parse(content, prompt, plugin.schema).model_dump()

        return FormRecord(
            doc_id=doc_id_for(image_path),
            source_file=_relative(image_path),
            run=RunInfo(
                reader=self.name,
                model=self.model,
                prompt_version=PROMPT_VERSION,
                seconds=round(time.monotonic() - start, 2),
                cost_usd=cost_usd(self.model, *self._tokens),
            ),
            form_type=form_type,
            form_type_confidence=round(confidence, 3),
            universal=universal,
            typed=typed,
        )

    def _parse(self, content: list[dict], prompt: str, schema: type[BaseModel]) -> BaseModel:
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}],
            response_format=schema,
        )
        if completion.usage:
            self._tokens[0] += completion.usage.prompt_tokens
            self._tokens[1] += completion.usage.completion_tokens
        message = completion.choices[0].message
        if message.parsed is None:
            raise RuntimeError(f"{self.model} returned no parsable output: {message.refusal or message.content!r}")
        return message.parsed


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(settings.data_dir.resolve()))
    except ValueError:
        return str(path)
