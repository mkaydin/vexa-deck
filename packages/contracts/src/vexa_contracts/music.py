"""Backend-owned musical plans and small writer responses for generation v2."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationSettings(BaseModel):
    """Validated preferences for future batches; hints are not measured audio facts."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)
    track_count: int = Field(default=10, ge=1, le=10)
    duration_s: float | None = Field(default=None, ge=150, le=180)
    genre: str = Field(default="", max_length=100)
    vocal_mode: Literal["auto", "instrumental", "singing", "rap"] = "auto"
    language: Literal["auto", "en", "tr", "zh", "ja", "ko", "es", "fr", "de"] = "auto"
    lyric_topic: str = Field(default="", max_length=400)
    lyric_direction: str = Field(default="", max_length=400)
    custom_lyrics: str = Field(default="", max_length=5000)
    bpm: float | None = Field(default=None, ge=40, le=220)
    tempo_policy: Literal["hint", "strict"] = "hint"
    energy_arc: Literal["dj", "steady", "build", "wind_down"] = "dj"
    energy: float = Field(default=0.55, ge=0, le=1)
    texture: str = Field(default="", max_length=180)
    groove: str = Field(default="", max_length=180)
    include_instruments: list[str] = Field(default_factory=list, max_length=12)
    exclude_instruments: list[str] = Field(default_factory=list, max_length=12)
    seed: int = Field(default=42, ge=0, le=2147483000)
    render_speed: Literal["fast", "balanced", "quality"] = "balanced"
    start_policy: Literal["depot", "buffer", "complete"] = "depot"
    buffer_tracks: int = Field(default=2, ge=1, le=10)
    plan_attempts: int = Field(default=3, ge=1, le=5)
    render_attempts: int = Field(default=3, ge=1, le=5)
    planning_timeout_s: int = Field(default=600, ge=60, le=1800)
    rendering_timeout_s: int = Field(default=6000, ge=600, le=14400)

    @model_validator(mode="after")
    def coherent_preferences(self):
        for name in ("include_instruments", "exclude_instruments"):
            values = [text.strip() for text in getattr(self, name)]
            if any(not text or len(text) > 60 or "\n" in text for text in values):
                raise ValueError("instruments must be nonempty names of at most 60 characters")
            setattr(self, name, list(dict.fromkeys(values)))
        if {s.casefold() for s in self.include_instruments} & {
            s.casefold() for s in self.exclude_instruments
        }:
            raise ValueError("an instrument cannot be both included and excluded")
        if self.vocal_mode == "instrumental" and self.custom_lyrics:
            raise ValueError("instrumental generation cannot use custom lyrics")
        return self


class ThemeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    theme: str = Field(min_length=1, max_length=4000)
    genre: str = Field(min_length=1)
    palette: str
    vocal_mode: Literal["instrumental", "singing", "rap"]
    language: str
    bpm: float = Field(ge=40, le=220)
    bpm_explicit: bool = False
    preferences: GenerationSettings | None = None


class PlannedSection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    section_id: str
    name: str
    bars: int = Field(gt=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)


class TrackPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    pipeline_version: Literal[2] = 2
    slot: int = Field(ge=0, le=9)
    generator: Literal["yue2", "acestep", "minimax"]
    intent: ThemeSpec
    duration_s: float = Field(ge=150, le=180)
    energy: float = Field(ge=0, le=1)
    role: str
    sections: list[PlannedSection] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def continuous_sections(self):
        cursor = 0.0
        ids = set()
        for section in self.sections:
            if section.section_id in ids or abs(section.start_s - cursor) > 0.001:
                raise ValueError("backend section IDs must be unique and timing continuous")
            if section.end_s <= section.start_s:
                raise ValueError("backend sections must have positive durations")
            cursor = section.end_s
            ids.add(section.section_id)
        if abs(cursor - self.duration_s) > 0.001:
            raise ValueError("backend sections must cover the target duration")
        return self


class CreativeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=100)
    production_notes: str = Field(min_length=30, max_length=1800)
    instruments: list[str] = Field(min_length=1, max_length=12)
    section_descriptions: dict[str, str]


class LyricsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sections: dict[str, list[str]]
