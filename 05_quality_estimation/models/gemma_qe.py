from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class DimScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # "accuracy_completeness": 0-10,
    # "terminology_consistency": 0-10,
    # "fluency_coherence": 0-10,
    # "style_tone_audience": 0-10,
    # "locale_formatting": 0-10,
    # "technical_integrity": 0-10,
    # "cultural_appropriateness": 0-10
    accuracy_completeness: StrictInt = Field(ge=0, le=10)
    terminology_consistency: StrictInt = Field(ge=0, le=10)
    fluency_coherence: StrictInt = Field(ge=0, le=10)
    style_tone_audience: StrictInt = Field(ge=0, le=10)
    locale_formatting: StrictInt = Field(ge=0, le=10)
    technical_integrity: StrictInt = Field(ge=0, le=10)
    cultural_appropriateness: StrictInt = Field(ge=0, le=10)


class QEResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dims_0to10: DimScores
    overall_0to100: StrictInt = Field(ge=0, le=100)
