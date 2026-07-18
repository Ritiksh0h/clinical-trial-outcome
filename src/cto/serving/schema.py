"""Pydantic models for the serving layer — the raw trial fields a user submits, and the
prediction response. Fields mirror the registration-time inputs the training pipeline reads;
all are optional with training-consistent defaults so a sparse form still yields a prediction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrialInput(BaseModel):
    """Raw, registration-time trial details. Encodings match the training pipeline
    (see cto.features.build); missing fields fall back to the same defaults training uses."""

    phase: int = Field(..., ge=1, le=3, description="Trial phase: 1, 2, or 3")
    eligibility_criteria: str = Field(
        "",
        description="Free-text eligibility criteria — drives TF-IDF + criteria-length/section counts",
    )
    enrollment: int | None = Field(None, ge=0, description="Planned enrollment (registration-time)")
    enrollment_type: str = Field(
        "ESTIMATED",
        description="ESTIMATED (planned, used) or ACTUAL (completion-time, excluded as leakage)",
    )
    number_of_arms: int | None = Field(None, ge=1)
    allocation: str | None = Field(None, description="RANDOMIZED / NON_RANDOMIZED")
    intervention_model: str | None = Field(
        None, description="PARALLEL / CROSSOVER / SINGLE_GROUP / …"
    )
    masking: str | None = Field(None, description="NONE / SINGLE / DOUBLE / TRIPLE / QUADRUPLE")
    primary_purpose: str | None = Field(None, description="TREATMENT / PREVENTION / DIAGNOSTIC / …")
    gender: str | None = Field(None, description="ALL / FEMALE / MALE")
    minimum_age: str | None = Field(None, description='e.g. "18 Years"')
    maximum_age: str | None = Field(None, description='e.g. "65 Years"')
    healthy_volunteers: bool | None = Field(None, description="Accepts healthy volunteers?")
    sponsor_class: str | None = Field(
        None,
        description="Lead sponsor agency class: INDUSTRY / NIH / FED / OTHER_GOV / NETWORK / OTHER",
    )
    has_industry_collaborator: bool = False
    has_nih_collaborator: bool = False
    intervention_types: list[str] = Field(
        default_factory=list, description='One per intervention, e.g. ["DRUG", "BIOLOGICAL"]'
    )
    registration_year: int | None = Field(
        None, ge=1990, le=2100, description="Year the trial was first registered"
    )


class PredictionResponse(BaseModel):
    """Option C: lead with the phase-relative completion TIER; keep the raw calibrated
    probability as a caveated secondary field."""

    phase: str = Field(..., description='Roman numeral: "I" / "II" / "III"')
    completion_tier: str = Field(
        ..., description="PRIMARY — Lower / Moderate / Higher completion likelihood"
    )
    tier_context: str = "relative to the evaluation set for this phase"
    raw_calibrated_probability: float = Field(..., ge=0.0, le=1.0, description="SECONDARY — technical")
    probability_caveat: str
    model_version: str
    phase_model: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[int]
