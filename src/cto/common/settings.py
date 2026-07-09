from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Empty-string defaults so imports succeed without a .env file.
    # aact_client validates non-empty before connecting.
    aact_user: str = ""
    aact_password: str = ""
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    hf_token: str = ""


settings = Settings()
