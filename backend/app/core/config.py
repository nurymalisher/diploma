from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Steam Game Recommender"
    STEAM_API_KEY: str
    SESSION_SECRET: str
    BASE_URL: str
    STEAM_OPENID_URL: str = "https://steamcommunity.com/openid/login"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()