import os
from dotenv import load_dotenv

class ApiKey:
    def __init__(self):
        load_dotenv()

    @property
    def openai_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")     
        if not key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return key  
