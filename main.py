from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
import os
import requests
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ------------------ TOOL ------------------

def run_command(cmd:str):
    result = os.system(cmd)
    return result




def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    response.encoding = "utf-8"

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"

    return "Something went wrong"


# ------------------ SYSTEM PROMPT ------------------
SYSTEM_PROMPT = """
You are a structured AI agent.

Execution flow:
START → PLAN → TOOL → OBSERVE → PLAN → OUTPUT

Rules:
- STRICT JSON only (no markdown, no ``` blocks)
- Only ONE step at a time
- Never skip steps
- After OUTPUT → STOP
- Never repeat same tool call with same input
- Never change user request
- Never introduce new cities

Capabilities:
- You ONLY have access to CURRENT weather
- You CANNOT predict future weather

If request cannot be fulfilled:
→ Go directly to OUTPUT explaining limitation

JSON format:
{
  "step": "START" | "PLAN" | "TOOL" | "OBSERVE" | "OUTPUT",
  "content": "string",
  "tool": "string",
  "input": "string"
}

Available tool:
- get_weather(city: str) : Takes a city name as a string and returns the current weather for that city.
- run_command(cmd: str) : Takes a system command as a string and executes the command on the user's system and returns the output from that command.


Example:

User: What is the weather in Ahmedabad and Indore?

{ "step": "START", "content": "User wants weather for Ahmedabad and Indore." }

{ "step": "PLAN", "content": "Fetch weather for Ahmedabad." }

{ "step": "TOOL", "tool": "get_weather", "input": "Ahmedabad" }

{ "step": "PLAN", "content": "Fetch weather for Indore." }

{ "step": "TOOL", "tool": "get_weather", "input": "Indore" }

{ "step": "PLAN", "content": "All data collected." }

{ "step": "OUTPUT", "content": "Ahmedabad: <weather>, Indore: <weather>" }
"""
