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


available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}

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


class MyOutputFormat(BaseModel) :
    step: str = Field(..., description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc")
    content: Optional[str] = Field(None, description="The optional string content for the step")
    tool: Optional[str] = Field(None, description="The ID of the tool to call.")
    input: Optional[str] = Field(None, description="The input params for the tool")


# ------------------ MAIN ------------------
def main():
    while True:
        user_query = input("👉 ")

        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        message_history = SYSTEM_PROMPT + f"\nUser: {user_query}\n"

        MAX_STEPS = 15
        steps = 0
        called_cities = set()

        while True:
            steps += 1

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=message_history,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MyOutputFormat,
                ),
            )

            raw_result = response.text

            if raw_result is None:
                print("⚠️ Empty response.")
                break


            try:
                parsed_result = response.parsed
            except Exception:
                print("⚠️ Invalid JSON:\n", raw_result)
                break

            if parsed_result is None:
                print("⚠️ Could not parse structured output. Raw response:\n", raw_result)
                break

            # The SDK can return either a Pydantic model or a dict depending on version.
            if isinstance(parsed_result, dict):
                parsed_result = MyOutputFormat.model_validate(parsed_result)

            # -------- HANDLE STEPS --------
            if parsed_result.step == "START":
                print("🔥", parsed_result.content)

            elif parsed_result.step == "PLAN":
                print("🧠", parsed_result.content)

            elif parsed_result.step == "TOOL":
                tool = parsed_result.tool
                city = parsed_result.input

                # prevent duplicate calls
                if city in called_cities:
                    print(f"⚠️ Already fetched {city}, skipping.")
                    continue

                called_cities.add(city)

                print(f"🛠️ : {tool} ({city})")

                result = available_tools[tool](city)

                print(f"🛠️ : {tool} ({city}) = {result}")

                observation = {
                    "step": "OBSERVE",
                    "tool": tool,
                    "content": result
                }

                message_history += f"\n{json.dumps(observation)}\n"
                continue

            elif parsed_result.step == "OUTPUT":
                print("🤖", parsed_result.content)
                break

            # append assistant step
            message_history += f"\n{parsed_result.model_dump_json()}\n"

        print("\n\n")


if __name__ == "__main__":
    main()