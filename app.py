from typing import Optional

import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ------------------ TOOL ------------------

def run_command(cmd: str) -> int:
    result = os.system(cmd)
    return result


def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url, timeout=10)
    response.encoding = "utf-8"

    if response.status_code == 200:
        # Replace degree symbol for reliable display in monospace logs.
        safe_text = response.text.replace("°", " deg ")
        return f"The weather in {city} is {safe_text}"

    return "Something went wrong"


available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
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


class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc")
    content: Optional[str] = Field(None, description="The optional string content for the step")
    tool: Optional[str] = Field(None, description="The ID of the tool to call.")
    input: Optional[str] = Field(None, description="The input params for the tool")


def append_log(message: str, log_placeholder: st.delta_generator.DeltaGenerator) -> None:
    st.session_state.logs.append(message)
    log_placeholder.code("\n".join(st.session_state.logs), language="text")


def run_agent(user_query: str, log_placeholder: st.delta_generator.DeltaGenerator) -> None:
    message_history = SYSTEM_PROMPT + f"\nUser: {user_query}\n"

    max_steps = 15
    steps = 0
    called_cities = set()

    while True:
        steps += 1
        if steps > max_steps:
            append_log("⚠️ Reached max steps, stopping.", log_placeholder)
            break

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
            append_log("⚠️ Empty response.", log_placeholder)
            break

        try:
            parsed_result = response.parsed
        except Exception:
            append_log(f"⚠️ Invalid JSON: {raw_result}", log_placeholder)
            break

        if parsed_result is None:
            append_log(f"⚠️ Could not parse structured output. Raw response: {raw_result}", log_placeholder)
            break

        # The SDK can return either a Pydantic model or a dict depending on version.
        if isinstance(parsed_result, dict):
            parsed_result = MyOutputFormat.model_validate(parsed_result)

        if parsed_result.step == "START":
            append_log(f"🔥 {parsed_result.content}", log_placeholder)

        elif parsed_result.step == "PLAN":
            append_log(f"🧠 {parsed_result.content}", log_placeholder)

        elif parsed_result.step == "TOOL":
            tool = parsed_result.tool
            city = parsed_result.input

            # prevent duplicate calls
            if city in called_cities:
                append_log(f"⚠️ Already fetched {city}, skipping.", log_placeholder)
                continue

            called_cities.add(city)

            append_log(f"🛠️ : {tool} ({city})", log_placeholder)
            result = available_tools[tool](city)
            append_log(f"🛠️ : {tool} ({city}) = {result}", log_placeholder)

            observation = {
                "step": "OBSERVE",
                "tool": tool,
                "content": result,
            }

            message_history += f"\n{json.dumps(observation)}\n"
            continue

        elif parsed_result.step == "OUTPUT":
            append_log(f"🤖 {parsed_result.content}", log_placeholder)
            break

        # append assistant step
        message_history += f"\n{parsed_result.model_dump_json()}\n"


def main() -> None:
    st.set_page_config(page_title="Weather Agent", page_icon="⛅")
    st.title("Weather Agent")
    st.write("See the agent's reasoning, tool calls, and output.")

    if "logs" not in st.session_state:
        st.session_state.logs = []

    user_query = st.text_input("Your request", placeholder="What is the weather in Ahmedabad and Indore?")
    col_run, col_clear = st.columns(2)

    with col_run:
        run_clicked = st.button("Run agent")

    with col_clear:
        clear_clicked = st.button("Clear logs")

    if clear_clicked:
        st.session_state.logs = []
        st.rerun()

    log_placeholder = st.empty()

    if st.session_state.logs:
        log_placeholder.code("\n".join(st.session_state.logs), language="text")

    if run_clicked:
        if not user_query.strip():
            st.warning("Please enter a request.")
        else:
            append_log(f"👉 {user_query}", log_placeholder)
            run_agent(user_query.strip(), log_placeholder)


if __name__ == "__main__":
    main()
