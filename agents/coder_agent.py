from __future__ import annotations

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI

from .memory import MemoryManager
from .models import CoderResult
from tools.file_tools import read_file, write_file
from tools.exec_tools import exec_python

load_dotenv()


class CoderAgent:
    def __init__(self, window_size: int = 6):
        self.memory = MemoryManager(window_size=window_size)
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    def _llm_call(self, messages: List[Dict[str, str]]) -> str:
        """
        Call Azure OpenAI ChatCompletion. Before every call, retrieve semantic memory
        and inject as context.
        """
        # Build semantic context from the user messages
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break

        semantic_context_docs = self.memory.search_semantic_memory(last_user, k=3)
        semantic_context = "\n\n".join(semantic_context_docs)

        system_prefix = (
            "You are a coding assistant using a ReACT style. "
            "You have access to tools: read_file, write_file, exec_python. "
            "You must think step by step and output JSON instructions for tools or final answer.\n\n"
        )

        if semantic_context:
            system_prefix += f"Relevant past context:\n{semantic_context}\n\n"

        full_messages = [
            {"role": "system", "content": system_prefix},
        ] + messages

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=full_messages,
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _react_loop(self, user_task: str, max_iters: int = 10) -> Dict[str, Any]:
        """
        Minimal ReACT loop:
        - LLM returns a JSON-like directive:
          { "thought": "...", "action": "read_file" | "write_file" | "exec_python" | "finish",
            "action_input": {...}, "plan": "..." }
        - We execute the action, append observation, and repeat.
        """
        self.memory.add_message("user", user_task)

        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    "You are a ReACT coding agent. Given the user's task, you will iterate:\n"
                    "1) THINK: Explain what you will do next.\n"
                    "2) ACT: Choose one of: read_file, write_file, exec_python, or finish.\n"
                    "Respond ONLY in JSON with keys: thought, action, action_input, plan.\n"
                    "action_input MUST always be a JSON object (never a string), e.g.:\n"
                    "{ \"path\": \"main.py\" } or { \"code\": \"print('hi')\" }.\n"
                    "When action == 'finish', action_input must contain keys: code, explanation, plan, and result_summary.\n"
                    f"\nUser task:\n{user_task}"
                ),
            }
        ]

        # Add recent conversation context
        for m in self.memory.get_recent_messages():
            messages.append(m)

        iteration = 0
        last_plan = ""
        last_code = ""
        last_result_summary = ""
        last_explanation = ""

        while iteration < max_iters:
            iteration += 1

            # LLM step
            llm_output = self._llm_call(messages)
            print("\n--- LLM OUTPUT ---\n", llm_output, "\n------------------\n")
            self.memory.add_message("assistant", llm_output)

            # Try to parse JSON-ish output robustly
            import json

            tool_directive = None
            try:
                # Some models may wrap JSON in text; find first and last braces
                start = llm_output.find("{")
                end = llm_output.rfind("}")
                json_str = llm_output[start : end + 1]
                tool_directive = json.loads(json_str)
            except Exception as e:
                observation = f"Failed to parse JSON: {e}. Output was: {llm_output}"
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                self.memory.add_semantic_memory(observation, {"type": "parse_error"})
                continue

            thought = tool_directive.get("thought", "")
            action = tool_directive.get("action", "finish")
            raw_action_input = tool_directive.get("action_input", {})

            # Normalize action_input: ensure it's always a dict
            if isinstance(raw_action_input, dict):
                action_input = raw_action_input
            elif isinstance(raw_action_input, str):
                # Try to parse if it looks like JSON, otherwise wrap it
                try:
                    parsed = json.loads(raw_action_input)
                    action_input = parsed if isinstance(parsed, dict) else {"value": parsed}
                except Exception:
                    action_input = {"value": raw_action_input}
            else:
                action_input = {"value": raw_action_input}

            plan = tool_directive.get("plan", "")
            last_plan = plan or last_plan

            observation = ""

            if action == "read_file":
                path = action_input.get("path") or action_input.get("value") or ""
                try:
                    content = read_file(path)
                    observation = f"read_file success. Content length: {len(content)}. Preview:\n{content[:500]}"
                    self.memory.add_semantic_memory(
                        f"Read file {path} with content: {content[:1000]}",
                        {"type": "file_read", "path": path},
                    )
                except Exception as e:
                    observation = f"read_file error: {e}"

            elif action == "write_file":
                path = action_input.get("path") or ""
                content = action_input.get("content") or action_input.get("value") or ""
                try:
                    result = write_file(path, content)
                    observation = f"write_file success: {result}"
                    self.memory.add_semantic_memory(
                        f"Wrote file {path} with content: {content[:1000]}",
                        {"type": "file_write", "path": path},
                    )
                except Exception as e:
                    observation = f"write_file error: {e}"

            elif action == "exec_python":
                code = action_input.get("code") or action_input.get("value") or ""
                rc, stdout, stderr = exec_python(code)
                observation = (
                    f"exec_python exit_code={rc}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )
                last_code = code or last_code
                last_result_summary = observation or last_result_summary
                self.memory.add_semantic_memory(
                    f"Executed code with result: {observation[:1000]}",
                    {"type": "exec", "exit_code": rc},
                )

            elif action == "finish":
                final = action_input
                last_code = final.get("code", last_code)
                last_explanation = final.get("explanation", "")
                last_result_summary = final.get("result_summary", last_result_summary)
                last_plan = final.get("plan", last_plan)
                break

            else:
                observation = f"Unknown action: {action}. Please choose a valid tool or 'finish'."

            # Append observation and thought to messages
            messages.append(
                {
                    "role": "user",
                    "content": f"Thought: {thought}\nAction: {action}\nObservation: {observation}",
                }
            )
            self.memory.add_message("user", f"Observation: {observation}")

        return {
            "code": last_code,
            "explanation": last_explanation or "See plan and comments in code for explanation.",
            "plan": last_plan,
            "result": last_result_summary,
        }

    def solve(self, task: str) -> CoderResult:
        """
        Public entrypoint for Phase 1 Coder Agent.
        """
        outcome = self._react_loop(task)
        return CoderResult(**outcome)
