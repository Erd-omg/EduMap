import json
from typing import Any, AsyncIterator

import httpx

SYSTEM_PROMPT = """你是一个学习画像分析专家。根据用户的对话文本，分析以下六个维度的学习特征。

请以 JSON 格式输出，每个维度包含数值(0-1)和置信度(0-1)：

{
  "learning_ability": {
    "understanding_speed": 0.0-1.0,
    "problem_solving": 0.0-1.0,
    "critical_thinking": 0.0-1.0,
    "memory_retention": 0.0-1.0,
    "analytical_ability": 0.0-1.0
  },
  "learning_motivation": {
    "intrinsic_interest": 0.0-1.0,
    "goal_oriented": 0.0-1.0,
    "extrinsic_motivation": 0.0-1.0
  },
  "knowledge_coverage": {
    "mastered": ["已掌握的知识点"],
    "learning": ["正在学习的知识点"],
    "not_started": ["未开始的知识点"]
  },
  "interaction_style": {
    "visual": 0.0-1.0,
    "textual": 0.0-1.0,
    "interactive": 0.0-1.0,
    "auditory": 0.0-1.0
  },
  "focus_characteristics": {
    "avg_focus_duration_min": 0-120,
    "distraction_frequency": 0.0-1.0,
    "recommended_session_length": 0-120
  },
  "confidence_scores": {
    "learning_ability": 0.0-1.0,
    "learning_motivation": 0.0-1.0,
    "knowledge_coverage": 0.0-1.0,
    "interaction_style": 0.0-1.0,
    "focus_characteristics": 0.0-1.0
  }
}

只输出 JSON，不要其他内容。"""


class ProfileAnalyzer:
    def __init__(
        self, api_key: str, api_base: str = "", model: str = "spark"
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self.model = model

    def _build_messages(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            for msg in conversation_history[-10:]:  # keep last 10
                messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        return messages

    async def analyze(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Full analysis, returns profile + confidence."""
        messages = self._build_messages(user_message, conversation_history)
        text = await self._call_llm(messages)
        return self._parse_response(text)

    async def analyze_stream(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming analysis. Yields token events and final profile_update."""
        messages = self._build_messages(user_message, conversation_history)
        full_text = ""
        async for chunk in self._call_llm_stream(messages):
            full_text += chunk
            yield {"type": "token", "content": chunk}

        # Parse final response
        try:
            result = self._parse_response(full_text)
            yield {
                "type": "profile_update",
                "profile": result.get("profile", {}),
                "confidence_scores": result.get("confidence_scores", {}),
                "analysis_text": full_text,
            }
        except json.JSONDecodeError:
            yield {"type": "error", "content": "画像分析失败，请重试"}

        yield {"type": "complete", "status": "done"}

    async def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM API (OpenAI-compatible). Fallback to mock if no API key."""
        if not self.api_key:
            return json.dumps(self._mock_analysis(), ensure_ascii=False)

        async with httpx.AsyncClient(timeout=60.0) as client:
            url = (
                f"{self.api_base}/chat/completions"
                if self.api_base
                else "https://api.openai.com/v1/chat/completions"
            )
            resp = await client.post(
                url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2048,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _call_llm_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        """Stream from LLM. Fallback to mock if no API key."""
        if not self.api_key:
            result = json.dumps(self._mock_analysis(), ensure_ascii=False)
            for char in result:
                yield char
            return

        async with httpx.AsyncClient(timeout=120.0) as client:
            url = (
                f"{self.api_base}/chat/completions"
                if self.api_base
                else "https://api.openai.com/v1/chat/completions"
            )
            async with client.stream(
                "POST",
                url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Parse LLM response JSON, clamp values to 0-1, return structured result."""
        # Extract JSON from markdown code block if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)

        def clamp(v: Any) -> float:
            if isinstance(v, (int, float)):
                return max(0.0, min(1.0, float(v)))
            return 0.0

        def clamp_dict(d: dict[str, Any]) -> dict[str, float]:
            return {k: clamp(v) for k, v in d.items()}

        profile: dict[str, Any] = {}
        for dim in [
            "learning_ability",
            "learning_motivation",
            "interaction_style",
        ]:
            profile[dim] = clamp_dict(data.get(dim, {}))

        profile["knowledge_coverage"] = data.get(
            "knowledge_coverage",
            {"mastered": [], "learning": [], "not_started": []},
        )

        # focus_characteristics has non-0-1 fields
        fc = data.get("focus_characteristics", {})
        profile["focus_characteristics"] = {
            "avg_focus_duration_min": max(
                0, min(120, int(fc.get("avg_focus_duration_min", 0)))
            ),
            "distraction_frequency": clamp(
                fc.get("distraction_frequency", 0)
            ),
            "recommended_session_length": max(
                0, min(120, int(fc.get("recommended_session_length", 0)))
            ),
        }

        confidence = clamp_dict(data.get("confidence_scores", {}))

        return {"profile": profile, "confidence_scores": confidence}

    def _mock_analysis(self) -> dict[str, Any]:
        """Return mock analysis for development without API key."""
        return {
            "learning_ability": {
                "understanding_speed": 0.6,
                "problem_solving": 0.5,
                "critical_thinking": 0.4,
                "memory_retention": 0.5,
                "analytical_ability": 0.6,
            },
            "learning_motivation": {
                "intrinsic_interest": 0.7,
                "goal_oriented": 0.6,
                "extrinsic_motivation": 0.3,
            },
            "knowledge_coverage": {
                "mastered": [],
                "learning": ["编程基础"],
                "not_started": [],
            },
            "interaction_style": {
                "visual": 0.6,
                "textual": 0.5,
                "interactive": 0.4,
                "auditory": 0.3,
            },
            "focus_characteristics": {
                "avg_focus_duration_min": 30,
                "distraction_frequency": 0.3,
                "recommended_session_length": 45,
            },
            "confidence_scores": {
                "learning_ability": 0.6,
                "learning_motivation": 0.5,
                "knowledge_coverage": 0.3,
                "interaction_style": 0.4,
                "focus_characteristics": 0.5,
            },
        }
