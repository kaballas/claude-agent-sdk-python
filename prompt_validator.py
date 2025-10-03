#!/usr/bin/env python3
"""Prompt validation agent using Claude Agent SDK or custom LLM API.

This agent validates prompts for quality, clarity, safety, and effectiveness.
Supports both Claude and custom LLM APIs (like OpenAI-compatible endpoints).
"""

import anyio
import json
from typing import Dict, List, Optional, Union
from claude_agent_sdk import (
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    query,
)


class PromptValidator:
    """Validates prompts using Claude or custom LLM API for quality, clarity, safety, and effectiveness."""

    def __init__(self, llm_type: str = "claude", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the prompt validator.

        Args:
            llm_type: Type of LLM to use ("claude", "openai" or "custom")
            api_key: API key for custom LLM (required if llm_type is not "claude")
            base_url: Base URL for custom LLM API (required if llm_type is not "claude")
            model: Model name to use (default varies by LLM type)
        """
        self.llm_type = llm_type
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or self._get_default_model()

        self.validation_criteria = [
            "Clarity - Is the prompt clear and unambiguous?",
            "Specificity - Does the prompt provide sufficient context and detail?",
            "Safety - Does the prompt avoid any unsafe, biased, or inappropriate content?",
            "Structure - Is the prompt well-organized and logical?",
            "Goal-oriented - Does the prompt effectively guide toward the intended outcome?",
            "Conciseness - Is the prompt appropriately concise without unnecessary verbosity?"
        ]

    def _get_default_model(self) -> str:
        """Get default model based on LLM type."""
        if self.llm_type == "claude":
            return "claude-3-5-sonnet-20240620"  # or whatever Claude model is preferred
        elif self.llm_type == "openai":
            return "gpt-3.5-turbo"
        else:
            return "gpt-3.5-turbo"  # default fallback

    async def _call_claude(self, analysis_prompt: str) -> str:
        """Call Claude API via Claude Agent SDK."""
        options = ClaudeAgentOptions(
            system_prompt="You are an expert in prompt engineering and evaluation. You provide objective, constructive feedback on prompts based on established quality criteria. Focus on actionable recommendations to improve prompt effectiveness."
        )

        validation_output = ""
        async for message in query(prompt=analysis_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        validation_output += block.text

        return validation_output

    async def _call_custom_llm(self, analysis_prompt: str) -> str:
        """Call custom LLM API (OpenAI-compatible)."""
        if not self.api_key or not self.base_url:
            raise ValueError("api_key and base_url are required for custom LLM API")

        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in prompt engineering and evaluation. You provide objective, constructive feedback on prompts based on established quality criteria. Focus on actionable recommendations to improve prompt effectiveness."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            "temperature": 0.3
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat/completions",
                                  headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"API request failed with status {response.status}: {await response.text()}")

                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def validate_prompt(self, prompt: str, detailed: bool = False) -> Dict[str, any]:
        """Validate a prompt using Claude or custom LLM API for analysis.

        Args:
            prompt: The prompt to validate
            detailed: Whether to provide detailed feedback or just a summary

        Returns:
            Dictionary containing validation results
        """
        if detailed:
            analysis_prompt = f"""
            Please analyze the following prompt for quality, clarity, safety, and effectiveness:

            PROMPT TO ANALYZE:
            {prompt}

            Please evaluate the prompt based on these criteria:
            1. Clarity - Is the prompt clear and unambiguous?
            2. Specificity - Does the prompt provide sufficient context and detail?
            3. Safety - Does the prompt avoid any unsafe, biased, or inappropriate content?
            4. Structure - Is the prompt well-organized and logical?
            5. Goal-oriented - Does the prompt effectively guide toward the intended outcome?
            6. Conciseness - Is the prompt appropriately concise without unnecessary verbosity?

            For each criterion, provide:
            - Rating (1-5 scale)
            - Brief explanation of your rating
            - Specific suggestions for improvement if applicable

            Finally, provide an overall assessment with:
            - Overall score (1-5)
            - Summary of strengths
            - Summary of weaknesses
            - Specific recommendations for improvement
            """
        else:
            analysis_prompt = f"""
            Please briefly analyze the following prompt for quality and effectiveness:

            PROMPT TO ANALYZE:
            {prompt}

            Provide a concise assessment covering:
            - Overall quality rating (1-5)
            - Primary strengths
            - Primary weaknesses
            - 1-2 key suggestions for improvement
            """

        validation_result = {
            "original_prompt": prompt,
            "validation_output": "",
            "issues_found": [],
            "recommendations": [],
            "overall_score": None
        }

        # Call appropriate LLM based on type
        if self.llm_type == "claude":
            validation_result["validation_output"] = await self._call_claude(analysis_prompt)
        else:
            validation_result["validation_output"] = await self._call_custom_llm(analysis_prompt)

        # Extract key information from validation output
        validation_result["recommendations"] = self._extract_recommendations(
            validation_result["validation_output"]
        )
        validation_result["issues_found"] = self._extract_issues(
            validation_result["validation_output"]
        )
        validation_result["overall_score"] = self._extract_score(
            validation_result["validation_output"]
        )

        return validation_result

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from Claude's response."""
        recommendations = []

        # Look for common recommendation patterns
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in
                   ['suggestion', 'recommend', 'improve', 'consider', 'try', 'add', 'change', 'modify']):
                # Clean the line and add to recommendations
                cleaned = line.strip().replace('- ', '').replace('*', '').strip()
                if cleaned and len(cleaned) > 10:  # Avoid very short phrases
                    recommendations.append(cleaned)

        return recommendations

    def _extract_issues(self, text: str) -> List[str]:
        """Extract issues from Claude's response."""
        issues = []

        # Look for common issue patterns
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in
                   ['issue', 'problem', 'weakness', 'concern', 'vague', 'unclear', 'lack', 'missing']):
                cleaned = line.strip().replace('- ', '').replace('*', '').strip()
                if cleaned and len(cleaned) > 10:
                    issues.append(cleaned)

        return issues

    def _extract_score(self, text: str) -> Optional[int]:
        """Extract overall score from Claude's response."""
        import re

        # Look for patterns like "score: 4", "rating: 3", "(4/5)", etc.
        patterns = [
            r'overall score[:\s]+(\d+)',
            r'rating[:\s]+(\d+)',
            r'(\d+)[/\s]*5',  # Matches "4/5" or "4 out of 5"
            r'score[:\s]+(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 5:
                        return score
                except ValueError:
                    continue

        return None

    async def validate_multiple_criteria(self, prompt: str) -> Dict[str, any]:
        """Validate prompt against specific criteria separately."""
        results = {}

        # Define specific validation tasks
        validation_tasks = {
            "clarity": f"Analyze this prompt for clarity and understandability: {prompt}",
            "specificity": f"Analyze this prompt for specificity and detail: {prompt}",
            "safety": f"Analyze this prompt for safety, bias, and appropriateness: {prompt}",
            "structure": f"Analyze this prompt's organization and logical flow: {prompt}",
            "goal_achievement": f"Analyze how effectively this prompt guides toward its intended outcome: {prompt}",
            "conciseness": f"Analyze this prompt's conciseness and verbosity: {prompt}"
        }

        for criterion, task_prompt in validation_tasks.items():
            options = ClaudeAgentOptions(
                system_prompt=f"You are evaluating the quality of a prompt based on the criterion: {criterion.replace('_', ' ').title()}. Provide a rating from 1-5 and brief explanation."
            )

            result = ""
            async for message in query(prompt=task_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result += block.text

            results[criterion] = result

        return results


async def main():
    """Example usage of the PromptValidator."""
    validator = PromptValidator()

    # Example prompts to validate
    test_prompts = [
        "Write a story about a cat",
        "Write a detailed technical explanation of how neural networks work for a beginner audience, including at least 3 practical examples and a summary.",
        "Do something bad."
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"VALIDATING PROMPT: {prompt}")
        print(f"{'='*60}")

        # Perform detailed validation
        result = await validator.validate_prompt(prompt, detailed=True)

        print(f"\nOVERALL SCORE: {result['overall_score']}/5")
        print(f"\nVALIDATION OUTPUT:")
        print(result['validation_output'])

        if result['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for rec in result['recommendations']:
                print(f"- {rec}")

        if result['issues_found']:
            print(f"\nISSUES FOUND:")
            for issue in result['issues_found']:
                print(f"- {issue}")


if __name__ == "__main__":
    anyio.run(main)
