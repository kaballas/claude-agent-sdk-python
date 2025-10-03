#!/usr/bin/env python3
"""Advanced prompt validation agent with multiple validation strategies."""

import anyio
from typing import Dict, List, Tuple
from claude_agent_sdk import (
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    query,
)


class AdvancedPromptValidator:
    """Advanced prompt validation using multiple validation strategies."""
    
    def __init__(self):
        self.validation_categories = {
            "clarity": "Does the prompt clearly state what is expected?",
            "completeness": "Does the prompt provide all necessary context?",
            "feasibility": "Is the requested task achievable within reasonable constraints?",
            "specificity": "Are the requirements detailed enough?",
            "tone": "Is the tone appropriate for the intended audience?",
            "safety": "Does the prompt avoid unsafe or inappropriate content?",
            "structure": "Is the prompt well-organized and easy to follow?"
        }
    
    async def validate_prompt_comprehensive(self, prompt: str) -> Dict[str, any]:
        """Perform comprehensive prompt validation."""
        results = {
            "prompt": prompt,
            "overall_assessment": "",
            "category_ratings": {},
            "detailed_feedback": {},
            "strengths": [],
            "improvements": [],
            "overall_score": 0
        }
        
        # Get overall assessment
        overall_prompt = f"""
        Please provide an overall assessment of this prompt:
        {prompt}
        
        Evaluate it holistically and provide:
        1. An overall score from 1-10
        2. Main strengths of the prompt
        3. Main areas for improvement
        4. Your general impression of its effectiveness
        """
        
        options = ClaudeAgentOptions(
            system_prompt="You are an expert prompt engineer. Provide objective, constructive feedback that helps improve prompt quality and effectiveness."
        )
        
        async for message in query(prompt=overall_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        results["overall_assessment"] += block.text
        
        # Validate each category
        for category, description in self.validation_categories.items():
            category_prompt = f"""
            Evaluate the following prompt based on the criterion: {description}
            
            PROMPT: {prompt}
            
            Provide:
            - A rating from 1-5
            - Brief explanation of the rating
            - Specific suggestions for improvement if needed
            """
            
            category_feedback = ""
            async for message in query(prompt=category_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            category_feedback += block.text
            
            results["detailed_feedback"][category] = category_feedback
        
        # Extract overall score from assessment
        results["overall_score"] = self._extract_score(results["overall_assessment"])
        
        # Extract strengths and improvements
        results["strengths"] = self._extract_items(results["overall_assessment"], "strength")
        results["improvements"] = self._extract_items(results["overall_assessment"], "improvement")
        
        return results
    
    async def validate_for_task(self, prompt: str, target_task: str) -> Dict[str, any]:
        """Validate how well a prompt fits a specific task."""
        validation_prompt = f"""
        Evaluate how well this prompt achieves the task: {target_task}
        
        PROMPT: {prompt}
        
        Consider:
        1. How well the prompt guides toward the specified task
        2. Whether the prompt provides appropriate constraints
        3. If the prompt would yield the desired output format
        4. What might be missing to better achieve the task
        
        Provide specific, actionable feedback.
        """
        
        options = ClaudeAgentOptions(
            system_prompt="You are an expert evaluator. Assess how well prompts align with their intended tasks and provide specific improvement suggestions."
        )
        
        feedback = ""
        async for message in query(prompt=validation_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        feedback += block.text
        
        return {
            "target_task": target_task,
            "validation_feedback": feedback,
            "prompt": prompt,
            "alignment_score": self._extract_score(feedback)
        }
    
    async def compare_prompts(self, prompt1: str, prompt2: str, task: str) -> Dict[str, any]:
        """Compare two prompts for the same task."""
        comparison_prompt = f"""
        Compare these two prompts for the task: {task}
        
        PROMPT 1: {prompt1}
        PROMPT 2: {prompt2}
        
        Evaluate which prompt would be more effective for achieving the task, and why.
        Provide:
        1. A preference with justification
        2. Strengths of each prompt
        3. Weaknesses of each prompt
        4. Suggestions for combining the best elements of both
        """
        
        options = ClaudeAgentOptions(
            system_prompt="You are an expert prompt evaluator. Provide objective, detailed comparisons between different prompts and their effectiveness for specific tasks."
        )
        
        comparison_result = ""
        async for message in query(prompt=comparison_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        comparison_result += block.text
        
        return {
            "task": task,
            "prompt1": prompt1,
            "prompt2": prompt2,
            "comparison": comparison_result
        }
    
    def _extract_score(self, text: str) -> int:
        """Extract score from text."""
        import re
        
        # Look for patterns like "score: 4", "rating: 3", "(4/5)", etc.
        patterns = [
            r'overall (?:score|rating)[:\s]+(\d+)',
            r'(\d+)[/\s]*10',  # Matches "7/10"
            r'score[:\s]+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    return min(max(score, 1), 10)  # Clamp between 1 and 10
                except ValueError:
                    continue
        
        return 5  # Default neutral score
    
    def _extract_items(self, text: str, item_type: str) -> List[str]:
        """Extract specific items (strengths, improvements) from text."""
        import re
        
        # Look for lists of items
        patterns = {
            "strength": [
                r'strengths?[:\s]*([^.!?]+?)(?:\n|$)',
                r'key strength.*?:\s*([^\n]+)',
                r'positive aspects?[:\s]*([^\n]+)'
            ],
            "improvement": [
                r'improvements?[:\s]*([^.!?]+?)(?:\n|$)',
                r'recommendations?[:\s]*([^.!?]+?)(?:\n|$)',
                r'areas? for improvement[:\s]*([^\n]+)'
            ]
        }
        
        items = []
        for pattern in patterns.get(item_type, []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split on common list separators
                subitems = re.split(r'[;\n\-\*]', match)
                for item in subitems:
                    item = item.strip()
                    if item and len(item) > 5:
                        items.append(item)
        
        return items


async def main():
    """Demonstrate the advanced prompt validator."""
    validator = AdvancedPromptValidator()
    
    print("Advanced Prompt Validation Examples")
    print("=" * 60)
    
    # Example 1: Comprehensive validation
    print("\n1. COMPREHENSIVE VALIDATION")
    print("-" * 60)
    
    test_prompt = "Write a detailed explanation of machine learning for beginners"
    result = await validator.validate_prompt_comprehensive(test_prompt)
    
    print(f"Prompt: {result['prompt']}")
    print(f"Overall Score: {result['overall_score']}/10")
    print(f"Overall Assessment:\n{result['overall_assessment']}")
    
    if result['strengths']:
        print(f"\nStrengths:")
        for s in result['strengths']:
            print(f"- {s}")
    
    if result['improvements']:
        print(f"\nImprovements:")
        for i in result['improvements']:
            print(f"- {i}")
    
    # Example 2: Task-specific validation
    print("\n2. TASK-SPECIFIC VALIDATION")
    print("-" * 60)
    
    task = "Generate a Python function that validates email addresses"
    task_result = await validator.validate_for_task(test_prompt, task)
    
    print(f"Task: {task_result['target_task']}")
    print(f"Validation Feedback:\n{task_result['validation_feedback']}")
    
    # Example 3: Prompt comparison
    print("\n3. PROMPT COMPARISON")
    print("-" * 60)
    
    prompt_a = "Write code to validate email addresses"
    prompt_b = "Create a Python function that validates email addresses using regex. The function should return True for valid emails and False for invalid ones. Include examples of valid and invalid emails in your response."
    
    comparison = await validator.compare_prompts(prompt_a, prompt_b, task)
    print(f"Task: {comparison['task']}")
    print(f"Comparison:\n{comparison['comparison']}")


if __name__ == "__main__":
    anyio.run(main)