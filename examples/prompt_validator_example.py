#!/usr/bin/env python3
"""Simple example of using the prompt validation agent with support for custom LLMs."""

import anyio
from prompt_validator import PromptValidator


async def simple_validation_example():
    """Simple example of prompt validation with Claude."""
    print("Testing with Claude (default)...")
    validator = PromptValidator(llm_type="claude")
    
    # Example prompts to validate
    prompts_to_test = [
        "Write a story about a cat",
        "Write a detailed technical explanation of how neural networks work for a beginner audience, including at least 3 practical examples and a summary.",
        "Refactor the following Python code to improve efficiency and readability."
    ]
    
    for prompt in prompts_to_test:
        print(f"\n{'='*60}")
        print(f"VALIDATING PROMPT: {prompt}")
        print(f"{'='*60}")
        
        try:
            # Perform brief validation
            result = await validator.validate_prompt(prompt, detailed=False)
            
            print(f"\nOverall Score: {result['overall_score']}/5")
            print(f"Validation Output:\n{result['validation_output']}")
            
            if result['recommendations']:
                print(f"\nRecommendations:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"{i}. {rec}")
        except Exception as e:
            print(f"Error with Claude validation: {e}")
            print("Note: Claude requires Claude Code to be installed and configured properly")
            break  # Just show one error message


async def custom_llm_example():
    """Example of using prompt validation with custom LLM API."""
    print(f"\n{'='*60}")
    print("EXAMPLE: How to use with custom LLM API (e.g., OpenAI-compatible)")
    print(f"{'='*60}")
    
    # Example of how to initialize for OpenAI
    print("For OpenAI API:")
    print("validator = PromptValidator(")
    print("    llm_type='openai',")
    print("    api_key='your-openai-api-key',")
    print("    base_url='https://api.openai.com/v1',")
    print("    model='gpt-3.5-turbo'")
    print(")")
    
    print("\nFor other OpenAI-compatible APIs (e.g., Azure OpenAI, local Ollama, etc.):")
    print("validator = PromptValidator(")
    print("    llm_type='custom',")
    print("    api_key='your-api-key',")
    print("    base_url='https://your-llm-provider.com/v1',  # or http://localhost:11434/v1 for Ollama")
    print("    model='your-preferred-model'")
    print(")")


async def detailed_validation_example():
    """Detailed example of prompt validation."""
    print(f"\n{'='*60}")
    print("DETAILED VALIDATION WITH CLAUDE (if available)")
    print(f"{'='*60}")
    
    validator = PromptValidator(llm_type="claude")
    
    # Test with a more complex prompt
    complex_prompt = """
    You are a senior software architect. Analyze the following code snippet for security vulnerabilities, 
    performance issues, and code quality. Provide a detailed report with specific recommendations for 
    improvements, including code examples where appropriate. Focus on best practices for Python web 
    development, and consider common security patterns like input validation, output encoding, and 
    secure session management. Your analysis should be structured with clear sections for 
    vulnerabilities, performance, and code quality.
    """
    
    print(f"Prompt: {complex_prompt}")
    
    try:
        result = await validator.validate_prompt(complex_prompt, detailed=True)
        
        print(f"\nOverall Score: {result['overall_score']}/5")
        print(f"\nDetailed Analysis:\n{result['validation_output']}")
        
        if result['recommendations']:
            print(f"\nActionable Recommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec}")
    except Exception as e:
        print(f"Error with Claude validation: {e}")
        print("Note: Claude requires Claude Code to be installed and configured properly")


async def main():
    """Run prompt validation examples."""
    print("Prompt Validation Agent Examples")
    print("=" * 60)
    
    await simple_validation_example()
    await detailed_validation_example()
    await custom_llm_example()


if __name__ == "__main__":
    anyio.run(main)