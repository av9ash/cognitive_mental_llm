from datetime import time
import json

exp_temp = {'Label': 'Final Classification'}


def zero_shot_cot(input_text):
    """Zero-shot Chain-of-Thought (CoT) prompting for suicide ideation classification."""
    prompt = f"""
    Analyze the following Reddit post and determine whether the user expresses suicidal thoughts. Think step by step before providing the final classification.

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases related to suicidal ideation, such as hopelessness, self-harm, or explicit suicidal statements.
    2. Determine whether the user describes distressing emotions and a lack of coping mechanisms.
    3. Assess if the user discusses intent, plans, or past suicide attempts.
    4. Based on this reasoning, classify the post into one of the following categories:

    - Suicidal (1): If the post indicates suicidal thoughts, plans, or extreme distress.
    - Not Suicidal (0): If the post does not contain signs of suicide ideation.

    Return the final classification: [1, 0], in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def few_shot_cot(input_text):
    """Few-shot Chain-of-Thought (CoT) prompting for suicide ideation classification."""

    example_prompts = """
    Example 1:
    Post: "I feel so alone, but I know this feeling will pass eventually."
    Step-by-step reasoning:
    1. The user acknowledges sadness but remains hopeful.
    2. No mention of suicidal thoughts or self-harm.
    3. Classification: Not Suicidal (0)

    Example 2:
    Post: "I don’t think I can do this anymore. I feel like there’s no way out."
    Step-by-step reasoning:
    1. The user expresses extreme distress and hopelessness.
    2. While not explicitly mentioning suicide, the language suggests suicide ideation.
    3. Classification: Suicidal (1)

    Example 3:
    Post: "I have written a letter to my family, I just can’t take it anymore."
    Step-by-step reasoning:
    1. The user references a suicide-related action (writing a farewell letter).
    2. Strong indicator of suicide intent.
    3. Classification: Suicidal (1)
    """

    prompt = f"""
    Below are examples of classifying Reddit posts based on suicide ideation. Follow the same reasoning method for the given post.

    {example_prompts}

    Now analyze the following post:

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases related to suicidal ideation, such as hopelessness, self-harm, or explicit suicidal statements.
    2. Determine whether the user describes distressing emotions and a lack of coping mechanisms.
    3. Assess if the user discusses intent, plans, or past suicide attempts.
    4. Based on this reasoning, classify the post into one of the categories: Suicidal (1) or Not Suicidal (0).

    Return only the final classification: [1, 0], in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def role_based_prompt(input_text, role="therapist"):
    """Role-based prompting for suicide ideation classification."""

    roles = {
        "therapist": """
        You are a licensed therapist analyzing a Reddit post to assess the risk of suicide ideation.
        Consider psychological indicators such as hopelessness, self-harm references, and distress levels.
        """,
        "researcher": """
        You are a mental health researcher categorizing Reddit posts based on suicide ideation risk.
        Use an evidence-based approach to classify the mental health state of the user.
        """
    }

    prompt = f"""
    {roles.get(role, roles['therapist'])}

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Analyze the language and emotions expressed in the post.
    2. Identify key indicators of suicide ideation, such as hopelessness, self-harm, or intent.
    3. Categorize the post into one of the following:
       - Suicidal (1): If the post indicates suicidal thoughts, plans, or extreme distress.
       - Not Suicidal (0): If the post does not contain signs of suicide ideation.

    Return only the final classification: [1, 0], in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def tree_of_thoughts(input_text):
    """Tree-of-Thought (ToT) prompting for suicide ideation classification (SDCNL dataset)."""

    prompt = f"""
    You are analyzing a Reddit post to determine whether the user is expressing suicidal thoughts.
    Instead of a single reasoning path, generate **three different thought trees** to explore multiple perspectives.

    Post: "{input_text}"

    **Tree 1: Emotional & Sentiment Analysis**
    - Identify key emotional tones (e.g., hopelessness, despair, frustration).
    - Does the post express extreme distress or any signs of hope?
    - Determine if emotional distress aligns with suicidal thoughts or if it is general sadness.

    **Tree 2: Behavioral Indicators**
    - Does the post mention suicidal behaviors (e.g., self-harm, plans, previous attempts)?
    - Does the user describe seeking help or avoiding social interactions?
    - Is there a history of discussing suicide attempts?

    **Tree 3: Risk & Protective Factors**
    - Are there protective factors (e.g., hope, therapy, social support)?
    - Are there clear indicators of increased suicide risk (e.g., farewell messages, withdrawal, explicit suicidal intent)?
    - How severe are the linguistic indicators of distress?

    **Evaluation & Final Classification**
    - Compare insights from all three thought trees.
    - Determine the most supported classification:
      - **Suicidal (1)**: If the post indicates suicidal thoughts, plans, or extreme distress.
      - **Not Suicidal (0)**: If the post does not contain signs of suicide ideation.

    Return only the final classification: [1, 0], in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt
