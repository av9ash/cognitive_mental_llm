from datetime import time
import json

exp_temp = {'Label': 'Final Classification'}
def zero_shot_cot(input_text):
    """Zero-shot Chain-of-Thought (CoT) prompting for stress classification."""
    prompt = f"""
    Analyze the following Reddit post and determine whether the user is experiencing stress. Think step by step before providing the final classification.

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases related to stress, such as frustration, overwhelm, anxiety, or emotional distress.
    2. Determine if the user describes external stressors (e.g., work, relationships, finances).
    3. Assess whether the user expresses difficulty coping with situations.
    4. Based on this reasoning, classify the post into one of the following categories:

    - Stressed (1): If the post indicates high emotional distress, anxiety, or external pressure.
    - Not Stressed (0): If the post does not indicate significant stress or distress.

    Return only the final classification [0 OR 1] in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def few_shot_cot(input_text):
    """Few-shot Chain-of-Thought (CoT) prompting for stress classification."""

    example_prompts = """
    Example 1:
    Post: "I have so many deadlines coming up, and I feel completely overwhelmed."
    Step-by-step reasoning:
    1. The user expresses feeling overwhelmed due to work deadlines.
    2. External stressor (work) is mentioned.
    3. Signs of stress are present.
    4. Classification: Stressed (1)

    Example 2:
    Post: "Today was a good day. I finally got to relax and take a break from work."
    Step-by-step reasoning:
    1. The user describes a positive, relaxing experience.
    2. No signs of emotional distress or external stressors.
    3. Classification: Not Stressed (0)

    Example 3:
    Post: "My finances are a mess, and I have no idea how I’m going to pay my bills this month."
    Step-by-step reasoning:
    1. The user describes financial problems as a major stressor.
    2. Mentions feelings of uncertainty and difficulty coping.
    3. Classification: Stressed (1)
    """

    prompt = f"""
    Below are examples of classifying Reddit posts based on stress levels. Follow the same reasoning method for the given post.

    {example_prompts}

    Now analyze the following post:

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases related to stress, such as frustration, overwhelm, anxiety, or emotional distress.
    2. Determine if the user describes external stressors (e.g., work, relationships, finances).
    3. Assess whether the user expresses difficulty coping with situations.
    4. Based on this reasoning, classify the post into one of the categories: Stressed (1) or Not Stressed (0).

    Return only the final classification [0 OR 1] in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def role_based_prompt(input_text, role="therapist"):
    """Role-based prompting for stress classification."""

    roles = {
        "therapist": """
        You are a licensed therapist analyzing a Reddit post to assess stress levels.
        Consider psychological indicators such as emotional distress, external pressures, and coping difficulties.
        """
    }

    prompt = f"""
    {roles.get(role, roles['therapist'])}

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Analyze the language and emotions expressed in the post.
    2. Identify key indicators of stress, such as anxiety, frustration, or overwhelm.
    3. Categorize the post into one of the following:
       - Stressed (1): High emotional distress, anxiety, or external pressure.
       - Not Stressed (0): No significant signs of stress.

    Return only the final classification [0 OR 1] in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def tree_of_thoughts(input_text):
    """Tree-of-Thought (ToT) prompting for stress classification (Dreaddit dataset)."""

    prompt = f"""
    You are analyzing a Reddit post to determine whether the user is experiencing stress.
    Instead of a single reasoning path, generate **three different thought trees** to explore multiple perspectives.

    Post: "{input_text}"

    **Tree 1: Emotional & Sentiment Analysis**
    - Identify key emotions (e.g., frustration, anxiety, exhaustion).
    - Does the user express feeling overwhelmed or unable to cope?
    - Differentiate between temporary frustration and chronic stress.

    **Tree 2: External & Social Stressors**
    - Does the user describe external stressors (e.g., work, relationships, finances)?
    - Are they experiencing social isolation, burnout, or high pressure?
    - Are there mentions of seeking support or struggling alone?

    **Tree 3: Coping Mechanisms & Resilience**
    - Is there evidence of positive coping (e.g., exercise, relaxation, problem-solving)?
    - Are there maladaptive coping mechanisms (e.g., avoidance, substance use)?
    - How well is the user managing their stressors?

    **Evaluation & Final Classification**
    - Compare insights from all three thought trees.
    - Determine the most supported classification:
      - **Stressed (1)**: If the post indicates significant distress, anxiety, or external stressors.
      - **Not Stressed (0)**: If the post does not indicate overwhelming stress.

    Return only the final classification [0 OR 1] in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt