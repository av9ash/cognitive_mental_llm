from datetime import time
import json

exp_temp = {'Label': 'Final Classification'}


def zero_shot_cot(input_text):
    """Zero-shot Chain-of-Thought (CoT) prompting for depression classification."""
    prompt = f"""
    Analyze the following Reddit post and determine the severity of depression expressed by the user. Think step by step before providing the final classification.

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases and emotions that indicate depressive symptoms.
    2. Assess the severity of distress, hopelessness, or other depressive markers.
    3. Determine whether the language suggests moderate or severe depression.
    4. Based on this reasoning, classify the post into one of the following categories:

    - Not Depression: If the post does not indicate significant depressive symptoms.
    - Moderate Depression: If the post reflects ongoing distress and mild functional impairment.
    - Severe Depression: If the post expresses intense distress, suicidal thoughts, or major functional impairment.
 
    Return the final classification: [Not Depression, Moderate, Severe], in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def few_shot_cot(input_text):
    """Few-shot Chain-of-Thought (CoT) prompting for depression classification."""

    example_prompts = """
    Example 1:
    Post: "I've been feeling okay lately, just a little tired from work but nothing too bad."
    Step-by-step reasoning:
    1. The user expresses general fatigue but not persistent sadness or distress.
    2. There is no mention of depressive symptoms like hopelessness or isolation.
    3. Classification: Not Depression

    Example 2:
    Post: "Lately, I feel like I'm stuck in a rut. I don’t enjoy things as much, and it’s been hard to find motivation."
    Step-by-step reasoning:
    1. The user describes ongoing distress and loss of motivation.
    2. Symptoms suggest moderate depression but not severe distress.
    3. Classification: Moderate Depression

    Example 3:
    Post: "I don’t see the point of anything anymore. I can’t get out of bed, and I keep thinking about ending it."
    Step-by-step reasoning:
    1. The post expresses extreme distress and suicidal ideation.
    2. This indicates severe depression with major functional impairment.
    3. Classification: Severe Depression
    """

    prompt = f"""
    Below are examples of classifying Reddit posts based on depression severity. Follow the same reasoning method for the given post.

    {example_prompts}

    Now analyze the following post:

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases and emotions that indicate depression symptoms.
    2. Assess the severity of distress, hopelessness, or other depressive markers.
    3. Determine whether the language suggests moderate or severe depression.
    4. Based on this reasoning, classify the post into one of the categories: Not Depression, Moderate Depression, Severe Depression.

    Return only the final classification: [Not Depression, Moderate, Severe], in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def role_based_prompt(input_text, role="therapist"):
    """Role-based prompting for depression classification."""

    roles = {
        "therapist": """
        You are a licensed therapist analyzing a Reddit post to assess the severity of depression.
        Consider psychological indicators such as emotional distress, loss of interest, and functional impairment.
        """
    }

    prompt = f"""
    {roles.get(role, roles['therapist'])}

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Analyze the language and emotions expressed in the post.
    2. Identify key indicators of mild, moderate, or severe depression.
    3. Categorize the post into one of the following:
       - Not Depression: If the post does not indicate significant depressive symptoms.
       - Moderate Depression: If the post reflects ongoing distress and mild functional impairment.
       - Severe Depression: If the post expresses intense distress, suicidal thoughts, or major functional impairment.

    Return only the final classification: [Not Depression, Moderate, Severe], in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def tree_of_thoughts(input_text):
    """Tree-of-Thought (ToT) prompting for depression severity classification (RedSam dataset)."""

    prompt = f"""
    You are analyzing a Reddit post to determine the severity of depression expressed by the user.
    Instead of a single reasoning path, generate **three different thought trees** to explore multiple perspectives.

    Post: "{input_text}"

    **Tree 1: Emotional & Cognitive Patterns**
    - Identify key emotional expressions (e.g., sadness, anxiety, hopelessness).
    - Assess cognitive distortions (e.g., "nothing matters," "I have no purpose").
    - Determine whether the language aligns with moderate or severe depression.

    **Tree 2: Functional Impairment & Behavioral Indicators**
    - Does the user describe withdrawal from daily activities or responsibilities?
    - Is there mention of sleep disturbances, appetite loss, or lack of motivation?
    - Are there signs of complete dysfunction or suicidal ideation?

    **Tree 3: Risk & Protective Factors**
    - Does the post contain protective elements (e.g., therapy, future goals, support systems)?
    - Are there risk factors such as isolation, self-harm, or major life stressors?
    - How persistent and severe are the depressive symptoms?

    **Evaluation & Final Classification**
    - Compare insights from all three thought trees.
    - Determine the most supported classification:
      - **Not Depression**: If the post does not indicate persistent depressive symptoms.
      - **Moderate Depression**: If distress and impairment exist but with some coping mechanisms.
      - **Severe Depression**: If extreme distress, dysfunction, or suicidal ideation is evident.

    Return only the final classification: [Not Depression, Moderate, Severe], in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt
