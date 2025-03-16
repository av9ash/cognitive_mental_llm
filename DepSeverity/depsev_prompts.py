from datetime import time
import json

exp_temp = {'Label': 'Final Classification'}

def zero_shot_cot(input_text):
    """Zero-shot Chain-of-Thought (CoT) prompting for depression severity classification."""
    prompt = f"""
    Analyze the following Reddit post and determine the severity of depression expressed by the user. Think step by step before providing the final classification.

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases and emotions that indicate depression symptoms.
    2. Assess the severity of distress, hopelessness, or other depressive markers.
    3. Determine whether the language suggests mild, moderate, or severe depression.
    4. Based on this reasoning, classify the post into one of the following categories:

    - Minimum: If the post shows no strong depressive symptoms.
    - Mild: If the post suggests occasional sadness but no significant impairment.
    - Moderate: If the post reflects ongoing distress and mild functional impairment.
    - Severe: If the post expresses intense distress, suicidal thoughts, or major functional impairment.

    Return only the final classification in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def few_shot_cot(input_text):
    """Few-shot Chain-of-Thought (CoT) prompting for depression severity classification."""

    example_prompts = """
    Example 1:
    Post: "I feel down sometimes, but I usually bounce back after a good night's sleep."
    Step-by-step reasoning:
    1. The user expresses sadness but not persistent distress.
    2. There is no indication of impairment or severe depressive symptoms.
    3. Classification: Minimum

    Example 2:
    Post: "I have been feeling really unmotivated lately. It’s getting harder to enjoy things, but I can still function."
    Step-by-step reasoning:
    1. The user mentions a loss of motivation but can still function.
    2. Suggests mild depression without significant distress.
    3. Classification: Mild

    Example 3:
    Post: "Lately, I feel like I’m trapped. I don’t enjoy anything, and everything feels pointless. I barely get out of bed."
    Step-by-step reasoning:
    1. The user describes a persistent sense of hopelessness.
    2. The lack of motivation and difficulty with daily activities indicate moderate depression.
    3. Classification: Moderate

    Example 4:
    Post: "I don’t see the point of anything anymore. I can’t get out of bed, and I keep thinking about ending it."
    Step-by-step reasoning:
    1. The post expresses extreme distress and suicidal ideation.
    2. This indicates severe depression with major functional impairment.
    3. Classification: Severe
    """

    prompt = f"""
    Below are examples of classifying Reddit posts based on depression severity. Follow the same reasoning method for the given post.

    {example_prompts}

    Now analyze the following post:

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases and emotions that indicate depression symptoms.
    2. Assess the severity of distress, hopelessness, or other depressive markers.
    3. Determine whether the language suggests mild, moderate, or severe depression.
    4. Based on this reasoning, classify the post into one of the categories: Minimum, Mild, Moderate, Severe.

    Return the final classification in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def role_based_prompt(input_text, role="therapist"):
    """Role-based prompting for classifying depression severity in Reddit posts."""

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
    2. Identify signs of mild, moderate, or severe depression.
    3. Categorize the post into one of the following:
       - Minimum: No strong depressive symptoms.
       - Mild: Occasional sadness, no significant impairment.
       - Moderate: Ongoing distress, mild functional impairment.
       - Severe: Intense distress, major impairment, or suicidal thoughts.

    Return only the final classification in a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt

def tree_of_thoughts(input_text):
    """Tree-of-Thought (ToT) prompting for depression severity classification (Reddit Depression dataset)."""

    prompt = f"""
    You are analyzing a Reddit post to determine the severity of depression expressed by the user.
    Instead of a single reasoning path, generate **three different thought trees** to explore multiple perspectives.

    Post: "{input_text}"

    **Tree 1: Emotional & Cognitive Patterns**
    - Identify key emotional expressions (e.g., sadness, despair, lack of motivation).
    - Assess thought patterns: does the user catastrophize, self-blame, or lose hope?
    - Determine if symptoms align with mild, moderate, or severe depression.

    **Tree 2: Functional Impairment & Daily Life Impact**
    - Does the user describe difficulty with daily tasks, relationships, or work?
    - Are there signs of avoidance, withdrawal, or social isolation?
    - Does the severity of impairment align with mild, moderate, or severe depression?

    **Tree 3: Risk & Protective Factors**
    - Does the post contain protective elements (e.g., seeking help, social support, therapy)?
    - Are there major risk factors such as past self-harm, suicidal thoughts, or trauma?
    - How persistent and severe are the depressive symptoms?

    **Evaluation & Final Classification**
    - Compare insights from all three thought trees.
    - Determine the most supported classification:
      - **Minimum**: No significant signs of depression.
      - **Mild**: Some distress but manageable symptoms.
      - **Moderate**: Persistent distress and impairment.
      - **Severe**: Major impairment, suicidal ideation, or complete dysfunction.

    Provide a structured JSON response summarizing each thought tree and the final classification:
    {{
        "Tree 1 - Emotional Patterns": "...",
        "Tree 2 - Functional Impairment": "...",
        "Tree 3 - Risk & Protective Factors": "...",
        "Label": "Final Classification"
    }}
    """
    return prompt
