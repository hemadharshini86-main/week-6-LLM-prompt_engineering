import os
import re
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 1. LOAD HUGGING FACE MODEL
# ============================================================

MODEL = "google/flan-t5-small"

print("Loading Hugging Face model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

print("Model loaded successfully!")


# ============================================================
# 2. FILE LOCATIONS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input dataset
DATASET = os.path.join(BASE_DIR, "dataset")

PROMPTS_FILE = os.path.join(
    DATASET,
    "prompts.csv"
)

# Output files - saved beside main.py
GENERATED_FILE = os.path.join(
    BASE_DIR,
    "generated_responses.csv"
)

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "evaluation_results.csv"
)

COMPARISON_FILE = os.path.join(
    BASE_DIR,
    "prompt_comparison.csv"
)
# ============================================================
# 3. GENERATE RESPONSE USING HUGGING FACE
# ============================================================

def generate_response(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


# ============================================================
# 4. EVALUATE RESPONSE
# ============================================================

def evaluate_response(prompt, response):

    evaluation_prompt = f"""
Evaluate this AI response.

Question:
{prompt}

Response:
{response}

Give a score from 1 to 5 for each:

Relevance
Accuracy
Completeness

Return exactly:

Relevance: 1
Accuracy: 1
Completeness: 1
"""

    result = generate_response(evaluation_prompt)

    relevance = 0
    accuracy = 0
    completeness = 0

    # Extract Relevance score
    match = re.search(
        r"Relevance\s*:\s*([1-5])",
        result,
        re.IGNORECASE
    )

    if match:
        relevance = int(match.group(1))

    # Extract Accuracy score
    match = re.search(
        r"Accuracy\s*:\s*([1-5])",
        result,
        re.IGNORECASE
    )

    if match:
        accuracy = int(match.group(1))

    # Extract Completeness score
    match = re.search(
        r"Completeness\s*:\s*([1-5])",
        result,
        re.IGNORECASE
    )

    if match:
        completeness = int(match.group(1))

    # If the model does not return scores,
    # give a neutral score so the program can continue.
    if relevance == 0:
        relevance = 3

    if accuracy == 0:
        accuracy = 3

    if completeness == 0:
        completeness = 3

    # Weighted overall score
    overall = (
        relevance * 0.4
        + accuracy * 0.4
        + completeness * 0.2
    )

    return (
        relevance,
        accuracy,
        completeness,
        round(overall, 2)
    )


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("       PROMPT ENGINEERING EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset folder
    # --------------------------------------------------------

    if not os.path.exists(DATASET):

        print("\nERROR: dataset folder not found!")
        print(f"Expected folder: {DATASET}")

        return

    # --------------------------------------------------------
    # Check prompts.csv
    # --------------------------------------------------------

    if not os.path.exists(PROMPTS_FILE):

        print("\nERROR: prompts.csv not found!")

        print(
            f"Expected location: {PROMPTS_FILE}"
        )

        return

    # --------------------------------------------------------
    # Read prompts.csv
    # --------------------------------------------------------

    print("\nReading prompts.csv...")

    try:

        prompts_df = pd.read_csv(
            PROMPTS_FILE
        )

    except Exception as error:

        print("\nERROR reading prompts.csv:")
        print(error)

        return

    print("\nColumns found:")

    print(
        list(prompts_df.columns)
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "prompt_id",
        "technique",
        "prompt"
    ]

    missing_columns = []

    for column in required_columns:

        if column not in prompts_df.columns:

            missing_columns.append(column)

    if missing_columns:

        print("\nERROR: Required columns are missing.")

        print(
            "Missing columns:",
            missing_columns
        )

        print(
            "\nYour prompts.csv should contain:"
        )

        print(
            "prompt_id, technique, prompt"
        )

        return

    # --------------------------------------------------------
    # Generate responses
    # --------------------------------------------------------

    generated_results = []

    print("\n" + "=" * 60)
    print("GENERATING RESPONSES")
    print("=" * 60)

    for _, row in prompts_df.iterrows():

        prompt_id = row["prompt_id"]

        technique = row["technique"]

        prompt = row["prompt"]

        print(
            f"\n{prompt_id} - {technique}"
        )

        print(
            "Prompt:",
            prompt
        )

        try:

            response = generate_response(
                prompt
            )

            print(
                "Response:",
                response
            )

            generated_results.append({

                "prompt_id": prompt_id,

                "technique": technique,

                "prompt": prompt,

                "generated_response": response

            })

        except Exception as error:

            print(
                "Generation error:",
                error
            )

            generated_results.append({

                "prompt_id": prompt_id,

                "technique": technique,

                "prompt": prompt,

                "generated_response": "ERROR"

            })

    # --------------------------------------------------------
    # Save generated responses
    # --------------------------------------------------------

    generated_df = pd.DataFrame(
        generated_results
    )

    generated_df.to_csv(
        GENERATED_FILE,
        index=False
    )
    # ========================================================
    # 6. EVALUATE RESPONSES
    # ========================================================

    evaluation_results = []

    print("\n" + "=" * 60)
    print("EVALUATING RESPONSES")
    print("=" * 60)

    for _, row in generated_df.iterrows():

        if (
            row["generated_response"]
            == "ERROR"
        ):

            continue

        prompt = row["prompt"]

        response = row[
            "generated_response"
        ]

        print(
            f"\nEvaluating {row['prompt_id']}..."
        )

        try:

            (
                relevance,
                accuracy,
                completeness,
                overall
            ) = evaluate_response(
                prompt,
                response
            )

            evaluation_results.append({

                "prompt_id":
                    row["prompt_id"],

                "technique":
                    row["technique"],

                "relevance":
                    relevance,

                "accuracy":
                    accuracy,

                "completeness":
                    completeness,

                "overall_score":
                    overall

            })

            print(
                f"Relevance: {relevance}"
            )

            print(
                f"Accuracy: {accuracy}"
            )

            print(
                f"Completeness: {completeness}"
            )

            print(
                f"Overall Score: {overall}"
            )

        except Exception as error:

            print(
                "Evaluation error:",
                error
            )


    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    evaluation_df = pd.DataFrame(
        evaluation_results
    )

    evaluation_df.to_csv(
        EVALUATION_FILE,
        index=False
    )
    # ========================================================
    # 7. COMPARE PROMPTING TECHNIQUES
    # ========================================================

    if not evaluation_df.empty:

        comparison_df = (

            evaluation_df

            .groupby("technique")

            .agg({

                "relevance": "mean",

                "accuracy": "mean",

                "completeness": "mean",

                "overall_score": "mean"

            })

            .reset_index()

            .round(2)

        )

        # ----------------------------------------------------
        # Save comparison
        # ----------------------------------------------------

        comparison_df.to_csv(
            COMPARISON_FILE,
            index=False
        )

        # ----------------------------------------------------
        # Display comparison
        # ----------------------------------------------------

        print("\n" + "=" * 60)

        print(
            "PROMPT COMPARISON"
        )

        print("=" * 60)

        print(
            comparison_df.to_string(
                index=False
            )
        )
    else:

        print(
            "\nNo evaluation results available."
        )
# ============================================================
# 9. RUN MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
      main()