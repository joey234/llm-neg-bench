"""This file is specifically for compiling anchoring from questions and true ansswers, but
the basic pattern should be generalisable"""
from typing import cast
import pandas as pd
from pathlib import Path

raw_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/anchoring")
processed_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data")
df = pd.read_csv(Path(raw_data_path, "questions.csv"))

# TEST: adding a fewshot example
fewshot_example = "Q: What is the distance from Edinburgh to London (in miles)?\nA: The distance from Edinburgh to London (in miles) = 403.\n\n"
anchor_template = (
    fewshot_example + "Q: {anchor_sentence}\n{question_sentence}\nA: {answer_sentence}"
)
control_template = fewshot_example + "Q: {question_sentence}\nA: {answer_sentence}"

filled_template_list = []
true_answers_list = []
anchors_list = []
# comparisons need to be in order
comparisons = {"anchor_bigger": "less", "anchor_smaller": "greater"}
# the adjustment made to the true answer to get an anchor
adjustments = [1, 0.5, 1.5]
for i, row in df.iterrows():
    true_answer = cast(float, row["true_answer"])
    anchor_sentence_template = cast(str, row["anchor_sentence"])
    question_sentence = cast(str, row["question_sentence"])
    answer_sentence = cast(str, row["answer_sentence"])
    for adjustment in adjustments:
        anchor = int(true_answer * adjustment)

        if anchor > true_answer:
            comparison = comparisons["anchor_bigger"]
            template = anchor_template
            
        elif anchor < true_answer:
            comparison = comparisons["anchor_smaller"]
            template = anchor_template
        else:
            comparison = "NONE"
            template = control_template

        anchor_sentence = anchor_sentence_template.format(
            comparison=comparison, anchor=anchor
        )

        filled_template = template.format(
            anchor_sentence=anchor_sentence,
            question_sentence=question_sentence,
            answer_sentence=answer_sentence,
        )

        filled_template_list.append(filled_template)
        true_answers_list.append(true_answer)
        anchors_list.append(anchor)
        print(f"===\n{filled_template}\n===")

filled_template_df = pd.DataFrame(
    {
        "filled_template": filled_template_list,
        "true_answer": true_answers_list,
        "anchor": anchors_list,
    }
)
filled_template_df.to_csv(Path(processed_data_path, "anchoring.csv"))
print(filled_template_df.head())
print(filled_template_df.info())
