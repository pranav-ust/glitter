import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import fire
import torch
import time
from datasets import load_dataset
import wandb


class PromptHelper:

    choices = [
        "AMBIGUOUS",
        "UNAMBIGUOUS (MALE)",
        "UNAMBIGUOUS (FEMALE)",
        "UNAMBIGUOUS (BOTH)",
    ]
    template = """You are a data filtering assistant. Your task is to determine whether the gender of the seed word is disambiguated by the context (previous, matching, and trailing sentences). Avoid making assumptions based on names or roles. Include the words "AMBIGUOUS" or "UNAMBIGUOUS" in your response. Keep your response concise and to the point.

CLEARLY DISAMBIGUATED EXAMPLES:

Parents as Trainers Example:
Seed: "trainers"
Previous: "Irina Iosifovna Turova (; born 10 August 1979), Irina Slavina (, also Irina Slavina-Turova), is a Russian chess player who holds the titles of Woman Grandmaster (2001) and International Master (2004). She won the Russian Women's Chess Championship in 2003."
Matching: "Biography Irina was born in Belarus, where her parents worked as chess trainers."
Trailing: "After Chernobyl disaster, her family moved to Arkhangelsk, where at age of eight Irina won the second place in the Arkhangelsk Oblast Women's Chess Championship."
Output: Based on the context, the word "parents" clearly indicates both male and female trainers working together. UNAMBIGUOUS (BOTH).

Female Employees Example:
Seed: "employees"
Previous: "The Kuwait government subsidizes certain types of finances for divorced or widowed Kuwaiti women provided they have children to care for. The Kuwait Labour Code provides maternity leave for women, who can receive 70-day of paid leave and up to four months of optional unpaid leave, during which employment termination is illegal."
Matching: "Additionally, employers with more than 50 female employees must provide infant childcare facilities by law."
Trailing: "The Civil Service Committee Decree of 1993 discriminates between women married to Kuwaiti men versus non-Kuwaiti men."
Output: I can see that the text explicitly uses the phrase "female employees" to specify gender. UNAMBIGUOUS (FEMALE).

Explicit Gender Mix Example:
Seed: "respondents"
Previous: "In Russia, according to a 2011 survey by the Southern Federal University, brunettes are considered more attractive than blondes."
Matching: "It is important to note that among the respondents in this study were 50% men and 50% women."
Trailing: "Another study by the University of Tampa, which also used male and female students..."
Output: Looking at the matching sentence, there's an explicit split of "50% men and 50% women" among the respondents. UNAMBIGUOUS (BOTH).

Female Participants Example:
Seed: "participants"
Previous: "The study focused on maternal health outcomes in rural communities."
Matching: "All participants were mothers who had given birth within the last year, recruited from local health centers."
Trailing: "The women reported varying levels of access to prenatal care."
Output: Based on multiple gender indicators - "mothers" and "women" - the participants are clearly identified as female. UNAMBIGUOUS (FEMALE).

Male Competitors Example:
Seed: "competitors"
Previous: "The heavyweight boxing division saw record attendance."
Matching: "The competitors, all male boxers between ages 20-35, were required to meet strict weight requirements."
Trailing: "Each fighter underwent medical screening before the matches."
Output: I can see that the text explicitly specifies "male boxers" when referring to the competitors. UNAMBIGUOUS (MALE).

Research Team Example:
Seed: "researchers"
Previous: "The international collaboration spanned multiple universities."
Matching: "The research team consisted of an equal split of male and female researchers, with 15 men and 15 women from various scientific backgrounds."
Trailing: "Their diverse perspectives contributed to the study's comprehensive approach."
Output: Looking at the matching sentence, there's an explicit count of "15 men and 15 women" researchers. UNAMBIGUOUS (BOTH).

AMBIGUOUS EXAMPLES:

Lawyers Example:
Seed: "lawyers"
Previous: "Earlier it was only given on certain grounds. Within the first two years of passing this law, the courts saw an exponential increase in khulʿ lawsuits."
Matching: "The law has yet to be approved by parliament, however, and it is still condemned by many lawyers to this day."
Trailing: "Nigeria Khulʿ is the most common form of divorce in Northern Nigeria."
Output: I see no gender indicators or specifications for these lawyers in any of the context sentences. AMBIGUOUS.

Investigators Example:
Seed: "investigators"
Previous: "A total of 43 knife wounds were inflicted on Uemura's body, including 31 to the neck during the assault, which lasted more than an hour. Populist weekly Shukan Shincho reported the wounds appeared to indicate that whoever killed Ryota may have been trying to decapitate him."
Matching: ""Some investigators suspect (the criminals) watched Internet videos showing the execution of hostages by Islamic State (IS) fighters and sought to mimic them," the magazine said, quoting an unnamed source close to police."
Trailing: "Prosecution On 27 February 2015, an 18-year-old boy was arrested by Kanagawa Police on suspicion of murder."
Output: Based on the context, there are no gender markers or specifications for the investigators mentioned. AMBIGUOUS.

Economists Example:
Seed: "economists"
Previous: "In response, the Club sponsored a petition of 1,028 economists who stated their opposition to protectionist policies against China. The list of economists included Nobel Laureates Finn Kydland, Edward Prescott, Thomas Schelling, and Vernon Smith."
Matching: "The petition played off a similar petition that was also signed by 1,028 economists in 1930 that opposed the Smoot-Hawley Tariff Act."
Trailing: "In 2008 and 2009, the Club for Growth opposed the $787 billion stimulus bill, Cash for Clunkers, cap and trade legislation, the Wall Street bailout, the auto bailout, the Affordable Care Act and the bailout of Fannie Mae and Freddie Mac."
Output: Looking at all context segments, I find no gender indicators for the general group of economists. AMBIGUOUS.

Beginners Example:
Seed: "beginners"
Previous: "Their game entertains without annoying, and there aren't many games, especially by mail, of which that can be said." In 1986, reviewer Dale A. Perkins stated that "If you are into Dungeons and Dragons' style combat, this is the game for you", recommending trying the game, regardless of gaming background."
Matching: "In 1991, reviewer Vickie Lloyd advised that her concerns with the game were "very minor" and Quest was "a great game and I very much recommend it", especially for beginners."
Trailing: "In 1985, the game tied with DuelMasters, Pellic Quest, and Power for Third Place in the 1st Annual Paper Mayhem Awards for "Best PBM Game"."
Output: I see no gender markers or specifications for the beginners mentioned in any of the context sentences. AMBIGUOUS.
"""

    def build_prompt(
        self,
        seed: str,
        p_ctx: str,
        m: str,
        t_ctx: str,
    ):
        # return f"""You are a data filtering assistant. Your task is to determine whether the gender of the seed word is disambiguated by the context (previous, matching, and trailing sentences). Avoid making assumptions based on names or roles. Include the words "AMBIGUOUS" or "UNAMBIGUOUS" in your response. Keep your response concise and to the point.
        return f"""{self.template}

Seed: "{seed}"
Previous: "{p_ctx}"
Matching: "{m}"
Trailing: "{t_ctx}"
Output:
"""


min_amounts = {
    "AMBIGUOUS": 1,
    "UNAMBIGUOUS (FEMALE)": 1,
    "UNAMBIGUOUS (MALE)": 900,
    "UNAMBIGUOUS (BOTH)": 1,
}
batch_size = 200


def main(model: str = "Qwen/Qwen2.5-72B-Instruct", dry_run: bool = False):
    # df = pd.read_json("results/filtering_sample_v1.jsonl", lines=True)
    # print(df.shape)
    wandb.init(
        project="building_bridges",
        name="label_with_model",
        config={
            "model": model,
            "dry_run": dry_run,
            "min_amounts": min_amounts,
            "batch_size": batch_size,
        },
    )

    data = load_dataset("Bainbridge/wikipedia_gnt_v2", split="train")
    if dry_run:
        print("Running a dry run")
        data = data.select(range(100))

    df = data.to_pandas()

    prompt_helper = PromptHelper()

    # r = r".*\s+(AMBIGUOUS|UNAMBIGUOUS \(MALE\)|UNAMBIGUOUS \(FEMALE\)|UNAMBIGUOUS \(BOTH\))\."
    # guided_decoding_params = GuidedDecodingParams(
    #     regex=r,
    # )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=128,
        # guided_decoding=guided_decoding_params,
    )

    llm = LLM(
        model=model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=1,
        max_model_len=2048,
    )

    texts = [
        prompt_helper.build_prompt(*row)
        for row in df[
            ["seed", "preceding_context", "matching_sentence", "trailing_context"]
        ].values
    ]

    print("Sample text")
    print(texts[0])

    def extract_label(x):
        if "UNAMBIGUOUS (MALE)" in x:
            return "UNAMBIGUOUS (MALE)"
        elif "UNAMBIGUOUS (FEMALE)" in x:
            return "UNAMBIGUOUS (FEMALE)"
        elif "UNAMBIGUOUS (BOTH)" in x:
            return "UNAMBIGUOUS (BOTH)"
        else:
            return "AMBIGUOUS"

    counts = {k: 0 for k in min_amounts.keys()}
    for i in range(0, df.shape[0], batch_size):
        print(f"Processing batch {i} to {i+batch_size}")

        outputs = llm.generate(texts[i : i + batch_size], sampling_params)
        outputs = [o.outputs[0].text for o in outputs]
        df.loc[i : i + batch_size - 1, "output"] = outputs

        labels = df.loc[i : i + batch_size - 1, "output"].apply(extract_label)
        df.loc[i : i + batch_size - 1, "label"] = labels

        # update counts for each label
        for label in labels:
            counts[label] += 1

        wandb.log(counts)

        # if all counts are above the minimum, break
        amount_of_processed_rows = i + batch_size
        partial_df = df.iloc[:amount_of_processed_rows]

        # Save first locally
        partial_df.to_json(
            f"results/wikipedia_gnt_v2_labeled_{model.replace('/', '--')}.json",
            orient="records",
            indent=2,
        )

        if all([counts[k] >= min_amounts[k] for k in min_amounts.keys()]):
            break

    # df = df.iloc[:amount_of_processed_rows]

    # outputs = llm.generate(texts, sampling_params)
    # outputs = [o.outputs[0].text for o in outputs]
    # df["output"] = outputs
    # df["label"] = df["output"].apply(extract_label)

    # Save first locally
    # df.to_json(
    #     f"results/wikipedia_gnt_v2_labeled_{model.replace('/', '--')}.json",
    #     orient="records",
    #     indent=2,
    # )
    # print("Saved into JSON")
    # df[
    #     [
    #         "seed",
    #         "preceding_context",
    #         "trailing_context",
    #         "matching_sentence",
    #         "max_attention_score",
    #         "output",
    #         "label",
    #     ]
    # ].to_csv(
    #     f"results/wikipedia_gnt_v2_labeled_{model.replace('/', '--')}.tsv", sep="\t"
    # )
    # print("Saved into TSV")

    data = data.select(range(amount_of_processed_rows))

    # Push to hub now
    data = data.add_column("output", partial_df["output"].tolist())
    data = data.add_column("label", df["label"].tolist())
    data.push_to_hub(
        "Bainbridge/wikipedia_gnt_v2",
        num_shards=4,
        config_name="qfilters_and_gwords_labeled",
    )


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Elapsed time: {time.time() - stime:.2f} seconds")
