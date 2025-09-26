from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import pandas as pd
import os
import tyro
import torch
from pydantic import BaseModel
from enum import Enum
import logging
import pdb
from dataclasses import dataclass
import json
import random
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError,
)

logger = logging.getLogger(__name__)


class GenderOption(str, Enum):
    gendered_female = "GENDERED FEMALE"
    gendered_male = "GENDERED MALE"
    gendered_both = "GENDERED BOTH"
    non_binary = "NON-BINARY"
    neutral_all = "NEUTRAL (ALL)"
    untranslated = "UNTRANSLATED"
    reworded = "REWORDED"
    error = "ERROR"


class Response(BaseModel):
    translated_seed: str
    gender: GenderOption


@dataclass
class FewShot:
    seed: str
    source: str
    passage: str
    assistant_response: dict

    def to_dict(self):
        return [
            {
                "role": "user",
                "content": f"Seed: {self.seed}\nSource:{self.source}\nTranslation: {self.passage}",
            },
            {"role": "assistant", "content": json.dumps(self.assistant_response)},
        ]


FEW_SHOTS = [
    FewShot(
        seed="lawyers",
        source="Clara Shortridge Foltz The project started with a single biography - that of Clara S. Foltz, the first woman lawyer in California. In the course of writing her life, Foltz's biographer, Professor Barbara Babcock, has compiled a wealth of information about her subject and the times in which she lived, and most particularly, the other women lawyers she knew. Therefore, an entire section of the website is devoted to Ms. Foltz and consists of her publications in the course of the biography-in-progress.",
        passage="Clara Shortridge Foltz Das Projekt begann mit einer einzigen Biografie - der von Clara S. Foltz, der ersten Anwältin in Kalifornien. Im Laufe des Schreibens ihres Lebens hat Foltz' Biografin, Professorin Barbara Babcock, eine Fülle von Informationen über ihr Thema und die Zeit, in der sie lebte, zusammengetragen, insbesondere über die anderen Anwältinnen, die sie kannte. Daher ist ein ganzer Abschnitt der Website der Frau Foltz gewidmet und besteht aus ihren Veröffentlichungen im Laufe der Biografie, die noch im Entstehen ist.",
        assistant_response={
            "translated_seed": "Anwältinnen",
            "gender": "GENDERED FEMALE",
        },
    ),
    FewShot(
        seed="experts",
        source="This left many women on a desperate quest for marriage leaving them vulnerable to the accusation of witchcraft whether they took part in it or not. Trial records from the Inquisition and secular courts discovered a link between prostitutes and supernatural practices. Professional prostitutes were considered experts in love and therefore knew how to make love potions and cast love related spells. Up until 1630, the majority of women accused of witchcraft were prostitutes.",
        passage="Dies ließ viele Frauen verzweifelt nach einer Heirat suchen, was sie anfällig für den Vorwurf der Hexerei machte, ob sie daran teilnahmen oder nicht. In den Akten der Inquisition und der weltlichen Gerichte wurde ein Zusammenhang zwischen Prostituierten und übernatürlichen Praktiken festgestellt. Professionelle Prostituierte galten als Experten in der Liebe und wussten daher, wie man Liebestränke braute und Liebeszauber wirken ließ. Bis 1630 waren die meisten der der Hexerei angeklagten Frauen Prostituierte.",
        assistant_response={
            "translated_seed": "Experten",
            "gender": "GENDERED MALE",
        },
    ),
    FewShot(
        seed="call center operators",
        source="The buzzing office was filled with the sound of ringing phones and busy conversations. Operators deftly handled inquiries with patience and professionalism. Call center operators, primarily women, played a vital role in maintaining customer satisfaction. Their ability to remain calm under pressure ensures a positive experience for all callers.",
        passage="Das summende Büro war erfüllt vom Klingeln der Telefone und den Gesprächen der Mitarbeiter. Die Mitarbeiterinnen und Mitarbeiter bearbeiteten Anfragen geschickt, geduldig und professionell. Die Mitarbeiterinnen und Mitarbeiter der Callcenter, die hauptsächlich Frauen waren, spielten eine wichtige Rolle bei der Aufrechterhaltung der Kundenzufriedenheit. Ihre Fähigkeit, unter Druck ruhig zu bleiben, sorgt für ein positives Erlebnis für alle Anrufer.",
        assistant_response={
            "translated_seed": "Die Mitarbeiterinnen und Mitarbeiter der Callcenter",
            "gender": "GENDERED BOTH",
        },
    ),
    FewShot(
        seed="participants",
        source="How it works The program is entirely online and self-paced. Candidates have 12 months to complete the program, though the average amount of time it takes most candidates is between 7 and 10 months. A bachelor's degree is required for acceptance into the American Board teaching certification program, and participants must pass a background check. Candidates enrolled in the program study for and take two certification exams- one that covers Professional Teaching Knowledge (PTK) and one that covers subject area knowledge.",
        passage="So funktioniert es Das Programm ist vollständig online und selbstgesteuert. Die Kandidaten haben 12 Monate Zeit, um das Programm abzuschließen, obwohl die durchschnittliche Zeit, die die meisten Kandidaten benötigen, zwischen 7 und 10 Monaten liegt. Für die Zulassung zum American Board Teaching Certification Program ist ein Bachelor-Abschluss erforderlich, und die Teilnehmer*innen müssen eine Hintergrundüberprüfung bestehen. Die im Programm eingeschriebenen Kandidaten studieren und legen zwei Zertifizierungsprüfungen ab - eine, die das Professional Teaching Knowledge (PTK) abdeckt, und eine, die das Fachwissen abdeckt.",
        assistant_response={
            "translated_seed": "Teilnehmer*innen",
            "gender": "NON-BINARY",
        },
    ),
    FewShot(
        seed="sailors",
        source="Early history According to old newspapers, rugby union in Hong Kong dates back to the late 1870s, which would establish Hong Kong as perhaps the oldest rugby playing nation in Asia. The players during this era were all British sailors and army/navy men, as well as police and merchant men. The first secretary of rugby in Hong Kong was Jock McGregor.",
        passage="Frühe Geschichte Nach alten Zeitungen reicht die Geschichte des Rugby-Union-Sports in Hongkong bis in die späten 1870er Jahre zurück, was Hongkong als die vielleicht älteste Rugby spielende Nation in Asien etablieren würde. Die Spieler in dieser Ära waren allesamt britische Seeleute und Armee-/Marineangehörige sowie Polizisten und Kaufleute. Der erste Sekretär des Rugby-Sports in Hongkong war Jock McGregor.",
        assistant_response={
            "translated_seed": "Seeleute",
            "gender": "NEUTRAL (ALL)",
        },
    ),
    FewShot(
        seed="freelancers",
        source="The Chopsticks, a Hong Kong female duo, covered this song on their first LP The Chopsticks: Sandra and Amina (1970). Ella Fitzgerald recorded it on her 1971 album Things Ain't What They Used to Be (And You Better Believe It) with English lyrics by Loryn Deane. The Sacramento Freelancers Drum and Bugle Corps performed this song as part of their 1976 show. Al Jarreau also did a cover version on his 1994 album Tenderness.",
        passage="Das Hongkonger Frauen-Duo The Chopsticks coverte den Song auf ihrer ersten LP The Chopsticks: Sandra and Amina (1970). Ella Fitzgerald nahm ihn 1971 mit englischen Texten von Loryn Deane für ihr Album Things Ain't What They Used to Be (And You Better Believe It) auf. Die Sacramento Freelancers Drum and Bugle Corps spielten den Song 1976 in ihrer Show. Al Jarreau coverte den Song ebenfalls auf seinem Album Tenderness von 1994.",
        assistant_response={
            "translated_seed": "Freelancers",
            "gender": "UNTRANSLATED",
        },
    ),
    FewShot(
        seed="participants",
        source="The competition location changes with each event, but is usually held in either Europe or South America. The Fistball World Championship for women is held one year prior to the Fistball World Championship for men, and generally in a different host country to the one chosen for the subsequent men's tournament. In the men's tournament, Germany, Austria, Switzerland, Brazil, Argentina, Chile, Namibia and Italy have competed in every tournament since 1990, while Serbia, USA and Japan have also been frequent participants. The first Fistball World Championship was held for men in 1968, and for women in 1994.",
        passage="Der Austragungsort des Wettbewerbs wechselt mit jedem Event, liegt aber normalerweise in Europa oder Südamerika. Die Faustball-Weltmeisterschaft der Frauen findet ein Jahr vor der Faustball-Weltmeisterschaft der Männer statt und wird normalerweise in einem anderen Gastgeberland als das für das nachfolgende Turnier der Männer ausgewählte Land ausgetragen. Im Turnier der Männer haben Deutschland, Österreich, die Schweiz, Brasilien, Argentinien, Chile, Namibia und Italien seit 1990 an jedem Turnier teilgenommen, während Serbien, die USA und Japan ebenfalls häufig teilgenommen haben. Die erste Faustball-Weltmeisterschaft wurde 1968 für Männer und 1994 für Frauen ausgetragen.",
        assistant_response={
            "translated_seed": "teilgenommen",
            "gender": "REWORDED",
        },
    ),
    FewShot(
        seed="hosts",
        source="The NYCGHA had 40 players in two teams in 2000, seven teams in 2005 and five teams in 2010. Players are sorted by skill level, and the league has a reputation for developing players' skills and welcoming women. The NYCGHA hosts and co-sponsors events to increase awareness of ice hockey in New York, gay sports in New York, and gay health issues. Since 2001, its annual Chelsea Challenge invites LGBT and LGBT-friendly players to compete in a friendly ice hockey tournament.",
        passage="Die NYCGHA hatte 2000 vierzig Spieler in zwei Teams, sieben Teams im Jahr 2005 und fünf Teams im Jahr 2010. Die Spieler werden nach ihrem Können eingeteilt, und die Liga hat den Ruf, die Fähigkeiten der Spieler zu entwickeln und Frauen willkommen zu heißen. Die NYCGHA veranstaltet und co-sponsert Veranstaltungen, um das Bewusstsein für Eishockey in New York, für Schwulen-Sport in New York und für Gesundheitsprobleme von Schwulen zu schärfen. Seit 2001 lädt ihre jährliche Chelsea Challenge LGBT- und LGBT-freundliche Spieler ein, an einem freundschaftlichen Eishockeyturnier teilzunehmen.",
        assistant_response={
            "translated_seed": "ERROR",
            "gender": "ERROR",
        },
    ),
]


class OpenAIClient:

    def __init__(
        self,
        model_name_or_path: str,
    ):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            logger.warning(
                "Could not import python-dotenv. Set environment variables manually."
            )

        self.model_name_or_path = model_name_or_path
        self.client = OpenAI()

    def __call__(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
    ):
        """Generate GPT4 completions using local images and prompts."""
        logger.info(f"Sample input: {texts[0]}")

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(payload):
            return self.client.beta.chat.completions.parse(**payload)

        completions = list()
        for idx, conv in tqdm(
            enumerate(texts),
            desc="Item",
            total=len(texts),
            disable=not show_progress_bar,
        ):
            payload = {
                "model": self.model_name_or_path,
                "messages": conv,
                "max_tokens": 512,
                "response_format": Response,
            }

            try:
                # chat_response = self.client.beta.chat.completions.parse(**payload)
                chat_response = completion_with_backoff(payload)
                response = chat_response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Retrying with OPENAI API failed.")
                logger.warning(f"Failing row {idx}, prompt: {conv}")
                print(e)
                response = '{"translation_seed": "ERROR", "gender": "ERROR"}'

            completions.append(response)

        return completions


def main(
    model_name_or_path: str,
    input_file: str,
    prompt_file: str,
    target_col: str,
    output_file: str,
    dry_run: bool = False,
):
    gpu_count = torch.cuda.device_count()

    # Load the input file
    df = pd.read_csv(input_file, sep="\t")
    if dry_run:
        df = df.head(3)

    df = df.loc[df[target_col].notna()]

    with open(prompt_file) as fp:
        prompt_template = fp.read()

    convs = list()
    for idx, row in df.iterrows():
        seed = row["seed"]
        passage = row[target_col]
        source = row["text"]

        few_shots = FEW_SHOTS.copy()
        # shuffle few_shots
        random.shuffle(few_shots)
        conv = list()
        for fs in few_shots:
            conv.extend(fs.to_dict())

        # add the prompt to the first shot's user content
        conv[0]["content"] = f"{prompt_template}\n\n{conv[0]['content']}"
        conv.append(
            {
                "role": "user",
                "content": f"Seed: {seed}\nSource:{source}\nTranslation: {passage}",
            }
        )
        convs.append(conv)

    if not "gpt" in model_name_or_path:
        has_rope_scaling = "gemma" in model_name_or_path

        # Load the model
        llm = LLM(
            model=model_name_or_path,
            dtype="bfloat16",
            max_model_len=8196,
            tensor_parallel_size=gpu_count,
            gpu_memory_utilization=0.9,
            disable_sliding_window=True if not has_rope_scaling else False,
            enable_prefix_caching=True if not has_rope_scaling else False,
        )
        tokenizer = llm.get_tokenizer()
        template_convs = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for conv in convs
        ]

        json_schema = Response.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=64,
            guided_decoding=guided_decoding_params,
        )

        outputs = llm.generate(
            template_convs,
            sampling_params,
        )

        output_texts = [o.outputs[0].text for o in outputs]
    else:
        client = OpenAIClient(model_name_or_path)
        output_texts = client(convs, show_progress_bar=True)

    output_df = df[["id", "seed", target_col]].copy()
    output_df["raw_output"] = output_texts

    output_df["label"] = [json.loads(o)["gender"] for o in output_texts]
    output_df.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Output saved to {output_file}")


if __name__ == "__main__":
    tyro.cli(main)
