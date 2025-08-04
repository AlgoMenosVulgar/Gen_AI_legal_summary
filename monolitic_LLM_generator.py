import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence, Tuple

import faiss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import spacy
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud


load_dotenv()
api1 = os.getenv('llama3_70b_8192')
api2 = os.getenv('gemma2_9b_it')
api3 = os.getenv('llama_instant')
api4 = os.getenv('llama3_8b')


os.chdir(os.path.dirname(os.path.abspath(__file__)))  # make ./ relative to script location
legal_data = pd.read_excel("./Datos/sentencias_pasadas.xlsx").drop(columns="Tipo")


def load_spacy_model(model_name):
    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError:
        print(f"model not loaded. downloading model: '{model_name}'...")
        spacy.cli.download(model_name)
        return spacy.load(model_name, disable=["ner"])


nlp = load_spacy_model("es_core_news_sm")

STOP = nlp.Defaults.stop_words

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def clean_lemmas(doc):
    """Devuelve lemas filtrados y en minúsculas para un spaCy Doc."""
    return [
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha                       # solo letras
           and tok.lemma_.lower() not in STOP
    ]


def word_count_generator(df):
    all_lemmas = (
        lemma
        for doc in nlp.pipe(df["sintesis"].astype(str), batch_size=256)
        for lemma in clean_lemmas(doc)
    )
    counts = Counter(all_lemmas)                      # {lemma: frecuencia}

    freq_df = (
        pd.DataFrame(counts.items(), columns=["lemma", "count"])
          .sort_values("count", ascending=False)
          .reset_index(drop=True)
    )

    wc = WordCloud(
        width=1600, height=900,
        background_color="white",
        colormap="viridis",
        prefer_horizontal=0.9,
        normalize_plurals=False
    ).generate_from_frequencies(counts)

    return wc, freq_df


def word_plot(wc_var):
    plt.figure(figsize=(7, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return


def save_wordcloud(wc_obj, fname="word_cloud.png", dpi=300):
    """
    Render the WordCloud object and save it to fname as a PNG.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc_obj, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_current_figure(fname="last_figure.png", dpi=300):
    """
    Save whatever was last drawn in matplotlib (if you didn't use save_wordcloud directly).
    """
    fig = plt.gcf()
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


MODEL_KEYS = [
    ("llama3-70b-8192",      api1),
    ("gemma2-9b-it",         api2),
    ("llama-3.1-8b-instant", api3),
    ("llama3-8b-8192",       api4),
]

MODELS = [m for m, _ in MODEL_KEYS]
CLIENTS = [Groq(api_key=k) for _, k in MODEL_KEYS]
CALL_INTERVAL = 60 / 29.5


def worker(task):
    idx, content, column = task
    base_slot = idx % len(MODELS)

    prompt = (
        "En español, en una sola oración neutra (≤30 palabras) resume quién pidió qué, el resultado y por qué, tal como aparece en el texto."
        "Usa terminología legal, pero simple sin ser tan sofisticada, pero precisa"
        "Devuelve solo un JSON válido: {\"answer\":\"...\"}. "
        f"Texto: {content}"
    )

    for attempt in range(3):
        slot = (base_slot + attempt) % len(MODELS)
        model, client = MODELS[slot], CLIENTS[slot]

        time.sleep(CALL_INTERVAL)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=100,
                response_format={"type": "json_object"},
            )
            result_json = json.loads(resp.choices[0].message.content)
            result = result_json.get("answer", "")
            print(f"✔︎ Model{slot+1} row{idx+1} ({column}) (try{attempt+1})", flush=True)
            return idx, result
        except Exception as e:
            print(f"✖︎ Model{slot+1} row{idx+1} ({column}) failed (try{attempt+1}): {e}",
                  file=sys.stderr, flush=True)

    print(f"⚠︎ Row{idx+1} ({column}) gave up after 3 tries", file=sys.stderr, flush=True)
    return idx, "Failed"


def LLM_aumentation(df):
    tasks = []
    for idx, row in df.iterrows():
        tasks.append((idx, row['resuelve'], "Sentencia_LLM"))
        tasks.append((idx, row['sintesis'], "De_qué_trata_LLM"))

    results_sentencia, results_sintesis = {}, {}

    with ThreadPoolExecutor(len(MODELS)) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for fut in as_completed(futures):
            idx, result = fut.result()
            column = futures[fut][2]
            if column == "Sentencia_LLM":
                results_sentencia[idx] = result
            else:
                results_sintesis[idx] = result

    df["Sentencia_LLM"] = df.index.map(results_sentencia.get)
    df["De_qué_trata_LLM"] = df.index.map(results_sintesis.get)

    return df


def filter_topic(df, topics, cols=None):
    if not topics or isinstance(topics, str):
        raise TypeError("topics must be a non-empty list of strings")

    def normalize(s):
        s = str(s).lower()
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

    pattern = re.compile(r"\b(?:" + "|".join(re.escape(normalize(t)) for t in topics) + r")\b")
    cols = cols or ["Tema - subtema", "resuelve", "sintesis"]
    mask = df[cols].astype(str).applymap(lambda v: bool(pattern.search(normalize(v)))).any(axis=1)
    filtered = df.loc[mask].reset_index(drop=True)
    return filtered, len(df), len(filtered)


# ------ test

TOPIC = [
    "red social", "redes sociales",
    "plataforma social", "plataformas sociales",
    "plataforma de medios sociales", "plataformas de medios sociales",
    "medio social", "medios sociales",
    "plataforma de interacción", "plataformas de interacción",
    "comunidad en línea", "comunidades en línea",
    "comunidad virtual", "comunidades virtuales",
    "plataforma digital", "plataformas digitales",
    "sitio de redes sociales", "sitios de redes sociales",
    "servicio de microblogging", "microblogging",

    "facebook", "meta",
    "instagram", "threads",
    "twitter", "x", "tweet", "tweets",
    "snapchat", "tiktok", "linkedin", "youtube", "yt", "reddit",
    "pinterest", "whatsapp", "wa", "telegram", "discord", "tumblr",
    "vimeo", "flickr",

    "red social corporativa",
    "foro en línea", "foros en línea",
    "plataforma de networking",
    "plataforma colaborativa",
]

filtered, len_df, len_filtered = filter_topic(legal_data, TOPIC)


a, b, c = random.sample(range(len(filtered)), 3)
tres_demandas = filtered.loc[[a, b, c]].reset_index(drop=True)

tres_demandas_con_LLMs = LLM_aumentation(tres_demandas)
tres_demandas_con_LLMs.to_csv("resultados/tres_demandas_con_LLMs.csv", encoding="utf-8-sig")

wc, freq_df = word_count_generator(filtered)
word_plot = word_plot(wc)

# Save the wordcloud as PNG (you asked for "pgn"; using .png extension)
save_wordcloud(wc, "forecasting_word_cloud.PNG")

