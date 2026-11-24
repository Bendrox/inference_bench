"""Générateur de charge multi-tours : N sessions concurrentes dont le contexte grandit à chaque tour.

`vllm bench serve` ne sait pas faire ça : chaque requête y est indépendante, aucun dataset ne
renvoie la vraie réponse du tour t dans le prompt du tour t+1 (sharegpt et ConversationDataset
ne gardent que le 1er tour, prefix_repetition a un préfixe fixe qui ne grandit pas).

    session k :  [system] -> u1 -> a1 (réponse réelle) -> u2 -> a2 -> ... -> uT
    requête t :  messages = system + u1 + a1 + ... + u(t-1) + a(t-1) + u(t)

Avec prefix caching, seul le nouveau morceau (a(t-1) + u(t)) est prefillé : TTFT ~ plat.
Sans, tout l'historique est re-prefillé : TTFT linéaire en la taille du contexte. Quand
sessions x contexte dépasse la capacité du KV cache, l'éviction LRU commence et le hit chute.

Sortie : <dossier>/run<i>.json avec les mêmes clés que `vllm bench serve --save-result`
(median_ttft_ms, output_throughput, ...) -> metrics.summarize marche tel quel, plus
`requests` (détail par tour) pour analysis/multiturn.py.

Paramètres : configs/dataset/multiturn.yaml (clés multiturn-*), surchargeables par --dataset-set.
"""
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

# Mots courants, ~1 token chacun avec le tokenizer Qwen : longueur approchée seulement,
# la vraie taille du prompt est relevée dans usage.prompt_tokens (c'est elle qu'on trace).
MOTS = ("le la un une de des et que pour dans avec sur plus mais comme tout faire voir "
        "agent outil code fichier test erreur valeur liste table appel retour").split()

# Valeurs par défaut des clés multiturn-* (configs/dataset/multiturn.yaml les surcharge)
DEFAUTS = {
    "multiturn-sessions": 16,        # nombre de conversations
    "multiturn-turns": 20,           # tours max par session
    "multiturn-system-len": 2048,    # prompt système (+ tools) en tokens
    "multiturn-shared-system": False,
    "multiturn-user-len": 256,       # nouveau message utilisateur / résultat d'outil à chaque tour
    "multiturn-output-len": 256,     # réponse du modèle à chaque tour
    "multiturn-think-time": 0.0,     # pause moyenne (s) entre deux tours d'une session
    "multiturn-timeout": 600,
}


def parametres(profil: dict) -> dict:
    """Profil dataset -> paramètres multi-tours ; refuse une clé inconnue (faute de frappe)."""
    inconnues = set(profil) - set(DEFAUTS) - {"dataset-name"}
    if inconnues:
        raise ValueError(f"clés inconnues pour dataset-name=multiturn : {sorted(inconnues)}")
    return {**DEFAUTS, **profil}


def texte_aleatoire(rng: random.Random, n_tokens: int) -> str:
    return " ".join(rng.choice(MOTS) for _ in range(n_tokens))


async def un_tour(client: AsyncOpenAI, model: str, messages: list, max_tokens: int) -> tuple[str, dict]:
    """Une requête chat en streaming. Retourne (texte de la réponse, mesures)."""
    t0 = time.perf_counter()
    ttft, morceaux, usage = None, [], None
    stream = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=0,
        stream=True,
        # Sans include_usage, le stream ne donne ni prompt_tokens ni cached_tokens.
        stream_options={"include_usage": True},
        extra_body={
            # Longueur de sortie fixe : sinon un tour qui s'arrête à 20 tokens et un autre à 256
            # font grandir le contexte à des vitesses différentes et les courbes deviennent bruitées.
            # Effet de bord : après son <|im_end|> le modèle continue en texte libre. Sans gravité
            # pour un bench, mais l'historique n'est pas une "vraie" conversation.
            "ignore_eos": True,
            # PIÈGE Qwen3 / Qwen3.5 : avec le thinking actif, le chat template RETIRE les
            # <think>...</think> des tours assistant passés. L'historique re-tokenisé au tour t+1
            # diffère donc de ce qui a été généré (et mis en cache) au tour t -> miss sur tout ce
            # qui suit la 1re réponse, et la comparaison cache / nocache est faussée.
            # Les templates qui ne connaissent pas ce paramètre l'ignorent.
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    async for ch in stream:
        if ch.choices:
            delta = ch.choices[0].delta
            # Le 1er chunk ne porte souvent que le rôle (contenu vide) : on attend du vrai texte.
            # reasoning_content au cas où un reasoning parser serait activé côté serveur.
            if ttft is None and (delta.content or getattr(delta, "reasoning_content", None)):
                ttft = time.perf_counter() - t0
            if delta.content:
                morceaux.append(delta.content)
        if ch.usage:
            usage = ch.usage
    e2e = time.perf_counter() - t0
    sortie = usage.completion_tokens if usage else 0
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return "".join(morceaux), {
        "ttft": ttft,
        "e2e": e2e,
        # même définition que vllm bench serve : temps de décodage / (tokens de sortie - 1)
        "tpot": (e2e - ttft) / (sortie - 1) if ttft is not None and sortie > 1 else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        # PIÈGE : cached_tokens n'est renvoyé que si le serveur tourne avec
        # --enable-prompt-tokens-details (cli.py l'ajoute pour ce dataset). Sinon : None.
        # PIÈGE modèle hybride (Qwen3.5 = couches linéaires GDN/Mamba + attention) : vLLM aligne
        # la taille de bloc sur la page Mamba ("Setting attention block size to 1056 tokens" dans
        # server.log) et le cache n'est réutilisable que par tranches de 1056 tokens. cached_tokens
        # monte donc en escalier (2112, 2112, 3168...) et jusqu'à 1055 tokens sont re-prefillés à
        # chaque tour : TTFT en dents de scie, hit < modèle à attention pure (bloc de 16, ex.
        # Qwen3-8B). Ne pas l'interpréter comme une éviction.
        "cached_tokens": details.cached_tokens if details else None,
        "output_tokens": sortie,
    }


async def session(k: int, p: dict, seed: int, client: AsyncOpenAI, model: str, max_ctx: int,
                  sem: asyncio.Semaphore, resultats: list) -> None:
    # Un rng par session : sessions différentes entre elles, mais reproductibles à seed égale.
    rng = random.Random(seed * 1_000_003 + k)
    # PIÈGE : system partagé = tous les agents ont les mêmes prompt système + tools. Le tour 0
    # touche alors le cache dès la 2e session : le gain du cache ne vient plus seulement de
    # l'historique. Laisser False pour mesurer l'effet "contexte qui grandit" seul.
    rng_sys = random.Random(seed) if p["multiturn-shared-system"] else rng
    messages = [{"role": "system", "content": texte_aleatoire(rng_sys, p["multiturn-system-len"])}]
    # PIÈGE : un seul message system, en tête. Le template Qwen exige system-first : un system
    # inséré en cours de conversation est fusionné dans le system de tête -> le préfixe change à
    # chaque tour et toute la conversation rate le cache (warning "system-first ordering" de vLLM).
    for t in range(p["multiturn-turns"]):
        messages.append({"role": "user", "content": texte_aleatoire(rng, p["multiturn-user-len"])})
        # PIÈGE concurrence : le sémaphore borne les requêtes EN VOL, pas les sessions.
        # Avec sessions > max-concurrency, des sessions attendent entre deux tours ; leurs blocs KV
        # ne sont plus référencés et deviennent évictables (LRU). C'est le cas agentique réel et
        # le paramètre le plus intéressant à balayer : sessions=conc ne montre pas l'éviction
        # tant que tout tient en KV cache.
        async with sem:
            try:
                reponse, m = await un_tour(client, model, messages, p["multiturn-output-len"])
            except Exception as exc:       # timeout, contexte trop long... : la session s'arrête
                resultats.append({"session": k, "turn": t, "error": repr(exc)})
                return
        # On renvoie la réponse sous forme de TEXTE : le serveur la re-tokenise au tour suivant.
        # PIÈGE (mineur) : la re-tokenisation peut découper autrement que les tokens générés, et
        # le template ajoute <|im_end|> que le modèle n'a pas émis (ignore_eos). Le préfixe commun
        # s'arrête donc vers la fin de a(t-1) : le hit n'atteint jamais 100 %, c'est normal.
        messages.append({"role": "assistant", "content": reponse})
        resultats.append({"session": k, "turn": t, **m})
        # PIÈGE max-model-len : sans cette borne, le serveur rejette la requête (400) quand le
        # contexte dépasse max-model-len et la session meurt sur une erreur au lieu de s'arrêter.
        prochain = (m["prompt_tokens"] or 0) + m["output_tokens"] + p["multiturn-user-len"] \
            + p["multiturn-output-len"] + 64          # marge : balises du chat template
        if prochain > max_ctx:
            return
        if p["multiturn-think-time"]:
            # Temps "outil / humain" entre deux tours (loi exponentielle). Plus il est long,
            # plus les blocs de la session ont de chances d'être évincés avant son retour.
            await asyncio.sleep(rng.expovariate(1 / p["multiturn-think-time"]))


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, round(q / 100 * (len(vals) - 1)))]


def resume(requetes: list[dict], duree: float) -> dict:
    """Clés au format `vllm bench serve --save-result`, lues par metrics.summarize."""
    ok = [r for r in requetes if "error" not in r and r["ttft"] is not None]
    ttft = [r["ttft"] * 1000 for r in ok]
    tpot = [r["tpot"] * 1000 for r in ok if r["tpot"] is not None]
    sortie = sum(r["output_tokens"] for r in ok)
    entree = sum(r["prompt_tokens"] or 0 for r in ok)
    caches = [r["cached_tokens"] for r in ok]
    return {
        "completed": len(ok),
        "failed": len(requetes) - len(ok),
        "duration": duree,
        "total_input_tokens": entree,
        "total_output_tokens": sortie,
        "request_throughput": len(ok) / duree,
        "output_throughput": sortie / duree,
        # Compte TOUS les tokens du prompt, y compris ceux servis par le cache : ce débit
        # "grimpe" avec le cache sans que le GPU calcule plus. Comparer plutôt output_throughput.
        "total_token_throughput": (entree + sortie) / duree,
        "median_ttft_ms": statistics.median(ttft) if ttft else None,
        "p99_ttft_ms": _pct(ttft, 99),
        "median_tpot_ms": statistics.median(tpot) if tpot else None,
        "p99_tpot_ms": _pct(tpot, 99),
        # Taux de hit vu par le client (tokens servis par le cache / tokens du prompt). None si
        # le serveur n'a pas --enable-prompt-tokens-details. À comparer à srv_prefix_cache_hit_rate.
        "cached_tokens_ratio": (sum(caches) / entree) if entree and None not in caches else None,
    }


async def _warmup(client: AsyncOpenAI, model: str, n: int, seed: int) -> None:
    """Requêtes courtes, en parallèle, au contenu distinct des sessions : compile / chauffe les
    graphes CUDA sans pré-remplir le cache avec des préfixes que les sessions réutiliseraient."""
    rng = random.Random(-1 - seed)
    async def une():
        await un_tour(client, model, [{"role": "user", "content": texte_aleatoire(rng, 64)}], 16)
    await asyncio.gather(*(une() for _ in range(n)))


async def _run(dossier: Path, i: int, model: str, port: int, max_ctx: int, client_cfg: dict):
    p = parametres(client_cfg["profil"])
    # PIÈGE inter-runs : le serveur reste allumé pendant les N runs. Avec la même seed, le run 2
    # retrouverait les conversations du run 1 déjà en cache -> --vary-seed (seed = n° du run).
    seed = i if client_cfg.get("vary_seed") else 0
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x",
                         timeout=p["multiturn-timeout"],
                         # Pas de retry : un retry silencieux fausserait TTFT et débit.
                         max_retries=0)
    # Le pool httpx par défaut d'openai (1000 connexions) dépasse largement nos concurrences :
    # pas de file d'attente cachée côté client.
    await _warmup(client, model, client_cfg["num_warmups"], seed)
    sem = asyncio.Semaphore(client_cfg["max_conc"])
    requetes: list[dict] = []
    t_start = time.time()
    await asyncio.gather(*(session(k, p, seed, client, model, max_ctx, sem, requetes)
                           for k in range(p["multiturn-sessions"])))
    t_end = time.time()
    await client.close()

    out = resume(requetes, t_end - t_start)
    out["requests"] = sorted(requetes, key=lambda r: (r["session"], r["turn"]))
    out["params"] = {**p, "max_concurrency": client_cfg["max_conc"], "seed": seed, "max_ctx": max_ctx}
    (dossier / f"run{i}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return t_start, t_end, out


def run_multiturn(i: int, dossier: Path, config: dict, client: dict) -> tuple[float, float]:
    """Équivalent de loadgen.run_bench pour dataset-name=multiturn : retourne (t_start, t_end)
    de la partie mesurée (hors warmup) et écrit run<i>.json + bench_run<i>.txt."""
    model = config.get("served-model-name", config["model"])
    t_start, t_end, out = asyncio.run(_run(dossier, i, model, config.get("port", 8000),
                                           # absent de la config = défaut du modèle, inconnu ici :
                                           # on suppose 8192 (prudent) plutôt que de déborder
                                           config.get("max-model-len", 8192), client))
    lignes = [f"{k:<24}{v}" for k, v in out.items() if k not in ("requests", "params")]
    texte = "\n".join(["== Multi-turn benchmark ==", *lignes])
    print(texte)
    (dossier / f"bench_run{i}.txt").write_text(texte + "\n", encoding="utf-8")
    return t_start, t_end
