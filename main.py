import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of this file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 1

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = self._get_research_prompt(question, researcher)

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            # PR-14 (2026-10-16): breadth -- append a grounded-web second
            # source (Perplexity Sonar via OpenRouter). A second-source
            # failure NEVER kills the primary research: log and ship
            # single-source exactly as before.
            if self._RESEARCH_BREADTH and research:
                try:
                    _pr14_sonar = GeneralLlm(
                        model=self._RESEARCH_SECOND_SOURCE,
                        temperature=0.1, timeout=120, allowed_tries=1,
                        max_tokens=4096)
                    _pr14_supp = await _pr14_sonar.invoke(prompt)
                    if _pr14_supp and _pr14_supp.strip():
                        research = (research
                                    + "\n\n### Supplementary grounded-web research (second source: Perplexity Sonar)\n"
                                    + _pr14_supp.strip())
                        logger.info(f"PR14-BREADTH: appended {len(_pr14_supp)} chars second-source research for {question.page_url}")
                except Exception as _pr14_e:
                    logger.warning(f"PR14-BREADTH: second source failed for {question.page_url} (continuing single-source): {_pr14_e!r}")
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # PR-16 (2026-10-17): base-rate anchor + resolution-state guard
        # injected into the prompt below. Field evidence: base-rate
        # anchoring was the "single biggest lever" for a Spring-2026
        # winner; open-vs-resolved misreads are a recurring catastrophic
        # miss class. Prompt-only change; zero added cost per question.
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            Base-rate discipline (mandatory): before weighing the research above, name
            the reference class this question belongs to and state its empirical base rate
            ("Base rate: XX%"). That base rate is your anchor. Move away from it only for
            specific, dated, verifiable evidence in the research, and say explicitly what
            evidence justifies the movement. If the research contains no real signal, stay
            at or near the base rate -- do not manufacture conviction.

            Resolution-state guard: treat this question as OPEN and unresolved unless the
            research explicitly reports the resolution event with a date and a named source.
            Never go near-certain (outside 3%-97%) on the belief that the outcome "already
            happened" without such explicit confirmation. Forecast the resolution criteria
            exactly as written above, not the headline version of the question.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    # ADOPT-5 PR-3 2026-10-12: roster swap per ensemble_roster.yaml
    # (source of truth ~/assaybot/ensemble_roster.yaml, roster_version
    # 2026-10-12-a, mirrored here because Actions runners never see the box).
    # Higher-reasoning tier; weighted median; throwing members EXCLUDED loud;
    # disagreement alarm logs loud and NEVER forces agreement. Cost guard
    # drops members in drop_order while est cost/question exceeds cap,
    # floor 2 members, loud breach past the floor -- the $9.91 balance
    # ceiling stays account-side (Auto Top-Up OFF), never raised from code.
    # Est = (_EST_IN_TOKENS*pin + _EST_OUT_TOKENS*pout)/1e6; prices probed
    # from openrouter.ai/api/v1/models on 2026-10-12; re-probed 2026-10-16 (PR-13, +deepseek-v4-pro).
    _EST_IN_TOKENS = 5000   # OPERATOR-TUNABLE: prompt+research per call
    _EST_OUT_TOKENS = 4000  # OPERATOR-TUNABLE: completion incl. reasoning
    # PR9-TUNE (2026-10-14): hard completion cap (incl. reasoning tokens).
    # Kills litellm's 65536 max_tokens default behind the OpenRouter 402
    # pre-flight exclusions of run 38481072279. Real usage est ~4k (above).
    _MAX_COMPLETION_TOKENS = 24576  # OPERATOR-TUNABLE
    _ENSEMBLE_PRICES = {  # (usd_per_1M_prompt, usd_per_1M_completion)
        "openrouter/openai/gpt-5.4": (2.50, 15.00),
        "openrouter/openai/gpt-6.1-sol": (2.00, 10.00),
        "openrouter/anthropic/claude-sonnet-5.5": (2.00, 10.00),
        "openrouter/x-ai/grok-4.7": (2.00, 6.00),
        "openrouter/moonshotai/kimi-k2": (0.57, 2.30),
        "openrouter/deepseek/deepseek-v4-pro": (0.30, 0.59),  # PR-13: 6th lab, ~$0.004/q/pass
    }
    _ENSEMBLE_MEMBERS = [  # (model, weight) -- weight feeds weighted median
        ("openrouter/openai/gpt-5.4", 1.0),
        ("openrouter/openai/gpt-6.1-sol", 1.0),
        ("openrouter/anthropic/claude-sonnet-5.5", 1.0),
        ("openrouter/x-ai/grok-4.7", 0.7),
        ("openrouter/moonshotai/kimi-k2", 0.5),
        ("openrouter/deepseek/deepseek-v4-pro", 0.4),  # PR-13: cheap diversity seat (trial weight)
    ]
    _ENSEMBLE_DROP_ORDER = [
        "openrouter/x-ai/grok-4.7",
        "openrouter/openai/gpt-6.1-sol",
        "openrouter/moonshotai/kimi-k2",
        "openrouter/deepseek/deepseek-v4-pro",  # PR-13: cheapest member, drop last
    ]
    _ENSEMBLE_REASONING = "high"
    _MAX_EST_COST_PER_QUESTION_USD = 0.25
    _DISAGREEMENT_ALARM_SPREAD = 0.35
    _MIN_ENSEMBLE_MEMBERS = 2
    # PR10-REFRESH (2026-10-15): freshness pass dials (used in __main__ tournament branch).
    _REFRESH_ENABLED = True  # OPERATOR-TUNABLE: master switch for stale-binary re-forecast
    _REFRESH_MIN_AGE_HOURS = 72  # OPERATOR-TUNABLE: re-forecast binaries older than this
    _REFRESH_MAX_PER_RUN = 2  # OPERATOR-TUNABLE: cost cap per run (~2x single-question cost)
    # PR-18 (2026-10-18): ledger-nominated refresh candidates (PR-10b).
    _REFRESH_TELEMETRY = True  # OPERATOR-TUNABLE: scan assay/ensemble_members.jsonl too
    _REFRESH_TELEMETRY_MAX_FETCH = 40  # OPERATOR-TUNABLE: post fetches/run (API only, no LLM)
    _EXTREMITY_CAPS = (0.03, 0.97)  # re-emitted: SPAN1 swallows the original
    # PR11-CPBLEND + PR-12 (2026-10-14): aggregation pack dials.
    _CP_BLEND_W = 0.25  # OPERATOR-TUNABLE: community blend weight; 0 disables
    _EXTREMIZE_GAMMA = 1.0  # OPERATOR-TUNABLE: >1 extremizes, <1 regresses; 1.0 no-op
    # PR-14 (2026-10-16): research breadth -- grounded-web second source.
    # Winners average 1.75 research sources; AskNews-only put us at 1.0.
    _RESEARCH_BREADTH = True  # OPERATOR-TUNABLE: append second-source research
    _RESEARCH_SECOND_SOURCE = "openrouter/perplexity/sonar"  # OPERATOR-TUNABLE (~$0.009/q)
    # PR-17 (2026-10-17): monotone-cubic smoothing of aggregated percentile
    # curves (numeric+date). A third of our served volume rides this path.
    _PCHIP_SMOOTH = True  # OPERATOR-TUNABLE: False -> coarse list passes through
    _CONFIG_ERA = "fall2026-r2-sp1-hi-low-tok24k-rb2-gsh-bra-pchip-tmem"  # PR-18: +ledger-memory refresh nominations
    _MEMBERS_DUMP_PATH = "assay/ensemble_members.jsonl"

    # PR-15 (2026-10-16): gamma shadow telemetry for the B3 A/B readout.
    # Counterfactual extremization (gamma=1.3) logged per binary forecast;
    # the POSTED value is untouched. Join onto ledger resolutions later.
    @staticmethod
    def _pr15_shadow_extremize(p, gamma=1.3, caps=(0.03, 0.97)):
        import math as _pr15_m
        _c = min(max(p, caps[0]), caps[1])
        return 1.0 / (1.0 + _pr15_m.exp(-gamma * _pr15_m.log(_c / (1.0 - _c))))

    # PR-17 (2026-10-17): PCHIP smoothing of aggregated percentile curves.
    # scipy is an OPTIONAL forecasting-tools dep (absent on the Actions
    # runner), so this is a dependency-free Fritsch-Carlson monotone cubic.
    # Shape-preserving by construction: no overshoot; non-decreasing values
    # in -> non-decreasing out. ANY caller-side error -> coarse list passes
    # through; a smoothing failure must never kill a forecast.
    @staticmethod
    def _pr17_pchip_smooth(percentile_list, step=0.05):
        from forecasting_tools.data_models.numeric_report import (
            Percentile as _pr17_P)
        xs = [float(p.percentile) for p in percentile_list]
        ys = [float(p.value) for p in percentile_list]
        n = len(xs)
        if n < 3:
            return percentile_list
        hs = [xs[i + 1] - xs[i] for i in range(n - 1)]
        if any(_h <= 1e-9 for _h in hs):
            return percentile_list
        ds = [(ys[i + 1] - ys[i]) / hs[i] for i in range(n - 1)]

        def _pr17_edge(_h0, _h1, _m0, _m1):
            _d = ((2.0 * _h0 + _h1) * _m0 - _h0 * _m1) / (_h0 + _h1)
            if _d * _m0 <= 0.0:
                return 0.0
            if _m0 * _m1 < 0.0 and abs(_d) > 3.0 * abs(_m0):
                return 3.0 * _m0
            return _d

        d = [0.0] * n
        d[0] = _pr17_edge(hs[0], hs[1], ds[0], ds[1]) if n > 2 else ds[0]
        d[-1] = _pr17_edge(hs[-1], hs[-2], ds[-1], ds[-2]) if n > 2 else ds[-1]
        for i in range(1, n - 1):
            if ds[i - 1] * ds[i] <= 0.0:
                d[i] = 0.0
            else:
                _w1 = 2.0 * hs[i] + hs[i - 1]
                _w2 = hs[i] + 2.0 * hs[i - 1]
                d[i] = (_w1 + _w2) / (_w1 / ds[i - 1] + _w2 / ds[i])
        qs = []
        _q = xs[0]
        while _q < xs[-1] - 1e-12:
            qs.append(_q)
            _q += step
        if not qs or abs(qs[-1] - xs[-1]) > 1e-9:
            qs.append(xs[-1])
        out = []
        for xq in qs:
            i = 0
            while i < n - 2 and xq > xs[i + 1] + 1e-12:
                i += 1
            _t = (xq - xs[i]) / hs[i]
            _t2 = _t * _t
            _t3 = _t2 * _t
            _v = ((2.0 * _t3 - 3.0 * _t2 + 1.0) * ys[i]
                  + (_t3 - 2.0 * _t2 + _t) * hs[i] * d[i]
                  + (-2.0 * _t3 + 3.0 * _t2) * ys[i + 1]
                  + (_t3 - _t2) * hs[i] * d[i + 1])
            out.append(_v)
        _incr = ys[-1] >= ys[0]  # paranoia monotone guard, direction-aware
        for i in range(1, len(out)):
            if _incr and out[i] < out[i - 1]:
                out[i] = out[i - 1]
            elif not _incr and out[i] > out[i - 1]:
                out[i] = out[i - 1]
        return [_pr17_P(percentile=_q, value=_v) for _q, _v in zip(qs, out)]

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        members = list(self._ENSEMBLE_MEMBERS)
        est_cost = sum(
            (self._EST_IN_TOKENS * self._ENSEMBLE_PRICES[slug][0]
             + self._EST_OUT_TOKENS * self._ENSEMBLE_PRICES[slug][1]) / 1e6
            for slug, _wt in members)
        for drop_slug in self._ENSEMBLE_DROP_ORDER:
            if est_cost <= self._MAX_EST_COST_PER_QUESTION_USD:
                break
            if len(members) <= self._MIN_ENSEMBLE_MEMBERS:
                logger.warning(
                    "COST-GUARD breach: est $%.3f/question over cap $%.3f "
                    "at %d-member floor; proceeding -- the $9.91 ceiling "
                    "is account-side, never code-side",
                    est_cost, self._MAX_EST_COST_PER_QUESTION_USD,
                    len(members))
                break
            hit = next((mm for mm in members if mm[0] == drop_slug), None)
            if hit is not None:
                members.remove(hit)
                _pin, _pout = self._ENSEMBLE_PRICES[drop_slug]
                est_cost -= (self._EST_IN_TOKENS * _pin
                             + self._EST_OUT_TOKENS * _pout) / 1e6
                logger.warning(
                    "COST-GUARD dropped %s per drop_order; est "
                    "$%.3f/question, %d members remain",
                    drop_slug, est_cost, len(members))
        if est_cost > self._MAX_EST_COST_PER_QUESTION_USD:
            logger.warning(
                "COST-GUARD exhausted: drop_order spent, est "
                "$%.3f/question still over cap $%.3f with %d primaries; "
                "the $9.91 ceiling stays account-side, never code-side",
                est_cost, self._MAX_EST_COST_PER_QUESTION_USD,
                len(members))

        decimals, weights, reasonings = [], [], []
        for model_name, weight in members:
            short = model_name.split("/")[-1]
            try:
                member = GeneralLlm(model=model_name, temperature=0.3,
                                    timeout=120, allowed_tries=2,
                                    reasoning_effort=self._ENSEMBLE_REASONING, max_tokens=self._MAX_COMPLETION_TOKENS)
                reasoning = await member.invoke(prompt)
                logger.info(f"[{short}] reasoning for {question.page_url}")
                parsed: BinaryPrediction = await structure_output(
                    reasoning,
                    BinaryPrediction,
                    model=self.get_llm("parser", "llm"),
                    num_validation_samples=self._structure_output_validation_samples,
                    additional_instructions=parsing_instructions,
                )
                decimals.append(float(parsed.prediction_in_decimal))
                weights.append(weight)
                reasonings.append(f"[{short}]\n{reasoning}")
            except Exception as e:
                logger.warning(f"[{short}] excluded from ensemble: {e!r}")
        if not decimals:
            raise RuntimeError(
                f"all ensemble members failed for {question.page_url}")
        pairs = sorted(zip(decimals, weights))  # ascending by probability
        half = sum(weights) / 2.0
        acc = 0.0
        mid = pairs[-1][0]
        for val, wt in pairs:
            acc += wt
            if acc >= half:
                mid = val
                break
        spread = max(decimals) - min(decimals)
        if spread > self._DISAGREEMENT_ALARM_SPREAD:
            logger.warning(
                f"DISAGREEMENT-ALARM {question.page_url}: member spread "
                f"{spread:.3f} exceeds {self._DISAGREEMENT_ALARM_SPREAD}; "
                f"keeping weighted median {mid:.3f}, NO forced agreement")
        # PR-11 (2026-10-14): community-prediction blend, applied pre-caps.
        _pr11_cp = getattr(question, "community_prediction_at_access_time", None)
        if self._CP_BLEND_W > 0 and _pr11_cp is not None:
            mid = (1.0 - self._CP_BLEND_W) * mid + self._CP_BLEND_W * float(_pr11_cp)
            logger.info(f"CP-BLEND {question.page_url}: w={self._CP_BLEND_W} cp={_pr11_cp:.3f} -> {mid:.3f}")
        decimal_pred = max(self._EXTREMITY_CAPS[0],
                           min(self._EXTREMITY_CAPS[1], mid))
        # PR-12 (2026-10-14): extremization dial, applied post-caps; result re-clamped inside caps.
        if self._EXTREMIZE_GAMMA != 1.0:
            import math as _pr12_math
            _pr12_c = min(max(decimal_pred, self._EXTREMITY_CAPS[0]), self._EXTREMITY_CAPS[1])
            decimal_pred = 1.0 / (1.0 + _pr12_math.exp(-self._EXTREMIZE_GAMMA * _pr12_math.log(_pr12_c / (1.0 - _pr12_c))))
            decimal_pred = min(max(decimal_pred, self._EXTREMITY_CAPS[0]), self._EXTREMITY_CAPS[1])
            logger.info(f"EXTREMIZE {question.page_url}: gamma={self._EXTREMIZE_GAMMA} -> {decimal_pred:.3f}")
        logger.info(f"Ensemble weighted median for {question.page_url}: "
                    f"{decimal_pred} from {pairs}")
        full_reasoning = ("\n\n---\n\n".join(reasonings)
                          + f"\n\nEnsemble median: {decimal_pred:.3f}")
        # ADOPT-5 PR-4: dump member (prob, weight) pairs per forecast so
        # the postmortem collector can join them onto the ledger by
        # question_id. Telemetry only -- a dump failure must NEVER kill a
        # forecast, hence the blanket try/except.
        try:
            member_record = {
                "question_id": getattr(question, "id_of_question", None),
                "post_id": getattr(question, "id_of_post", None),
                "page_url": getattr(question, "page_url", None),
                "config_era": self._CONFIG_ERA,
                "member_pairs": [[round(float(v), 4), float(w)]
                                 for v, w in pairs],
                "median": decimal_pred,
                "shadow_gamma_1_3": round(self._pr15_shadow_extremize(decimal_pred), 4),
                "shadow_gamma_sqrt3": round(self._pr15_shadow_extremize(decimal_pred, gamma=3.0 ** 0.5), 4),  # PR-16: Neyman-Roughgarden d=sqrt(3) counterfactual
                "n_members": len(pairs),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            dump_dir = os.path.dirname(self._MEMBERS_DUMP_PATH)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            with open(self._MEMBERS_DUMP_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(member_record) + "\n")
        except Exception as dump_err:
            logger.warning(
                f"MEMBERS-DUMP failed (forecast unaffected): {dump_err!r}")
        return ReasonedPrediction(prediction_value=decimal_pred,
                                  reasoning=full_reasoning)

    # ---- ADOPT-5 PR-3b 2026-10-12: ensemble machinery for MC/numeric/date.
    # Mirrors the binary path (PR-3): same roster, cost guard, exclusion
    # doctrine, disagreement alarm, member dump. Binary keeps its inline
    # loop untouched; these helpers serve the other three question types.
    # pprr stays 2 by operator decision -- the base class runs each of
    # these twice per question and aggregates; cost ceiling is account-side.
    def _cost_guarded_members(self):
        members = list(self._ENSEMBLE_MEMBERS)
        est_cost = sum(
            (self._EST_IN_TOKENS * self._ENSEMBLE_PRICES[slug][0]
             + self._EST_OUT_TOKENS * self._ENSEMBLE_PRICES[slug][1]) / 1e6
            for slug, _wt in members)
        for drop_slug in self._ENSEMBLE_DROP_ORDER:
            if est_cost <= self._MAX_EST_COST_PER_QUESTION_USD:
                break
            if len(members) <= self._MIN_ENSEMBLE_MEMBERS:
                logger.warning(
                    "COST-GUARD breach: est $%.3f/question over cap $%.3f "
                    "at %d-member floor; proceeding -- the $9.91 ceiling "
                    "is account-side, never code-side",
                    est_cost, self._MAX_EST_COST_PER_QUESTION_USD,
                    len(members))
                break
            hit = next((mm for mm in members if mm[0] == drop_slug), None)
            if hit is not None:
                members.remove(hit)
                _pin, _pout = self._ENSEMBLE_PRICES[drop_slug]
                est_cost -= (self._EST_IN_TOKENS * _pin
                             + self._EST_OUT_TOKENS * _pout) / 1e6
                logger.warning(
                    "COST-GUARD dropped %s per drop_order; est "
                    "$%.3f/question, %d members remain",
                    drop_slug, est_cost, len(members))
        if est_cost > self._MAX_EST_COST_PER_QUESTION_USD:
            logger.warning(
                "COST-GUARD exhausted: drop_order spent, est "
                "$%.3f/question still over cap $%.3f with %d primaries; "
                "the $9.91 ceiling stays account-side, never code-side",
                est_cost, self._MAX_EST_COST_PER_QUESTION_USD,
                len(members))
        return members

    async def _ensemble_reasonings(self, question, prompt):
        """Run the roster on prompt -> [(short, weight, reasoning)].
        Throwing members excluded loud; all-fail raises RuntimeError."""
        members_ok = []
        for model_name, weight in self._cost_guarded_members():
            short = model_name.split("/")[-1]
            try:
                member = GeneralLlm(model=model_name, temperature=0.3,
                                    timeout=120, allowed_tries=2,
                                    reasoning_effort=self._ENSEMBLE_REASONING, max_tokens=self._MAX_COMPLETION_TOKENS)
                reasoning = await member.invoke(prompt)
                logger.info(f"[{short}] reasoning for {question.page_url}")
                members_ok.append((short, weight, reasoning))
            except Exception as e:
                logger.warning(f"[{short}] excluded from ensemble: {e!r}")
        if not members_ok:
            raise RuntimeError(
                f"all ensemble members failed for {question.page_url}")
        return members_ok

    @staticmethod
    def _weighted_median_pairs(pairs):
        pairs = sorted(pairs)
        half = sum(w for _v, w in pairs) / 2.0
        acc = 0.0
        mid = pairs[-1][0]
        for val, wt in pairs:
            acc += wt
            if acc >= half:
                mid = val
                break
        return mid

    def _dump_member_record(self, question, qtype, extra):
        # Telemetry only -- a dump failure must NEVER kill a forecast.
        try:
            member_record = {
                "question_id": getattr(question, "id_of_question", None),
                "post_id": getattr(question, "id_of_post", None),
                "page_url": getattr(question, "page_url", None),
                "config_era": self._CONFIG_ERA,
                "question_type": qtype,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            member_record.update(extra)
            dump_dir = os.path.dirname(self._MEMBERS_DUMP_PATH)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            with open(self._MEMBERS_DUMP_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(member_record) + "\n")
        except Exception as dump_err:
            logger.warning(
                f"MEMBERS-DUMP failed (forecast unaffected): {dump_err!r}")

    async def _aggregate_percentile_members(self, question, members_ok,
                                            parsing_instructions, kind):
        """kind: "numeric" | "date". Per-level weighted median across
        members (coordinate-wise median of monotone series stays monotone;
        float noise coerced below). Level-set mismatch or parse failure
        excludes that member loud. Returns (percentile_list, member_pct)."""
        out_type = list[DatePercentile] if kind == "date" else list[Percentile]
        per_level = {}
        member_pct = {}
        level_set = None
        for short, weight, reasoning in members_ok:
            try:
                plist = await structure_output(
                    reasoning,
                    out_type,
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=self._structure_output_validation_samples,
                )
                conv = []
                for p in plist:
                    if kind == "date":
                        conv.append(Percentile(percentile=float(p.percentile),
                                               value=float(p.value.timestamp())))
                    else:
                        conv.append(Percentile(percentile=float(p.percentile),
                                               value=float(p.value)))
                conv.sort(key=lambda p: p.percentile)
                levels = [p.percentile for p in conv]
                if level_set is None:
                    level_set = levels
                if levels != level_set:
                    raise ValueError(
                        f"percentile level mismatch: {levels} vs {level_set}")
                member_pct[short] = [[round(p.percentile, 4), round(p.value, 6)]
                                     for p in conv]
                for p in conv:
                    per_level.setdefault(p.percentile, []).append((p.value, weight))
            except Exception as e:
                logger.warning(f"[{short}] {kind} parse excluded: {e!r}")
        if not member_pct:
            raise RuntimeError(
                f"all ensemble members failed to parse for {question.page_url}")
        percentile_list = [
            Percentile(percentile=lv,
                       value=self._weighted_median_pairs(pairs))
            for lv, pairs in sorted(per_level.items())
        ]
        for i in range(1, len(percentile_list)):
            if percentile_list[i].value < percentile_list[i - 1].value:
                percentile_list[i].value = percentile_list[i - 1].value
        # Disagreement alarm on the middle level, normalized to the bound
        # range so the 0.35 threshold means what it means on the binary path.
        if len(percentile_list) >= 2:
            mid_lv = percentile_list[len(percentile_list) // 2].percentile
            vals = [v for v, _w in per_level[mid_lv]]
            try:
                if kind == "date":
                    rng = (question.upper_bound.timestamp()
                           - question.lower_bound.timestamp())
                else:
                    rng = question.upper_bound - question.lower_bound
            except Exception:
                rng = None
            if rng and rng > 0 and vals:
                spread = (max(vals) - min(vals)) / rng
                if spread > self._DISAGREEMENT_ALARM_SPREAD:
                    logger.warning(
                        f"DISAGREEMENT-ALARM {question.page_url}: {kind} "
                        f"mid-level spread {spread:.3f} exceeds "
                        f"{self._DISAGREEMENT_ALARM_SPREAD}; keeping weighted "
                        f"median, NO forced agreement")

        # PR-8 (2026-10-13): clamp percentile values into the question's
        # declared bounds. Aggregating disagreeing members can push outer
        # percentiles out of bounds (observed: question 45340, bounds
        # [150000, 250000], p90 at 480000/520000); NumericDistribution
        # validation then rejects the whole prediction and the forecast never
        # posts. Clamp so every forecast can post. Date values are stored as
        # float timestamps, so bounds go through .timestamp() for kind=date.
        # Any clamp error logs and passes through unclamped (pre-PR-8
        # behavior); the clamp can never introduce a new failure mode.
        try:
            if kind == "date":
                _pr8_lo = question.lower_bound.timestamp()
                _pr8_hi = question.upper_bound.timestamp()
            else:
                _pr8_lo = question.lower_bound
                _pr8_hi = question.upper_bound
            if _pr8_lo is not None and _pr8_hi is not None:
                _pr8_clamped = 0
                for p in percentile_list:
                    if p.value < _pr8_lo:
                        p.value = _pr8_lo
                        _pr8_clamped += 1
                    elif p.value > _pr8_hi:
                        p.value = _pr8_hi
                        _pr8_clamped += 1
                if _pr8_clamped:
                    logger.warning(
                        f"PR8-CLAMP {question.page_url}: {kind} "
                        f"{_pr8_clamped} percentile(s) clamped into bounds")
        except Exception as e:
            logger.warning(f"PR8-CLAMP skipped ({e!r}); passing through unclamped")
        return percentile_list, member_pct

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        # ADOPT-5 PR-3b 2026-10-12: roster ensemble replaces the default-LLM
        # single-shot on this path (binary got it in PR-3). Per-option
        # weighted median, renormalized to sum 1 (PredictedOptionList
        # validator requires [0.99, 1.01]); member option-set mismatch or
        # parse failure excludes that member loud; dump/alarm same doctrine
        # as binary. pprr stays 2 by operator decision 2026-10-12.
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        members_ok = await self._ensemble_reasonings(question, prompt)
        option_names = [str(o) for o in question.options]
        per_option = {name: [] for name in option_names}
        member_probs = {}
        member_weights = {}
        for short, weight, reasoning in members_ok:
            try:
                pol: PredictedOptionList = await structure_output(
                    text_to_structure=reasoning,
                    output_type=PredictedOptionList,
                    model=self.get_llm("parser", "llm"),
                    num_validation_samples=self._structure_output_validation_samples,
                    additional_instructions=parsing_instructions,
                )
                got = {str(o.option_name): float(o.probability)
                       for o in pol.predicted_options}
                if set(got) != set(option_names):
                    raise ValueError(
                        f"option set mismatch: {sorted(got)} vs {sorted(option_names)}")
                member_probs[short] = got
                member_weights[short] = weight
                for name in option_names:
                    per_option[name].append((got[name], weight))
            except Exception as e:
                logger.warning(f"[{short}] MC parse excluded: {e!r}")
        if not member_probs:
            raise RuntimeError(
                f"all ensemble members failed to parse for {question.page_url}")
        medians = {name: self._weighted_median_pairs(per_option[name])
                   for name in option_names}
        total = sum(medians.values())
        if total <= 0:
            logger.warning(
                f"MC medians summed to {total} for {question.page_url}; "
                f"uniform fallback")
            medians = {name: 1.0 for name in option_names}
            total = float(len(option_names))
        final_probs = {name: medians[name] / total for name in option_names}
        worst = 0.0
        for name in option_names:
            vals = [p for p, _w in per_option[name]]
            if vals:
                worst = max(worst, max(vals) - min(vals))
        if worst > self._DISAGREEMENT_ALARM_SPREAD:
            logger.warning(
                f"DISAGREEMENT-ALARM {question.page_url}: MC option spread "
                f"{worst:.3f} exceeds {self._DISAGREEMENT_ALARM_SPREAD}; "
                f"keeping weighted medians, NO forced agreement")
        predicted_option_list = PredictedOptionList(predicted_options=[
            {"option_name": name, "probability": final_probs[name]}
            for name in option_names])
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        full_reasoning = (
            "\n\n---\n\n".join(f"[{s}]\n{r}" for s, _w, r in members_ok)
            + "\n\nEnsemble MC weighted medians (renormalized): "
            + json.dumps({k: round(v, 4) for k, v in final_probs.items()}))
        self._dump_member_record(question, "mc", {
            "member_option_probs": {
                s: {k: round(v, 4) for k, v in mp.items()}
                for s, mp in member_probs.items()},
            "member_weights": member_weights,
            "medians": {k: round(v, 4) for k, v in final_probs.items()},
            "n_members": len(member_probs),
        })
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=full_reasoning
        )
    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        # ADOPT-5 PR-3b 2026-10-12: roster ensemble replaces the default-LLM
        # single-shot on this path (binary got it in PR-3). parsing_instructions
        # kept VERBATIM from the pre-patch method. pprr stays 2 by operator
        # decision 2026-10-12.
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        members_ok = await self._ensemble_reasonings(question, prompt)
        percentile_list, member_pct = await self._aggregate_percentile_members(
            question, members_ok, parsing_instructions, "numeric"
        )
        if self._PCHIP_SMOOTH:  # PR-17
            try:
                _pr17_n0 = len(percentile_list)
                percentile_list = self._pr17_pchip_smooth(percentile_list)
                logger.info(f"PR17-PCHIP {question.page_url}: numeric smoothed {_pr17_n0} -> {len(percentile_list)} levels")
            except Exception as _pr17_e:
                logger.warning(f"PR17-PCHIP skipped for {question.page_url} ({_pr17_e!r}); coarse list passes through")
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        full_reasoning = (
            f"Ensemble: {len(members_ok)} members | medians: "
            + json.dumps([[round(p.percentile, 4), round(p.value, 6)] for p in percentile_list])
            + "\n\n"
            + "\n\n".join(
                f"===== {short} (w={weight}) =====\n{reasoning}"
                for short, weight, reasoning in members_ok
            )
        )
        self._dump_member_record(
            question,
            "numeric",
            {
                "member_percentiles": member_pct,
                "medians": [[round(p.percentile, 4), round(p.value, 6)] for p in percentile_list],
                "n_members": len(member_pct),
            },
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=full_reasoning
        )
    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self, question: DateQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        # ADOPT-5 PR-3b 2026-10-12: roster ensemble replaces the default-LLM
        # single-shot on this path (binary got it in PR-3). parsing_instructions
        # kept VERBATIM from the pre-patch method. pprr stays 2 by operator
        # decision 2026-10-12.
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        members_ok = await self._ensemble_reasonings(question, prompt)
        percentile_list, member_pct = await self._aggregate_percentile_members(
            question, members_ok, parsing_instructions, "date"
        )
        if self._PCHIP_SMOOTH:  # PR-17
            try:
                _pr17_n0 = len(percentile_list)
                percentile_list = self._pr17_pchip_smooth(percentile_list)
                logger.info(f"PR17-PCHIP {question.page_url}: date smoothed {_pr17_n0} -> {len(percentile_list)} levels")
            except Exception as _pr17_e:
                logger.warning(f"PR17-PCHIP skipped for {question.page_url} ({_pr17_e!r}); coarse list passes through")
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        full_reasoning = (
            f"Ensemble: {len(members_ok)} members | medians: "
            + json.dumps([[round(p.percentile, 4), str(p.value)[:10]] for p in percentile_list])
            + "\n\n"
            + "\n\n".join(
                f"===== {short} (w={weight}) =====\n{reasoning}"
                for short, weight, reasoning in members_ok
            )
        )
        self._dump_member_record(
            question,
            "date",
            {
                "member_percentiles": member_pct,
                "medians": [[round(p.percentile, 4), str(p.value)[:10]] for p in percentile_list],
                "n_members": len(member_pct),
            },
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=full_reasoning
        )
    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. The `llms=` block below is commented out to use
    # whichever default models forecasting-tools picks based on your env vars;
    # uncomment and edit to pin specific models.
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # PR-13 2026-10-16: single-pass; roster diversity replaces the second sample, halves burn
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
                skip_previously_forecasted_questions=True,

                llms={
            "default": GeneralLlm(
                model="openrouter/google/gemini-2.5-flash",
                temperature=0.3,
                timeout=120,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/google/gemini-2.5-flash",
            "researcher": "asknews/deep-research/low-depth",  # PR9B-REVERT 2026-10-14: medium pulls X (Spelunker-gated 403000); back to low
            "parser": "openrouter/google/gemini-2.5-flash",
        },

    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    if run_mode == "tournament":
        # PR-15 (B5, 2026-10-16): cross-tournament dedup shim. A question
        # co-listed in 33121 and minibench is forecasted once per run, not
        # twice. Instance-attribute shadow of the SDK method; passthrough
        # for anything not seen this run (incl. PR-10 refresh picks).
        _pr15_seen = set()
        _pr15_orig_fq = template_bot.forecast_questions
        async def _pr15_dedup(questions, *args, **kwargs):
            _fresh = []
            for _q in questions:
                _qid = (getattr(_q, "id_of_question", None)
                        or getattr(_q, "id_of_post", None) or id(_q))
                if _qid in _pr15_seen:
                    print(f"B5-DEDUP: skipping duplicate forecast for question {_qid} (already done this run)")
                    continue
                _pr15_seen.add(_qid)
                _fresh.append(_q)
            if not _fresh:
                return []
            return await _pr15_orig_fq(_fresh, *args, **kwargs)
        template_bot.forecast_questions = _pr15_dedup
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                33121, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
        # PR10-REFRESH (2026-10-15): re-forecast our oldest stale binaries.
        # Spot-at-close scoring punishes one-shot forecasts; top bots refresh.
        if template_bot._REFRESH_ENABLED:
            from datetime import datetime as _pr10_dt, timezone as _pr10_tz
            from forecasting_tools.data_models.questions import BinaryQuestion as _pr10_BQ
            _pr10_now = _pr10_dt.now(_pr10_tz.utc)
            _pr10_stale = []
            for _pr10_tid in (33121, client.CURRENT_MINIBENCH_ID):
                try:
                    _pr10_qs = template_bot.metaculus_client.get_all_open_questions_from_tournament(_pr10_tid)
                except Exception as _pr10_e:
                    print(f"PR10-REFRESH: candidate fetch failed for {_pr10_tid}: {_pr10_e}")
                    continue
                for _pr10_q in _pr10_qs:
                    _pr10_pfs = getattr(_pr10_q, "previous_forecasts", None)
                    if isinstance(_pr10_q, _pr10_BQ) and _pr10_pfs:
                        _pr10_age_h = (_pr10_now - max(_pf.timestamp for _pf in _pr10_pfs)).total_seconds() / 3600.0
                        if _pr10_age_h >= template_bot._REFRESH_MIN_AGE_HOURS:
                            _pr10_stale.append((_pr10_age_h, _pr10_q))
            # PR-18 (2026-10-18): PR10B-TMEMORY - ledger-nominated refresh candidates.
            # Feeds discover new questions; the ledger remembers old ones. Reads the
            # telemetry dump (_MEMBERS_DUMP_PATH), nominates tracked questions whose
            # last forecast is stale, fetches their posts (unpack subquestions for
            # group posts), and merges open binaries into the stale pool ahead of the
            # sort/cap. Feed candidates win dedup; all failure modes fail open.
            if getattr(template_bot, "_REFRESH_TELEMETRY", False):
                try:
                    import json as _pr10b_j
                    import os as _pr10b_os
                    import re as _pr10b_re
                    _pr10b_nom = {}
                    _pr10b_path = getattr(template_bot, "_MEMBERS_DUMP_PATH", "assay/ensemble_members.jsonl")
                    if _pr10b_path and _pr10b_os.path.exists(_pr10b_path):
                        with open(_pr10b_path) as _pr10b_f:
                            for _pr10b_line in _pr10b_f:
                                try:
                                    _pr10b_rec = _pr10b_j.loads(_pr10b_line)
                                except Exception:
                                    continue
                                _pr10b_qid = _pr10b_rec.get("question_id")
                                _pr10b_ts = _pr10b_rec.get("ts")
                                if not isinstance(_pr10b_qid, int) or not _pr10b_ts:
                                    continue
                                _pr10b_pid = _pr10b_rec.get("post_id")
                                if not isinstance(_pr10b_pid, int):
                                    _pr10b_m = _pr10b_re.search(r"/questions/(\d+)", str(_pr10b_rec.get("page_url") or ""))
                                    _pr10b_pid = int(_pr10b_m.group(1)) if _pr10b_m else None
                                if _pr10b_pid is None:
                                    continue
                                if _pr10b_qid not in _pr10b_nom or str(_pr10b_ts) > str(_pr10b_nom[_pr10b_qid][0]):
                                    _pr10b_nom[_pr10b_qid] = (_pr10b_ts, _pr10b_pid)
                    _pr10b_have = {getattr(_q, "id_of_question", None) for _a, _q in _pr10_stale}
                    _pr10b_stale = []
                    for _pr10b_qid, (_pr10b_ts, _pr10b_pid) in _pr10b_nom.items():
                        if _pr10b_qid in _pr10b_have:
                            continue
                        try:
                            _pr10b_dtlast = _pr10_dt.fromisoformat(str(_pr10b_ts).replace("Z", "+00:00"))
                        except Exception:
                            continue
                        _pr10b_age = (_pr10_now - _pr10b_dtlast).total_seconds() / 3600.0
                        if _pr10b_age >= template_bot._REFRESH_MIN_AGE_HOURS:
                            _pr10b_stale.append((_pr10b_age, _pr10b_qid, _pr10b_pid))
                    _pr10b_stale.sort(key=lambda _t: _t[0], reverse=True)
                    _pr10b_posts = {}
                    for _pr10b_age, _pr10b_qid, _pr10b_pid in _pr10b_stale:
                        _pr10b_posts.setdefault(_pr10b_pid, []).append((_pr10b_age, _pr10b_qid))
                    _pr10b_added = 0
                    _pr10b_fetched = 0
                    for _pr10b_pid, _pr10b_subs in _pr10b_posts.items():
                        if _pr10b_fetched >= getattr(template_bot, "_REFRESH_TELEMETRY_MAX_FETCH", 40):
                            break
                        _pr10b_fetched += 1
                        try:
                            _pr10b_got = template_bot.metaculus_client.get_question_by_post_id(
                                _pr10b_pid, group_question_mode="unpack_subquestions"
                            )
                        except Exception:
                            continue
                        if not isinstance(_pr10b_got, list):
                            _pr10b_got = [_pr10b_got]
                        _pr10b_want = {_qid: _age for _age, _qid in _pr10b_subs}
                        for _pr10b_q in _pr10b_got:
                            _pr10b_qid = getattr(_pr10b_q, "id_of_question", None)
                            if _pr10b_qid not in _pr10b_want:
                                continue
                            if isinstance(_pr10b_q, _pr10_BQ) and getattr(getattr(_pr10b_q, "state", None), "value", getattr(_pr10b_q, "state", None)) == "open":
                                _pr10_stale.append((_pr10b_want[_pr10b_qid], _pr10b_q))
                                _pr10b_added += 1
                    print(f"PR10B-TMEMORY: ledger {len(_pr10b_nom)} tracked; {len(_pr10b_stale)} stale nominees; +{_pr10b_added} open binaries ({_pr10b_fetched} post fetches)")
                except Exception as _pr10b_e:
                    print(f"PR10B-TMEMORY: pass failed open: {_pr10b_e!r}")
            _pr10_stale.sort(key=lambda _t: _t[0], reverse=True)
            _pr10_pick = [_q for _a, _q in _pr10_stale[: template_bot._REFRESH_MAX_PER_RUN]]
            print(f"PR10-REFRESH: {len(_pr10_stale)} stale binaries >{template_bot._REFRESH_MIN_AGE_HOURS}h; refreshing {len(_pr10_pick)} (cap {template_bot._REFRESH_MAX_PER_RUN})")
            if _pr10_pick:
                template_bot.skip_previously_forecasted_questions = False
                try:
                    forecast_reports = forecast_reports + asyncio.run(
                        template_bot.forecast_questions(_pr10_pick, return_exceptions=True))
                finally:
                    template_bot.skip_previously_forecasted_questions = True
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )


    # PR-7 (2026-10-13): closed-question tolerance. A question that closes
    # before the forecast POST returns 405 'already closed to forecasting';
    # it can never succeed, so counting it as a failure reds the whole run,
    # blocks the artifact commit, and re-burns research credit every cron.
    # Filter those exceptions out of the summary; real errors still raise.
    def _pr7_is_closed_skip(r):
        return isinstance(r, Exception) and "already closed to forecasting" in str(r)
    _pr7_closed_skips = [r for r in forecast_reports if _pr7_is_closed_skip(r)]
    if _pr7_closed_skips:
        print(f"PR7-SKIP: {len(_pr7_closed_skips)} question(s) already closed to forecasting; skipped, not counted as failures")
        forecast_reports = [r for r in forecast_reports if not _pr7_is_closed_skip(r)]
    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
        )
