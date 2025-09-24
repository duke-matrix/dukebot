# A forecasting bot using a multi-agent debate judged by a committee of five synthesizers.
import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import numpy as np
from forecasting_tools import (
    AskNewsSearcher,
    SmartSearcher,
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from newsapi import NewsApiClient
from tavily import TavilyClient

# -----------------------------
# Environment & API Keys
# -----------------------------
NEWSAPI_API_KEY = os.getenv("NEWSAPI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CommitteeForecastBot")


class CommitteeForecastingBot(ForecastBot):
    """
    This bot uses a proponent/opponent debate structure, which is then evaluated
    by a "committee" of multiple, independent synthesizer models. The median
    prediction from the committee is used as the final forecast.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        """
        Registers custom agent roles to suppress warnings.
        """
        defaults = super()._llm_config_defaults()
        defaults.update({
            "proponent": "mistralai/mistral-large-latest",
            "opponent": "openai/gpt-4o",
            "synthesizer_1": "openai/gpt-4o",
            "synthesizer_2": "anthropic/claude-3-opus-20240229",
            "synthesizer_3": "mistralai/mistral-large-latest",
            "synthesizer_4": "qwen/qwen-2-72b-instruct",
            "synthesizer_5": "openai/gpt-4o-mini",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.synthesizer_keys = [k for k in self.llms.keys() if k.startswith("synthesizer")]
        if not self.synthesizer_keys:
            raise ValueError("No synthesizer models found in LLM configuration. Please define at least one 'synthesizer_1'.")
        logger.info(f"Initialized with a committee of {len(self.synthesizer_keys)} synthesizers.")


    # --- Custom Research Implementation ---
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            logger.info(f"--- Running Raw Data Research for: {question.question_text} ---")
            loop = asyncio.get_running_loop()
            tasks = { "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text), "news": loop.run_in_executor(None, self.call_newsapi, question.question_text), }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            tavily_results, news_results = results[0], results[1]
            raw_research = f"Tavily Research Summary:\n{tavily_results}\n\nRecent News:\n{news_results}"
            logger.info(f"--- Synthesizing Raw Research for: {question.question_text} ---")
            synthesis_prompt = clean_indents(f"""
                Analyze the following raw research data from Tavily and NewsAPI, then provide a concise, synthesized summary for a forecaster.
                Focus on the key drivers, potential turning points, and any conflicting information.

                Raw Data:
                {raw_research}

                Synthesized Summary:
            """)
            # --- SYNTAX ERROR FIX: This line was moved from line 251 back to its correct location here ---
            synthesized_research = await self.get_llm("researcher", "llm").invoke(synthesis_prompt)
            logger.info(f"--- Research Complete for Q {question.page_url} ---\n{synthesized_research[:400]}...\n--------------------")
            return synthesized_research

    def call_tavily(self, query: str) -> str:
        if not self.tavily_client.api_key: return "Tavily search not performed (API key not set)."
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response['results']])
        except Exception as e: return f"Tavily search failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_client.api_key: return "NewsAPI search not performed (API key not set)."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'): return "No recent news articles found."
            return "\n".join([f"- Title: {a['title']}\n  Snippet: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e: return f"NewsAPI search failed: {e}"


    # --- Committee Forecasting Logic ---
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        logger.info(f"--- Starting Committee Debate for: {question.page_url} ---")
        today = datetime.now().strftime("%Y-%m-%d")

        proponent_prompt = clean_indents(f"""
            You are a professional superforecaster acting as a PROPONENT. Your goal is to build the strongest possible case for a YES outcome.
            Question: {question.question_text}, Background: {question.background_info}, Research: {research}, Today is {today}.
            Analyze the information and construct a persuasive argument for why the answer to the question will be YES.
            Start your response with your detailed rationale. Do not output a probability.
        """)
        proponent_argument = await self.get_llm("proponent", "llm").invoke(proponent_prompt)
        logger.info(f"Proponent argument generated for {question.page_url}")

        opponent_prompt = clean_indents(f"""
            You are a professional superforecaster acting as an OPPONENT. Your goal is to build the strongest possible case for a NO outcome.
            Question: {question.question_text}, Background: {question.background_info}, Research: {research}, Today is {today}.
            Analyze the information and construct a persuasive argument for why the answer to the question will be NO.
            Start your response with your detailed rationale. Do not output a probability.
        """)
        opponent_argument = await self.get_llm("opponent", "llm").invoke(opponent_prompt)
        logger.info(f"Opponent argument generated for {question.page_url}")

        synthesizer_prompt = clean_indents(f"""
            You are a professional superforecaster acting as a judge on a forecasting committee.
            Your task is to evaluate competing arguments to arrive at a final, precise probability.
            The question is: "{question.question_text}"
            Resolution Criteria: {question.resolution_criteria}
            Research Summary: {research}
            --- Proponent's Case for YES ---\n{proponent_argument}\n--- END OF PROPONENT'S CASE ---
            --- Opponent's Case for NO ---\n{opponent_argument}\n--- END OF OPPONENT'S CASE ---
            Today is {today}.
            Now, perform the following steps:
            1. Impartially summarize the strongest point from the proponent and the opponent.
            2. Identify any gaps or weaknesses in their arguments.
            3. Based on your evaluation, write your final integrated rationale.
            4. The very last thing you write is your final probability as: "Probability: ZZ%", from 0-100.
        """)
        
        logger.info(f"Presenting debate to the committee of {len(self.synthesizer_keys)} synthesizers...")
        tasks = [self.get_llm(key, "llm").invoke(synthesizer_prompt) for key in self.synthesizer_keys]
        synthesizer_reasonings_list = await asyncio.gather(*tasks, return_exceptions=True)
        synthesizer_reasonings_dict = dict(zip(self.synthesizer_keys, synthesizer_reasonings_list))

        logger.info("Parsing predictions from committee members...")
        parsing_tasks = [structure_output(r, BinaryPrediction, self.get_llm("parser", "llm")) for r in synthesizer_reasonings_list if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p.prediction_in_decimal for p in predictions if not isinstance(p, Exception)]
        
        if not valid_preds: raise ValueError("All synthesizer predictions failed parsing.")

        median_pred = float(np.median(valid_preds))
        final_pred = max(0.01, min(0.99, median_pred))
        
        combined_comment = self._format_committee_comment(proponent_argument, opponent_argument, synthesizer_reasonings_dict)

        logger.info(f"Forecasted {question.page_url} with committee median prediction: {final_pred} from {len(valid_preds)} valid predictions.")
        return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_comment)

    def _format_committee_comment(self, proponent_arg: str, opponent_arg: str, synth_reasonings: dict) -> str:
        comment = "--- DEBATE STAGE ---\n\n"
        comment += f"--- Argument from Proponent Agent ({self.get_llm('proponent', 'model_name')}) ---\n\n{proponent_arg}\n\n"
        comment += f"--- Argument from Opponent Agent ({self.get_llm('opponent', 'model_name')}) ---\n\n{opponent_arg}\n\n"
        comment += "--- COMMITTEE EVALUATION STAGE ---\n\n"

        for agent_key, reasoning in synth_reasonings.items():
            model_name = self.get_llm(agent_key, "model_name")
            comment += f"--- Synthesizer Analysis from {agent_key} ({model_name}) ---\n\n"
            comment += f"ERROR: {reasoning}\n\n" if isinstance(reasoning, Exception) else f"{reasoning}\n\n"
        return comment
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CommitteeForecastingBot.")
    parser.add_argument( "--mode", type=str, choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument( "--tournament-ids", nargs='+', type=str)
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode

    committee_bot = CommitteeForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model="openai/gpt-4o-mini"),
            "summarizer": GeneralLlm(model="openai/gpt-4o-mini"),
            "researcher": GeneralLlm(model="openai/gpt-4o", temperature=0.1),
            "parser": GeneralLlm(model="openai/gpt-4o"),
            
            "proponent": GeneralLlm(model="mistralai/mistral-large-latest", temperature=0.4),
            "opponent": GeneralLlm(model="openai/gpt-4o", temperature=0.4),

            "synthesizer_1": GeneralLlm(model="openai/gpt-4o", temperature=0.2),
            "synthesizer_2": GeneralLlm(model="anthropic/claude-3-opus-20240229", temperature=0.2),
            "synthesizer_3": GeneralLlm(model="mistralai/mistral-large-latest", temperature=0.2),
            "synthesizer_4": GeneralLlm(model="qwen/qwen-2-72b-instruct", temperature=0.2),
            "synthesizer_5": GeneralLlm(model="openai/gpt-4o-mini", temperature=0.2),
        },
    )

    try:
        if run_mode == "tournament":
            logger.info("Running in tournament mode...")
            tournament_ids_to_run = args.tournament_ids or [MetaculusApi.CURRENT_AI_COMPETITION_ID]
            logger.info(f"Targeting tournaments: {tournament_ids_to_run}")
            all_reports = []
            for tournament_id in tournament_ids_to_run:
                reports = asyncio.run(committee_bot.forecast_on_tournament(tournament_id, return_exceptions=True))
                all_reports.extend(reports)
            forecast_reports = all_reports
        elif run_mode == "test_questions":
            logger.info("Running in test questions mode...")
            EXAMPLE_QUESTIONS = ["https://www.metaculus.com/questions/578/human-extinction-by-2100/"]
            questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
            forecast_reports = asyncio.run(committee_bot.forecast_questions(questions, return_exceptions=True))

        committee_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")
    except Exception as e:
        # --- SYNTAX ERROR FIX: This line is now clean ---
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)


        if not self.tavily_client.api_key: return "Tavily search not performed (API key not set)."
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response['results']])
        except Exception as e: return f"Tavily search failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_client.api_key: return "NewsAPI search not performed (API key not set)."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'): return "No recent news articles found."
            return "\n".join([f"- Title: {a['title']}\n  Snippet: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e: return f"NewsAPI search failed: {e}"


    # --- Committee Forecasting Logic ---
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        logger.info(f"--- Starting Committee Debate for: {question.page_url} ---")
        today = datetime.now().strftime("%Y-%m-%d")

        # 1. Proponent builds the case FOR the event
        proponent_prompt = clean_indents(f"""
            You are a professional superforecaster acting as a PROPONENT. Your goal is to build the strongest possible case for a YES outcome.
            Question: {question.question_text}, Background: {question.background_info}, Research: {research}, Today is {today}.
            Analyze the information and construct a persuasive argument for why the answer to the question will be YES.
            Start your response with your detailed rationale. Do not output a probability.
        """)
        proponent_argument = await self.get_llm("proponent", "llm").invoke(proponent_prompt)
        logger.info(f"Proponent argument generated for {question.page_url}")

        # 2. Opponent builds the case AGAINST the event
        opponent_prompt = clean_indents(f"""
            You are a professional superforecaster acting as an OPPONENT. Your goal is to build the strongest possible case for a NO outcome.
            Question: {question.question_text}, Background: {question.background_info}, Research: {research}, Today is {today}.
            Analyze the information and construct a persuasive argument for why the answer to the question will be NO.
            Start your response with your detailed rationale. Do not output a probability.
        """)
        opponent_argument = await self.get_llm("opponent", "llm").invoke(opponent_prompt)
        logger.info(f"Opponent argument generated for {question.page_url}")

        # 3. Committee of Synthesizers evaluates the debate in parallel
        synthesizer_prompt = clean_indents(f"""
            You are a professional superforecaster acting as a judge on a forecasting committee.
            Your task is to evaluate competing arguments to arrive at a final, precise probability.
            The question is: "{question.question_text}"
            Resolution Criteria: {question.resolution_criteria}
            Research Summary: {research}
            --- Proponent's Case for YES ---\n{proponent_argument}\n--- END OF PROPONENT'S CASE ---
            --- Opponent's Case for NO ---\n{opponent_argument}\n--- END OF OPPONENT'S CASE ---
            Today is {today}.
            Now, perform the following steps:
            1. Impartially summarize the strongest point from the proponent and the opponent.
            2. Identify any gaps or weaknesses in their arguments.
            3. Based on your evaluation, write your final integrated rationale.
            4. The very last thing you write is your final probability as: "Probability: ZZ%", from 0-100.
        """)
        
        logger.info(f"Presenting debate to the committee of {len(self.synthesizer_keys)} synthesizers...")
        tasks = [self.get_llm(key, "llm").invoke(synthesizer_prompt) for key in self.synthesizer_keys]
        synthesizer_reasonings_list = await asyncio.gather(*tasks, return_exceptions=True)
        synthesizer_reasonings_dict = dict(zip(self.synthesizer_keys, synthesizer_reasonings_list))

        # 4. Parse all valid predictions from the committee members
        logger.info("Parsing predictions from committee members...")
        parsing_tasks = [structure_output(r, BinaryPrediction, self.get_llm("parser", "llm")) for r in synthesizer_reasonings_list if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p.prediction_in_decimal for p in predictions if not isinstance(p, Exception)]
        
        if not valid_preds: raise ValueError("All synthesizer predictions failed parsing.")

        # 5. Aggregate predictions to get the final forecast (Median)
        median_pred = float(np.median(valid_preds))
        final_pred = max(0.01, min(0.99, median_pred))
        
        # 6. Combine all arguments for the final comment
        combined_comment = self._format_committee_comment(proponent_argument, opponent_argument, synthesizer_reasonings_dict)

        logger.info(f"Forecasted {question.page_url} with committee median prediction: {final_pred} from {len(valid_preds)} valid predictions.")
        return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_comment)

    def _format_committee_comment(self, proponent_arg: str, opponent_arg: str, synth_reasonings: dict) -> str:
        """Formats the full debate and all committee reasonings into a single string."""
        comment = "--- DEBATE STAGE ---\n\n"
        comment += f"--- Argument from Proponent Agent ({self.get_llm('proponent', 'model_name')}) ---\n\n{proponent_arg}\n\n"
        comment += f"--- Argument from Opponent Agent ({self.get_llm('opponent', 'model_name')}) ---\n\n{opponent_arg}\n\n"
        comment += "--- COMMITTEE EVALUATION STAGE ---\n\n"

        for agent_key, reasoning in synth_reasonings.items():
            model_name = self.get_llm(agent_key, "model_name")
            comment += f"--- Synthesizer Analysis from {agent_key} ({model_name}) ---\n\n"
            comment += f"ERROR: {reasoning}\n\n" if isinstance(reasoning, Exception) else f"{reasoning}\n\n"
        return comment
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CommitteeForecastingBot.")
    parser.add_argument( "--mode", type=str, choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument( "--tournament-ids", nargs='+', type=str)
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode

    committee_bot = CommitteeForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model="openai/gpt-4o-mini"),
            "summarizer": GeneralLlm(model="openai/gpt-4o-mini"),
            "researcher": GeneralLlm(model="openai/gpt-4o", temperature=0.1),
            "parser": GeneralLlm(model="openai/gpt-4o"),
            
            # --- Debate Agents ---
            "proponent": GeneralLlm(model="mistralai/mistral-large-latest", temperature=0.4),
            "opponent": GeneralLlm(model="openai/gpt-4o", temperature=0.4),

            # --- MODIFIED: Committee of Five Synthesizers ---
            "synthesizer_1": GeneralLlm(model="openai/gpt-4o", temperature=0.2), # Covers "gpt o3" request
            "synthesizer_2": GeneralLlm(model="anthropic/claude-3-opus-20240229", temperature=0.2), # Covers "o3" request
            "synthesizer_3": GeneralLlm(model="mistralai/mistral-large-latest", temperature=0.2),
            "synthesizer_4": GeneralLlm(model="qwen/qwen-2-72b-instruct", temperature=0.2), # Added in place of Gemini
            "synthesizer_5": GeneralLlm(model="openai/gpt-4o-mini", temperature=0.2),
        },
    )

    try:
        if run_mode == "tournament":
            logger.info("Running in tournament mode...")
            tournament_ids_to_run = args.tournament_ids or [MetaculusApi.CURRENT_AI_COMPETITION_ID]
            logger.info(f"Targeting tournaments: {tournament_ids_to_run}")
            all_reports = []
            for tournament_id in tournament_ids_to_run:
                reports = asyncio.run(committee_bot.forecast_on_tournament(tournament_id, return_exceptions=True))
                all_reports.extend(reports)
            forecast_reports = all_reports
        elif run_mode == "test_questions":
            logger.info("Running in test questions mode...")
            EXAMPLE_QUESTIONS = ["https://www.metaculus.com/questions/578/human-extinction-by-2100/"]
            questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
            forecast_reports = asyncio.run(committee_bot.forecast_questions(questions, return_exceptions=True))

        committee_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")
    except Exception as e:
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)            synthesized_research = await self.get_llm("researcher", "llm").invoke(synthesis_prompt)

            logger.info(f"--- Research Complete for Q {question.page_url} ---\n{synthesized_research[:400]}...\n--------------------")
            return synthesized_research

    def call_tavily(self, query: str) -> str:
        """Performs a research search using Tavily."""
        if not self.tavily_client.api_key:
            return "Tavily search not performed (API key not set)."
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response['results']])
        except Exception as e:
            return f"Tavily search failed: {e}"

    def call_newsapi(self, query: str) -> str:
        """Fetches recent news articles from NewsAPI."""
        if not self.newsapi_client.api_key:
            return "NewsAPI search not performed (API key not set)."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'):
                return "No recent news articles found."
            return "\n".join([f"- Title: {a['title']}\n  Snippet: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e:
            return f"NewsAPI search failed: {e}"

    # --- Multi-Model Forecasting Logic ---

    async def _get_reasonings_from_all_models(self, prompt: str) -> list[str]:
        """Invokes all configured forecaster models with the same prompt."""
        tasks = [self.get_llm(key, "llm").invoke(prompt) for key in self.forecaster_keys]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _combine_reasonings_for_comment(self, reasonings: list) -> str:
        """Formats reasonings from all models into a single string for submission."""
        comment = ""
        for i, reasoning in enumerate(reasonings):
            model_key = self.forecaster_keys[i]
            model_name = self.get_llm(model_key, "model_name")
            comment += f"--- Reasoning from Model: {model_name} ---\n\n"
            comment += f"ERROR: {reasoning}\n\n" if isinstance(reasoning, Exception) else f"{reasoning}\n\n"
        return comment

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional superforecaster. Your interview question is:
            {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            {question.fine_print}
            Your research assistant says:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A scenario that results in a No outcome.
            (d) A scenario that results in a Yes outcome.
            You write your rationale, putting extra weight on the status quo.
            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasonings = await self._get_reasonings_from_all_models(prompt)
        parsing_tasks = [structure_output(r, BinaryPrediction, self.get_llm("parser", "llm")) for r in reasonings if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p.prediction_in_decimal for p in predictions if not isinstance(p, Exception)]
        if not valid_preds: raise ValueError("All model predictions failed parsing.")
        median_pred = float(np.median(valid_preds))
        final_pred = max(0.01, min(0.99, median_pred))
        combined_reasoning = self._combine_reasonings_for_comment(reasonings)
        logger.info(f"Forecasted {question.page_url} with median prediction: {final_pred}")
        return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional superforecaster. Your question is:
            {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            {question.resolution_criteria}
            Research:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            Write your rationale, considering the status quo and unexpected outcomes.
            The last thing you write is your final probabilities for the options {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            """
        )
        parsing_instructions = f"Ensure option names are one of: {question.options}"
        reasonings = await self._get_reasonings_from_all_models(prompt)
        parsing_tasks = [structure_output(r, PredictedOptionList, self.get_llm("parser", "llm"), additional_instructions=parsing_instructions) for r in reasonings if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p for p in predictions if not isinstance(p, Exception)]
        if not valid_preds: raise ValueError("All model predictions failed parsing.")
        avg_probs = {option: np.mean([p.get_prob(option) for p in valid_preds]) for option in question.options}
        total_prob = sum(avg_probs.values())
        final_probs = {option: prob / total_prob for option, prob in avg_probs.items()}
        final_prediction = PredictedOptionList(list(final_probs.items()))
        combined_reasoning = self._combine_reasonings_for_comment(reasonings)
        logger.info(f"Forecasted {question.page_url} with prediction: {final_prediction}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional superforecaster. Your question is:
            {question.question_text}
            Background: {question.background_info}
            Units for answer: {question.unit_of_measure or "Not stated"}
            Research:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_bound_message}
            {upper_bound_message}
            Write your rationale, considering expert expectations and unexpected scenarios.
            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 50: XX
            Percentile 90: XX
            "
            """
        )
        reasonings = await self._get_reasonings_from_all_models(prompt)
        parsing_tasks = [structure_output(r, list[Percentile], self.get_llm("parser", "llm")) for r in reasonings if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p for p in predictions if not isinstance(p, Exception)]
        if not valid_preds: raise ValueError("All model predictions failed parsing.")
        median_percentiles = []
        percentile_levels = sorted({p.percentile for pred_list in valid_preds for p in pred_list})
        for level in percentile_levels:
            values = [p.value for pred_list in valid_preds for p in pred_list if p.percentile == level]
            if values:
                median_percentiles.append(Percentile(percentile=level, value=np.median(values)))
        final_prediction = NumericDistribution.from_question(median_percentiles, question)
        combined_reasoning = self._combine_reasonings_for_comment(reasonings)
        logger.info(f"Forecasted {question.page_url} with prediction: {final_prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        upper_bound = question.nominal_upper_bound or question.upper_bound
        lower_bound = question.nominal_lower_bound or question.lower_bound
        upper_msg = f"Likely not higher than {upper_bound}." if question.open_upper_bound else f"Cannot be higher than {upper_bound}."
        lower_msg = f"Likely not lower than {lower_bound}." if question.open_lower_bound else f"Cannot be lower than {lower_bound}."
        return upper_msg, lower_msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the UnifiedForecastingBot.")
    parser.add_argument(
        "--mode", type=str, choices=["tournament", "test_questions"],
        default="tournament", help="Specify the run mode.",
    )
    parser.add_argument(
        "--tournament-ids", nargs='+', type=str,
        help="One or more tournament IDs or slugs to forecast on."
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode

    unified_bot = UnifiedForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model="openai/gpt-4o-mini"),
            "summarizer": GeneralLlm(model="openai/gpt-4o-mini"),
            "researcher": GeneralLlm(model="openai/gpt-4o-mini", temperature=0.1),
            "parser": GeneralLlm(model="openai/gpt-4o-mini"),
            # --- Three-Model Ensemble for Forecasting (No GPT-3.5) ---
            "forecaster_1": GeneralLlm(model="qwen/qwen-2-72b-instruct", temperature=0.3),
            "forecaster_2": GeneralLlm(model="openai/gpt-4o-mini", temperature=0.3),
            "forecaster_3": GeneralLlm(model="mistralai/mistral-large-latest", temperature=0.3),
        },
    )

    forecast_reports = []
    try:
        if run_mode == "tournament":
            logger.info("Running in tournament mode...")
            tournament_ids_to_run = args.tournament_ids or [
                MetaculusApi.CURRENT_AI_COMPETITION_ID,
                MetaculusApi.CURRENT_MINIBENCH_ID
            ]
            logger.info(f"Targeting tournaments: {tournament_ids_to_run}")

            all_reports = []
            for tournament_id in tournament_ids_to_run:
                reports = asyncio.run(
                    unified_bot.forecast_on_tournament(
                        tournament_id, return_exceptions=True
                    )
                )
                all_reports.extend(reports)
            forecast_reports = all_reports

        elif run_mode == "test_questions":
            logger.info("Running in test questions mode...")
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            ]
            questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
            forecast_reports = asyncio.run(
                unified_bot.forecast_questions(questions, return_exceptions=True)
            )

        unified_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")

    except Exception as e:
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)

