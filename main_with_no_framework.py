import requests
import logging
import asyncio
import random
from typing import List, Dict, Any

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ForecastBot")

BASE_URL = "https://www.metaculus.com/api2/questions/"

# -----------------------------
# Fetcher
# -----------------------------
def list_posts_from_tournament(
    tournament_id: int | str,
    count: int = 50,
) -> list[dict]:
    """
    Fetch all questions from a given Metaculus tournament (with pagination).
    """
    questions = []
    url = f"{BASE_URL}?limit={count}&tournament={tournament_id}&include_description=true"
    url += "&forecast_type=binary,multiple_choice,numeric,discrete"

    page = 1
    while url:
        logger.info(f"Fetching page {page} for tournament {tournament_id}")
        resp = requests.get(url)

        if resp.status_code != 200:
            logger.error(f"Failed to fetch page {page}: {resp.status_code} {resp.text}")
            break

        data = resp.json()
        results = data.get("results", [])
        logger.info(f"Retrieved {len(results)} questions from page {page}")

        questions.extend(results)
        url = data.get("next")  # next page URL or None
        page += 1

    logger.info(f"Total retrieved for tournament {tournament_id}: {len(questions)} questions")
    return questions


def fetch_filtered_questions(tournament_id: int | str) -> list[dict]:
    """
    Retrieve and lightly filter questions from a tournament.
    """
    all_qs = list_posts_from_tournament(tournament_id)
    logger.info(f"Raw retrieved: {len(all_qs)} questions from {tournament_id}")

    filtered_qs = [q for q in all_qs if not q.get("is_hidden", False)]
    logger.info(f"After filter: {len(filtered_qs)} questions remain for {tournament_id}")
    return filtered_qs

# -----------------------------
# Forecasting logic
# -----------------------------
async def forecast_binary(question: dict) -> dict:
    """Generate a forecast for a binary question."""
    prob = random.uniform(0.1, 0.9)  # Replace with model output
    logger.debug(f"Binary forecast Q{question['id']}: {prob:.2f}")
    return {"type": "binary", "probability_yes": prob}


async def forecast_numeric(question: dict) -> dict:
    """Generate a forecast for a numeric question."""
    low = random.uniform(0, 50)
    high = low + random.uniform(10, 100)
    logger.debug(f"Numeric forecast Q{question['id']}: {low:.1f}–{high:.1f}")
    return {"type": "numeric", "low": low, "high": high}


async def forecast_multiple_choice(question: dict) -> dict:
    """Generate a forecast for a multiple-choice question."""
    options = question.get("options", [{"id": i} for i in range(4)])
    weights = [random.random() for _ in options]
    total = sum(weights)
    probs = [w / total for w in weights]
    forecast = {opt["id"]: p for opt, p in zip(options, probs)}
    logger.debug(f"MC forecast Q{question['id']}: {forecast}")
    return {"type": "multiple_choice", "probabilities": forecast}


# -----------------------------
# Forecasting loop
# -----------------------------
async def forecast_individual_question(question: dict, submit_prediction: bool = False) -> dict:
    """
    Forecast a single question depending on its type.
    """
    try:
        qtype = question.get("type", "binary")
        if qtype == "binary":
            result = await forecast_binary(question)
        elif qtype in ("numeric", "discrete"):
            result = await forecast_numeric(question)
        elif qtype == "multiple_choice":
            result = await forecast_multiple_choice(question)
        else:
            raise ValueError(f"Unsupported question type: {qtype}")

        if submit_prediction:
            # TODO: Add API submission logic
            logger.debug(f"Submitting forecast for Q{question['id']}")

        return {"id": question["id"], "forecast": result}

    except Exception as e:
        logger.warning(f"Failed forecasting Q{question.get('id')}: {e}")
        return {"id": question.get("id"), "error": str(e)}


async def forecast_questions(tournament_id: int | str, submit_prediction: bool = False) -> None:
    """
    Forecast all questions in a tournament.
    """
    questions = fetch_filtered_questions(tournament_id)
    logger.info(f"Starting forecasting loop for {len(questions)} questions in {tournament_id}")

    forecasted = 0
    skipped = 0
    results = []

    for q in questions:
        res = await forecast_individual_question(q, submit_prediction=submit_prediction)
        if "error" in res:
            skipped += 1
        else:
            forecasted += 1
            results.append(res)

    logger.info(
        f"Tournament {tournament_id}: Forecasted {forecasted}, "
        f"Skipped {skipped}, Total {len(questions)}"
    )

    # (Optional) Save to file for debugging
    with open(f"forecast_results_{tournament_id}.json", "w") as f:
        import json
        json.dump(results, f, indent=2)


# -----------------------------
# Main entrypoint
# -----------------------------
async def main():
    tournaments = ["32813", "fall-aib-2025","minibench"]  # Example tournaments
    for tid in tournaments:
        await forecast_questions(tid, submit_prediction=False)


if __name__ == "__main__":
    asyncio.run(main())
