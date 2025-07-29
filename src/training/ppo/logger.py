import sys

from loguru import logger

PALETTE = {
    "ppo_evaluator": "green",
    "ppo_env": "blue",
}

LEVEL_PER_COMPONENT = {
    "ppo_env": "WARNING",
}


def component_filter(record):
    comp = record["extra"].get("component", "")
    min_level = logger.level(LEVEL_PER_COMPONENT.get(comp, "DEBUG")).no
    return record["level"].no >= min_level


def formatter(record):
    comp = record["extra"].get("component", "")
    id = record["extra"].get("id", "")
    colour = PALETTE.get(comp, "white")

    # The tag lives in the *template* that the sink receives,
    # so Loguru will translate it to ANSI codes.
    if id:
        return (
            "{time:HH:mm:ss} | "
            f"<{colour}>{comp:<15} | {id:<15}</> | "
            "<level>{message}</level>\n"
        )
    else:
        return (
            "{time:HH:mm:ss} | "
            f"<{colour}>{comp:<15}</> | "
            "<level>{message}</level>\n"
        )


logger.remove()
logger.add(sys.stderr, format=formatter, filter=component_filter, colorize=True)
