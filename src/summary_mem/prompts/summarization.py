# A plain running summary that isn't primed with the BFI questionnaire items.
SUMMARY = (
    # overall goal of the prompt
    "Your goal is to maintain a concise, factual running summary of a speaker's "
    "turns in a conversation.\n\n"

    # primes the model on what to capture
    "Identify the following information to include in the summary. Never "
    "fabricate or infer anything the turns do not support:\n"

    "1. The speaker's stable state: their preferences and opinions, personal "
    "details and relationships, plans, decisions, and goals, professional "
    "details, health and well-being, and any other concrete facts or events "
    "they reveal about themselves.\n"

    "2. Specifics such as proper nouns, exact quantities, and dates or clear "
    "temporal references.\n\n"

    # merging instructions
    "Integrate the new turns into the existing summary. Update or correct "
    "anything that has changed, and keep everything that still holds while "
    "dropping filler and small talk.\n\n"

    # keeps the rolling summary from growing without bound as more turns are folded in
    "Keep the summary under {max_sum_words} words — a strict maximum, not a target. If "
    "integrating new turns would push past it, cut the weakest or most "
    "redundant existing details rather than shortening or dropping the new turns.\n\n"

    # output format instructions
    "Respond with a single JSON object:\n"
    "{{\"summary\": <string>}}\n"
    "- \"summary\": the full updated summary text, written in the third person, with no preamble.\n"
    "Output only the JSON — no markdown formatting, no surrounding text.\n\n"

    # final instructions for the model to follow
    "Existing summary of {speaker}:\n{existing}\n\n"
    "New turns spoken by {speaker}:\n{turns}"
)


# One-shot analog of SUMMARY: summarizes a speaker's entire turn history in a
# single pass instead of folding turns into a running summary one at a time,
# so it drops the existing-summary/merge framing and reasons over the whole
# transcript at once.
BATCH_SUMMARY = (
    # overall goal of the prompt
    "Your goal is to write a concise, factual running summary of a speaker's "
    "turns in a conversation.\n\n"

    # primes the model on what to capture
    "Identify the following information to include in the summary. Never "
    "fabricate or infer anything the turns do not support:\n"

    "1. The speaker's stable state: their preferences and opinions, personal "
    "details and relationships, plans, decisions, and goals, professional "
    "details, health and well-being, and any other concrete facts or events "
    "they reveal about themselves.\n"

    "2. Specifics such as proper nouns, exact quantities, and dates or clear "
    "temporal references.\n\n"

    # since it's a single pass, dedupe and reconcile changes rather than merge
    "Drop filler and small talk. If something the speaker says changes or "
    "contradicts an earlier turn, keep only the most current version.\n\n"

    # keeps the summary from just restating every turn
    "Keep the summary under {max_sum_words} words — a strict maximum, not a target. If "
    "everything relevant would push past it, cut the weakest or most "
    "redundant details rather than the strongest ones.\n\n"

    # output format instructions
    "Respond with a single JSON object:\n"
    "{{\"summary\": <string>}}\n"
    "- \"summary\": the full summary text, written in the third person, with no preamble.\n"
    "Output only the JSON — no markdown formatting, no surrounding text.\n\n"

    # final instructions for the model to follow
    "All turns spoken by {speaker}, across every conversation they appear in:\n{turns}"
)