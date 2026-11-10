"""Shared output measurements for the final-answer comparison."""

ANSWER_PREFIX = "\n\nFinal Answer:"


def thinking_end_id(tokenizer):
    tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError("answer forcing requires a single </think> token")
    return tokens[0]


def transition_metrics(tokenizer, output, method):
    tokens = output.get("token_ids")
    if tokens is None:
        raise ValueError("answer forcing requires completion token_ids")
    end_id = thinking_end_id(tokenizer)
    if end_id not in tokens:
        return {"method": method, "transition": "no_reasoning_end", "has_answer": False}
    content_tokens = tokens[tokens.index(end_id) + 1:]
    content = tokenizer.decode(content_tokens, skip_special_tokens=True)
    if not content.startswith(ANSWER_PREFIX):
        return {"method": method, "transition": "marker_missing", "has_answer": False}
    return {"method": method, "transition": "marker_written",
            "has_answer": bool(content[len(ANSWER_PREFIX):].strip())}
