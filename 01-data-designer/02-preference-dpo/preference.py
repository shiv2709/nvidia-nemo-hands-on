import data_designer.config as dd

# MODELS: my own aliases, so models are easy to swap if one goes offline
MODELS = [
    dd.ModelConfig(
        alias="lightning",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=1.0, top_p=0.95, max_parallel_requests=4,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    ),
    dd.ModelConfig(
        alias="super",
        model="nvidia/nemotron-3.5-super-vl-120b-a12b",   # replaces nemotron-3-super (endpoint returned 404)
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=1.0, top_p=0.95, max_parallel_requests=4,
        ),
    ),
]

# THE RUBRIC: the judge only enforces rules written here
preference_score = dd.Score(
    name="Preference",
    description=(
        "Which response better answers the question for the given audience. "
        "Judge on: (1) factual accuracy, (2) clarity for the audience, "
        "(3) staying under 120 words - a response over 120 words breaks the "
        "instructions and should lose unless the other is factually wrong. "
        "Do not prefer a response just because it is longer."
    ),
    options={
        "A": "Response A is clearly better.",
        "B": "Response B is clearly better.",
        "Tie": "Both responses are about equally good or equally flawed.",
    },
)

def load_config_builder(
    params: dd.DataDesignerScriptParams | None = None,
) -> dd.DataDesignerConfigBuilder:
    b = dd.DataDesignerConfigBuilder(model_configs=MODELS)

    b.add_column(dd.SamplerColumnConfig(
        name="topic",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=[
            "overfitting", "gradient descent", "attention in transformers",
            "mixture of experts", "LoRA fine-tuning", "tokenization",
            "embeddings", "RLHF", "quantization", "retrieval-augmented generation",
        ]),
    ))
    b.add_column(dd.SamplerColumnConfig(
        name="audience",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=[
            "high school student", "non-technical manager",
            "undergraduate CS student", "experienced software engineer",
        ]),
    ))

    b.add_column(dd.LLMTextColumnConfig(
        name="question",
        model_alias="lightning",
        prompt=(
            "Write one short question a {{ audience }} might ask about {{ topic }}. "
            "Return only the question."
        ),
    ))

    answer_prompt = "Answer this question for a {{ audience }} in under 120 words.\n\nQuestion: {{ question }}"
    b.add_column(dd.LLMTextColumnConfig(
        name="response_a", model_alias="lightning", prompt=answer_prompt,
    ))
    b.add_column(dd.LLMTextColumnConfig(
        name="response_b", model_alias="super", prompt=answer_prompt,
    ))

    b.add_column(dd.LLMJudgeColumnConfig(
        name="pref_judge",
        model_alias="super",
        prompt=(
            "Compare two answers to the same question.\n"
            "Audience: {{ audience }}\n"
            "Question: {{ question }}\n\n"
            "Response A:\n{{ response_a }}\n\n"
            "Response B:\n{{ response_b }}\n\n"
            "Judge which response is better for this audience."
        ),
        scores=[preference_score],
    ))

    b.add_column(dd.ExpressionColumnConfig(
        name="winner", expr="{{ pref_judge.Preference.score }}",
    ))
    b.add_column(dd.ExpressionColumnConfig(
        name="chosen",
        expr="{% if winner == 'B' %}{{ response_b }}{% else %}{{ response_a }}{% endif %}",
    ))
    b.add_column(dd.ExpressionColumnConfig(
        name="rejected",
        expr="{% if winner == 'B' %}{{ response_a }}{% else %}{{ response_b }}{% endif %}",
    ))

    return b
