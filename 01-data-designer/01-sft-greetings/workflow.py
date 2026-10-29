import data_designer.config as dd

def load_config_builder(
    params: dd.DataDesignerScriptParams | None = None,
) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder()

    # SAMPLERS: controlled variety, no LLM calls
    builder.add_column(dd.SamplerColumnConfig(
        name="language",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["English", "Spanish", "French", "German", "Italian", "Hindi", "Kannada"],
        ),
    ))
    builder.add_column(dd.SamplerColumnConfig(
        name="formality",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["casual", "formal"]),
    ))
    builder.add_column(dd.SamplerColumnConfig(
        name="situation",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=[
            "meeting a new colleague", "starting an email", "greeting a teacher",
            "answering the phone", "texting a friend", "welcoming a guest",
        ]),
    ))

    # LLM COLUMN 1: strict prompt for clean output
    builder.add_column(dd.LLMTextColumnConfig(
        name="greeting",
        model_alias="nvidia-text",
        prompt=(
            "Write one {{ formality }} greeting in {{ language }} for this situation: {{ situation }}. "
            "Return only the greeting itself, with no explanation, labels, quotes, or translation."
        ),
    ))

    # LLM COLUMN 2: chained, uses the greeting from column 1
    builder.add_column(dd.LLMTextColumnConfig(
        name="english_translation",
        model_alias="nvidia-text",
        prompt=(
            "Translate this {{ language }} greeting into English. "
            "Return only the translation.\n\n{{ greeting }}"
        ),
    ))

    return builder
