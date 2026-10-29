import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()                    # connects to the configured providers
config_builder = dd.DataDesignerConfigBuilder()   # the recipe

# SAMPLER: picks a random language for each row (no LLM call)
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="language",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["English", "Spanish", "French", "German", "Italian"],
        ),
    )
)

# LLM TEXT: {{ language }} is filled in for each row
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="greeting",
        model_alias="nvidia-text",
        prompt="Write a casual and formal greeting in {{ language }}.",
    )
)

results = data_designer.preview(config_builder)   # 10 rows, in memory only
results.display_sample_record()
