from meow_embed.server import _ensure_xlm_roberta_create_position_ids_compat


def test_xlm_roberta_create_position_ids_compat() -> None:
    _ensure_xlm_roberta_create_position_ids_compat()
    from transformers.models.xlm_roberta.modeling_xlm_roberta import (
        create_position_ids_from_input_ids,
    )

    assert callable(create_position_ids_from_input_ids)
