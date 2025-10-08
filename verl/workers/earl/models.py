import torch
import torch.nn as nn


def create_earl_head(base_model, tool_config, model_config, dtype):
    feature_size = base_model.lm_head.weight.shape[1]
    # get expanded action space size
    expansion_size =tool_config.total_size
    assert expansion_size > 0, "Expanded action space size must be greater than 0"
    if model_config.get('type', 'nn.Linear') == 'nn.Linear':
        # Use nn.Linear for the earl head
        earl_head = nn.Linear(
            in_features=feature_size,
            out_features=expansion_size,
            bias=model_config.get('earl_head_use_bias', False)
        ).to(base_model.lm_head.weight.device, dtype=dtype)
        if model_config.get('init_from_base', True):
            init_weights = torch.zeros_like(earl_head.weight.data)
            for i in range(expansion_size):
                id = tool_config.org_vocab_size + i
                original_ids = tool_config.id_to_seq[id]
                init_weight_with = tool_config.id_to_init_weight[id]
                # breakpoint()
                with torch.no_grad():
                    # init_weights[i].copy_(
                    #     torch.mean(base_model.lm_head.weight[original_ids], dim=0).detach()
                    # )
                    # if id not in tool_config.tool_action_ids:
                    init_weights[i].copy_(
                        base_model.lm_head.weight[original_ids[init_weight_with]].detach()
                    )
            with torch.no_grad():
                earl_head.weight.copy_(init_weights)
    else:
        raise NotImplementedError(
            f"Unsupported earl head type: {model_config.get('type', 'nn.Linear')}"
        )
    return earl_head