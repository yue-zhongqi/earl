import torch
from verl.utils.torch_functional import logprobs_from_logits
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor, logits_mask: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    logits_zero_masked = logits.clone().masked_fill_(~logits_mask, 0.0)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits_zero_masked, dim=-1)
    return entropy

def entropy_from_logits_with_chunking(logits: torch.Tensor, logits_mask: torch.Tensor, chunk_size: int = 2048):
    """Memory-efficient entropy calculation with chunking."""
    entropy = torch.zeros(logits.shape[0], device=logits.device)
    for i in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[i : i + chunk_size].float()
        logits_chunk_zero_masked = logits_chunk.clone().masked_fill_(~logits_mask[i : i + chunk_size], 0.0)
        pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk_zero_masked, dim=-1)
        entropy[i : i + chunk_size] = entropy_chunk
    return entropy


def compute_log_probs(logits, labels, inplace_backward, tool_mask, org_vocab_size, tool_size):
    tool_logits = logits[tool_mask, -tool_size:]
    tool_labels = labels[tool_mask] - org_vocab_size
    org_logits = logits[~tool_mask]
    org_labels = labels[~tool_mask]
    tool_log_probs = logprobs_from_logits(
        logits=tool_logits,
        labels=tool_labels,
        inplace_backward=inplace_backward,
    )
    org_log_probs = logprobs_from_logits(
        logits=org_logits,
        labels=org_labels,
        inplace_backward=inplace_backward,
    )
    log_probs = torch.empty_like(labels, dtype=tool_log_probs.dtype)
    log_probs[tool_mask] = tool_log_probs
    log_probs[~tool_mask] = org_log_probs
    return log_probs

def top_k_top_p_sample(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        
        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    
    pred_token = torch.multinomial(F.softmax(logits, -1), 1) # [BATCH_SIZE, 1]
    return pred_token