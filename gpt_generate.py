import torch
import torch.nn.functional as F
import gpt_generate_tests as tests


@torch.no_grad()
def greedy_search(
    model,  # GPT2LMHeadModel
    input_ids,  # Tensor(batch_size, seq_len)
    max_new_tokens: int,
):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
    return input_ids


# TODO: ngram penalty
@torch.no_grad()
def beam_search(
    model,  # GPT2LMHeadModel
    input_ids,  # Tensor(1, seq_len)
    max_new_tokens: int,
    num_beans: int,
):
    candidates = [(0, input_ids)]  # [(log prob of this sequence, input_ids so far)]
    for _ in range(max_new_tokens):
        next_candidates = []
        for logp, inp_ids in candidates:
            logits = model(inp_ids).logits
            next_token_logits = logits[:, -1, :]
            next_token_logprobs = next_token_logits.log_softmax(dim=-1)
            topk_logprobs, topk_indices = torch.topk(
                next_token_logprobs, k=num_beans, dim=-1
            )
            for logprob, idx in zip(topk_logprobs.squeeze(0), topk_indices.squeeze(0)):
                next_logprob = logp + logprob.item()
                next_ids = torch.cat(
                    (inp_ids, torch.tensor([idx]).unsqueeze(0)), dim=-1
                )
                next_candidates.append((next_logprob, next_ids))
        # prune: only keep the best num_beans candidates
        next_candidates.sort(key=lambda x: -x[0])
        candidates = next_candidates[:num_beans]

    _, out_ids = candidates[0]
    return out_ids


@torch.no_grad()
def sample(
    model,  # GPT2LMHeadModel
    input_ids,  # Tensor(batch_size, seq_len)
    max_new_tokens: int,
):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(next_token_probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
    return input_ids


@torch.no_grad()
def sample_temperature(
    model,  # GPT2LMHeadModel
    input_ids,  # Tensor(batch_size, seq_len)
    max_new_tokens: int,
    temperature: float,
):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :] / temperature
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(next_token_probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
    return input_ids


def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = -float("Inf")
    return logits


@torch.no_grad()
def sample_top_k(
    model,  # GPT2LMHeadModel
    input_ids,  # Tensor(batch_size, seq_len)
    max_new_tokens: int,
    top_k: int,
):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :]
        filtered_next_token_logits = top_k_logits(next_token_logits, top_k)
        filtered_next_token_probs = F.softmax(filtered_next_token_logits, dim=-1)
        next_tokens = torch.multinomial(filtered_next_token_probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
    return input_ids


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs >= p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits[indices_to_remove] = -float("inf")
    return logits


@torch.no_grad()
def sample_top_p(
    model,  # GPT2LMHeadModel
    input_ids,  # Tensor(batch_size, seq_len)
    max_new_tokens: int,
    top_p: float,
):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :]
        filtered_next_token_logits = top_p_logits(next_token_logits, top_p)
        filtered_next_token_probs = F.softmax(filtered_next_token_logits, dim=-1)
        next_tokens = torch.multinomial(filtered_next_token_probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
    return input_ids


if __name__ == "__main__":
    tests.test_greedy_search(greedy_search)
    tests.test_beam_search(beam_search)
    tests.test_sample(sample)
    tests.test_sample_temperature(sample_temperature)
    tests.test_sample_top_k(sample_top_k)
    tests.test_sample_top_p(sample_top_p)
