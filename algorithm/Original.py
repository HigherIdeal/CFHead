import time
from dataclasses import dataclass

import torch


@dataclass
class NextTokenResult:
    token_ids: torch.Tensor
    logits: torch.Tensor
    lm_head_time_ms: float


class OriginalHeadSelector:
    """Original BF16 LM head next-token selector."""

    def __init__(
        self,
        lm_head: torch.nn.Module,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.lm_head = lm_head
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def select(self, last_hidden: torch.Tensor) -> NextTokenResult:
        if last_hidden.dim() == 1:
            last_hidden = last_hidden.unsqueeze(0)

        if last_hidden.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = self.lm_head(last_hidden)
        token_ids = self._sample(logits) if self.do_sample else torch.argmax(logits, dim=-1)
        if last_hidden.is_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return NextTokenResult(token_ids=token_ids, logits=logits, lm_head_time_ms=elapsed_ms)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = max(self.temperature, 1e-6)
        probs = torch.softmax(logits / temperature, dim=-1)

        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative_probs > self.top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(dim=-1, index=sampled_sorted).squeeze(-1)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)
