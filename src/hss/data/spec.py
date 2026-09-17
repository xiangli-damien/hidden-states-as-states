"""Source-independent selection contract for hidden-state matrices."""

from dataclasses import dataclass


@dataclass
class DataSpec:
    paths: list[str]
    source_format: str = "openact"
    dataset_id: str | None = None
    representation: str = "mean"
    final_norm: str = "post"
    layers: list[int] | None = None
    label: str = "is_correct"
    label_file: str = "correctness"
    expected_samples: int | None = None
    max_shards: int | None = None
    max_samples: int | None = None
    batch_tokens: int = 128
    only_valid: bool = True
    exclude_truncated: bool = False
    require_copy_receipt: bool = True
    min_free_gib: float = 64
    token_entropy_path: str | None = None
    token_logprob_path: str | None = None
    expected_model: str | None = None

    def validate(self):
        if self.source_format not in ("openact", "arrays"):
            raise ValueError("source_format must be openact or arrays")
        if not self.paths or self.representation not in (
            "mean",
            "prompt_last",
            "prefix",
            "sentence",
            "tokens",
        ):
            raise ValueError(
                "Data requires paths and representation=mean/prompt_last/prefix/sentence/tokens"
            )
        if self.final_norm not in ("pre", "post"):
            raise ValueError("final_norm must be pre or post")
        for value in (self.expected_samples, self.max_shards, self.max_samples):
            if value is not None and value < 1:
                raise ValueError("Data limits must be positive")
        if self.batch_tokens < 1 or self.min_free_gib < 0:
            raise ValueError("Invalid token batch or disk reserve")
