"""Frozen linear projections augmented by a routed LoRA expert bank."""

from torch import Tensor, nn

from molor.config import LoRABankConfig
from molor.lora_bank import LoRAExpertBank
from molor.manifests import AdapterBankManifest
from molor.router import RouterOutput


class MixtureLoRALinear(nn.Module):
    """Apply a frozen base projection plus a weighted sum of LoRA deltas."""

    def __init__(
        self,
        base: nn.Linear,
        config: LoRABankConfig,
        *,
        bank_manifest: AdapterBankManifest | None = None,
        module_name: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")

        self.base = base
        self.base.requires_grad_(False)
        self.expert_bank = LoRAExpertBank(
            base.in_features,
            base.out_features,
            config,
            device=base.weight.device,
            dtype=base.weight.dtype,
            bank_manifest=bank_manifest,
            module_name=module_name,
        )

    def forward(
        self,
        inputs: Tensor,
        routing: Tensor | RouterOutput,
        *,
        bank_fingerprint: str | None = None,
    ) -> Tensor:
        return self.base(inputs) + self.expert_bank(
            inputs,
            routing,
            bank_fingerprint=bank_fingerprint,
        )


class FactorizedMixtureLoRALinear(nn.Module):
    """Apply capability and memory LoRA deltas on independent routing axes.

    The memory bank contains exactly one version-pinned adapter plus its null route.  Its
    weights are never concatenated with or normalized against the capability route.
    """

    def __init__(
        self,
        base: nn.Linear,
        capability_config: LoRABankConfig,
        memory_config: LoRABankConfig,
        *,
        capability_manifest: AdapterBankManifest | None = None,
        memory_manifest: AdapterBankManifest | None = None,
        module_name: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if memory_config.num_experts != 1 or not memory_config.include_null_expert:
            raise ValueError("memory_config must declare one adapter and an explicit null route")
        if (capability_manifest is None) != (memory_manifest is None):
            raise ValueError("capability and memory manifests must be bound together")
        if capability_manifest is not None and memory_manifest is not None:
            if capability_manifest.base.fingerprint != memory_manifest.base.fingerprint:
                raise ValueError("capability and memory adapters must target the same frozen base")
            if len(memory_manifest.adapters) != 1 or not memory_manifest.include_null:
                raise ValueError("memory manifest must contain one adapter and a null route")

        self.base = base
        self.base.requires_grad_(False)
        self.capability_bank = LoRAExpertBank(
            base.in_features,
            base.out_features,
            capability_config,
            device=base.weight.device,
            dtype=base.weight.dtype,
            bank_manifest=capability_manifest,
            module_name=module_name,
        )
        self.memory_bank = LoRAExpertBank(
            base.in_features,
            base.out_features,
            memory_config,
            device=base.weight.device,
            dtype=base.weight.dtype,
            bank_manifest=memory_manifest,
            module_name=module_name,
        )

    def forward(
        self,
        inputs: Tensor,
        capability_routing: Tensor | RouterOutput,
        memory_routing: Tensor | RouterOutput,
        *,
        capability_bank_fingerprint: str | None = None,
        memory_bank_fingerprint: str | None = None,
    ) -> Tensor:
        """Sum both complete LoRA deltas without cross-axis normalization."""

        return (
            self.base(inputs)
            + self.capability_bank(
                inputs,
                capability_routing,
                bank_fingerprint=capability_bank_fingerprint,
            )
            + self.memory_bank(
                inputs,
                memory_routing,
                bank_fingerprint=memory_bank_fingerprint,
            )
        )
