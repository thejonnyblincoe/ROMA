"""Shared base class for ROMA-DSPy DSPy modules."""

from __future__ import annotations

import dspy
import inspect
from typing import Union, Any, Optional, Dict, Mapping, Sequence, Mapping as TMapping, List

from src.roma_dspy.types.prediction_strategy import PredictionStrategy
from src.roma_dspy.resilience import with_module_resilience
from src.roma_dspy.settings import settings


class BaseModule(dspy.Module):
    """
    Common functionality for ROMA DSPy modules:
    - Per-instance LM configuration via dspy.context (thread/async-safe).
    - Accept an existing dspy.LM or build one from (model + config).
    - Build a predictor from a PredictionStrategy for a given signature.
    - Sync and async entrypoints (forward / aforward) with optional tools, context and per-call kwargs.
    """

    def __call__(self, *args, **kwargs):
        """Delegate to forward method for compatibility with runtime calls."""
        return self.forward(*args, **kwargs)

    def __init__(
        self,
        *,
        signature: Any,
        prediction_strategy: Union[PredictionStrategy, str] = PredictionStrategy.CHAIN_OF_THOUGHT,
        lm: Optional[dspy.LM] = None,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        context_defaults: Optional[Dict[str, Any]] = None,
        **strategy_kwargs: Any,
    ) -> None:
        super().__init__()

        self.signature = signature

        if isinstance(prediction_strategy, str):
            prediction_strategy = PredictionStrategy.from_string(prediction_strategy)

        self._tools: List[Any] = self._normalize_tools(tools)

        build_kwargs = dict(strategy_kwargs)
        if prediction_strategy in (PredictionStrategy.REACT, PredictionStrategy.CODE_ACT) and self._tools:
            build_kwargs.setdefault("tools", self._tools)

        self._predictor = prediction_strategy.build(self.signature, **build_kwargs)

        if lm is not None and model is not None:
            pass

        if lm is None:
            if model is None:
                raise ValueError(
                    "Either provide an existing lm=dspy.LM(...) or a model='provider/model' to build one."
                )
            lm_kwargs = dict(model_config or {})
            lm = dspy.LM(model, **lm_kwargs)

        self._lm: dspy.LM = lm
        self._context_defaults: Dict[str, Any] = dict(context_defaults or {})

    # ---------- Public API ----------

    def forward(
        self,
        input_task: str,
        *,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
        **call_kwargs: Any,
    ):
        """
        Args:
            input_task: The string for the signature input field ('goal').
            tools: Optional tools (dspy.Tool objects) to use for this call.
            config: Optional per-call LM overrides.
            context: Dict passed into dspy.context(...) for this call.
            call_params: Extra kwargs to pass to predictor call (strategy-specific).
            **call_kwargs: Additional kwargs merged into call_params for convenience.
        """
        runtime_tools = self._merge_tools(self._tools, tools)

        # Build context kwargs (merge defaults and per-call), ensure an LM is set
        ctx = dict(self._context_defaults)
        if context:
            ctx.update(context)
        ctx.setdefault("lm", self._lm)

        # Prepare predictor-call kwargs (merge call_params + call_kwargs)
        extra = dict(call_params or {})
        if call_kwargs:
            extra.update(call_kwargs)
        if config is not None:
            extra["config"] = config
        if runtime_tools:
            extra["tools"] = runtime_tools

        # Filter extras to what the predictor's forward accepts (avoid TypeError)
        target_method = getattr(self._predictor, "forward", None)
        filtered = self._filter_kwargs(target_method, extra)

        with dspy.context(**ctx):
            return self._execute_predictor(input_task, filtered)

    async def aforward(
        self,
        input_task: str,
        *,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
        **call_kwargs: Any,
    ):
        """
        Async version of forward(...). Uses acall(...) when available, filtering kwargs
        based on aforward(...) if present, otherwise forward(...).
        """
        runtime_tools = self._merge_tools(self._tools, tools)

        ctx = dict(self._context_defaults)
        if context:
            ctx.update(context)
        ctx.setdefault("lm", self._lm)

        extra = dict(call_params or {})
        if call_kwargs:
            extra.update(call_kwargs)
        if config is not None:
            extra["config"] = config
        if runtime_tools:
            extra["tools"] = runtime_tools

        # Choose method to derive accepted kwargs
        method_for_filter = getattr(self._predictor, "aforward", None) or getattr(self._predictor, "forward", None)
        filtered = self._filter_kwargs(method_for_filter, extra)

        with dspy.context(**ctx):
            return await self._execute_predictor_async(input_task, filtered, method_for_filter)
    
    def get_model_config(self, *, redact_secrets: bool = True) -> Dict[str, Any]:
        """
        Return LM configuration from the underlying BaseLM/dspy.LM instance.
        Fields: model, model_type, cache, kwargs (e.g., temperature, max_tokens, provider-specific args).
        """
        lm = self._lm
        cfg: Dict[str, Any] = {}

        model = getattr(lm, "model", None)
        if model is not None:
            cfg["model"] = model

        model_type = getattr(lm, "model_type", None)
        if model_type is not None:
            cfg["model_type"] = model_type

        cache = getattr(lm, "cache", None)
        if cache is not None:
            cfg["cache"] = cache

        kwargs = getattr(lm, "kwargs", None)
        if isinstance(kwargs, dict):
            safe_kwargs = dict(kwargs)
            if redact_secrets:
                for k in list(safe_kwargs.keys()):
                    if any(s in k.lower() for s in ("key", "token", "secret", "password")):
                        safe_kwargs[k] = "****"
            cfg["kwargs"] = safe_kwargs
        else:
            cfg["kwargs"] = {}

        return cfg

    # ---------- Conveniences ----------

    @property
    def lm(self) -> dspy.LM:
        return self._lm

    def replace_lm(self, lm: dspy.LM) -> "BaseModule":
        self._lm = lm
        return self

    @property
    def tools(self) -> List[Any]:
        return list(self._tools)

    def set_tools(self, tools: Optional[Union[Sequence[Any], TMapping[str, Any]]]) -> "BaseModule":
        self._tools = self._normalize_tools(tools)
        return self

    def add_tools(self, *tools: Any) -> "BaseModule":
        for t in tools:
            if not any(t is existing for existing in self._tools):
                self._tools.append(t)
        return self

    def clear_tools(self) -> "BaseModule":
        self._tools.clear()
        return self

    # ---------- Internals ----------

    @staticmethod
    def _normalize_tools(tools: Optional[Union[Sequence[Any], TMapping[str, Any]]]) -> List[Any]:
        if tools is None:
            return []
        if isinstance(tools, dict):
            return list(tools.values())
        if isinstance(tools, (list, tuple)):
            return list(tools)
        raise TypeError("tools must be a sequence of dspy.Tool or a mapping name->dspy.Tool")

    @staticmethod
    def _merge_tools(default_tools: List[Any], runtime_tools: Optional[Union[Sequence[Any], TMapping[str, Any]]]) -> List[Any]:
        if runtime_tools is None:
            return list(default_tools)
        merged = list(default_tools)
        to_add = BaseModule._normalize_tools(runtime_tools)
        for t in to_add:
            if not any(t is existing for existing in merged):
                merged.append(t)
        return merged

    @staticmethod
    def _get_allowed_kwargs(func: Optional[Any]) -> Optional[set]:
        if func is None:
            return None
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return None
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_kw:
            return None  # accepts any kwargs
        allowed = set(
            name
            for name, p in sig.parameters.items()
            if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        return allowed

    @staticmethod
    def _filter_kwargs(func: Optional[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        allowed = BaseModule._get_allowed_kwargs(func)
        if allowed is None:
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in allowed}

    # ---------- Resilient Predictor Execution ----------

    @with_module_resilience(module_name="base_predictor")
    def _execute_predictor(self, input_task: str, filtered: Dict[str, Any]):
        """Execute predictor with resilience protection."""
        return self._predictor(goal=input_task, **filtered)

    @with_module_resilience(module_name="base_predictor")
    async def _execute_predictor_async(self, input_task: str, filtered: Dict[str, Any], method_for_filter: Optional[Any]):
        """Execute predictor asynchronously with resilience protection."""
        acall = getattr(self._predictor, "acall", None)
        if acall is not None and method_for_filter is not None and hasattr(self._predictor, "aforward"):
            return await self._predictor.acall(goal=input_task, **filtered)
        # Fallback to sync if async not available
        return self._predictor(goal=input_task, **filtered)

    @classmethod
    def with_settings_resilience(
        cls,
        *,
        signature: Any,
        prediction_strategy: Union[PredictionStrategy, str] = PredictionStrategy.CHAIN_OF_THOUGHT,
        **kwargs
    ):
        """Create BaseModule with resilience configuration from global settings."""
        # Settings are now available through the resilience decorators
        # The decorators will use the global circuit breaker and retry policies
        return cls(
            signature=signature,
            prediction_strategy=prediction_strategy,
            **kwargs
        )
