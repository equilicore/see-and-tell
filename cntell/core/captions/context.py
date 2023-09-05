import os
from cntell.core.captions.captions_improved import generate_captions
from cntell.core.captions.noun_phrases_detection import NLP_SingletonInitializer
from cntell.common.component import Component, Input, Output
from typing import Generic, TypeVar


ContextData = TypeVar("ContextData")


class Context(Component, Generic[ContextData]):
    """Incorporates more context into captions."""

    class ContextInput(
        Input,
    ):
        """The input to the Context component."""

        captions: list[str]
        context: ContextData

        class Config:
            arbitrary_types_allowed = True

    class ContextOutput(Output):
        """The output of the Context component."""

        captions: list[str]
        indices: list[int]

    input_model = ContextInput
    output_model = ContextOutput

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def prepare(
        self, use_dir: os.PathLike = None, use_gpu: bool = False, *args, **kwargs
    ) -> None:
        super().prepare(use_dir, use_gpu, *args, **kwargs)
        NLP_SingletonInitializer()

    def _run(self, inputs: ContextInput) -> ContextOutput:
        """Incorporates more context into captions.

        Args:
            captions (list[str]): The captions to incorporate context into.
            context (ContextData): The context to incorporate.

        Returns:
            ContextOutput: The captions with context incorporated.
        """
        captions, idxs = generate_captions(inputs.captions, inputs.context)
        return self.ContextOutput(captions=captions, indices=idxs)

    def run(self, __input: input_model = None, **kwargs) -> output_model:
        return super().run(__input, **kwargs)
