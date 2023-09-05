from cntell.common.cms import CacheManagementSystem
from cntell.common.component import Input, Output, Component


class NoCMS(CacheManagementSystem):
    def can_skip(self, input: Input) -> bool:
        return False
    
    def save_output(self, input: Input, output: Output) -> None:
        raise NotImplementedError()
    
    
    def prepare(self) -> None:
        pass
    
    
    def suspend(self) -> None:
        pass
    
    
    def get_output(self, input: Input) -> Output:
        return None
    
    
    def run(self, component: Component, inputs: Input) -> Output:
        return component.run(inputs)
    