from abc import ABC, abstractmethod
from typing import TypeVar, Type, Generic
from cntell.common.component import Input, Output, Component


class CacheManagementSystem(ABC):
    @abstractmethod
    def can_skip(self, input: Input) -> bool:
        pass
    
    
    @abstractmethod
    def get_output(self, input: Input) -> Output:
        pass
    
    
    @abstractmethod
    def save_output(self, input: Input, output: Output) -> None:
        pass
    
    
    @abstractmethod
    def prepare(self) -> None:
        pass
    
    
    @abstractmethod
    def suspend(self) -> None:
        pass
        
    
    def run(self, component: Component, inputs: Input) -> Output:
        if self.can_skip(inputs):
            if isinstance(component, CacheHitCallback):
                return component.on_hit(inputs, self.get_output(inputs))
            else:
                return self.get_output(inputs)
        else:
            inputs, output = component.run(inputs, return_inputs=True)
            if isinstance(component, CacheMissCallback):
                return component.on_miss(inputs, output)
            else:
                self.save_output(inputs, output)
                return output
        
        
        
class CacheHitCallback:
    def on_hit(self, input: Input, output: Output) -> Output:
        pass
    
    
class CacheMissCallback:
    def on_miss(self, input: Input, output: Output) -> Output:
        pass
    
        
        