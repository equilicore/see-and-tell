import hashlib
import os
import time
import uuid
import pydantic
import pickle

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Generic, TypeVar
from cntell.common.functions import fieldsof


InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')

class Base(ABC, pydantic.BaseModel):
    id: str =  pydantic.Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = pydantic.Field(default_factory=lambda: time.time())
    run_id: Optional[str] = None
    
    
    def to_bytes(self) -> bytes:
        """Returns the input as bytes.
        
        Returns:
            bytes: The input as bytes.
        """
        return pickle.dumps(self)
    
    
    def checksum(self) -> str:
        """Returns the checksum of the input.
        
        Returns:
            str: The checksum of the input.
        """
        _bytes = self.model_dump(exclude=['id', 'timestamp', 'run_id'])
        _bytes = pickle.dumps(_bytes)
        return hashlib.sha256(_bytes).hexdigest()
    
    
    def save(self, dir: os.PathLike) -> None:
        """Saves the input to a directory.
        
        Args:
            dir (os.PathLike): The directory to save the input to.
        """
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, self.id), 'wb') as f:
            f.write(self.to_bytes())
            
    
    @staticmethod
    def load(dir: os.PathLike, id: str) -> None:
        """Loads the input from a directory.
        
        Args:
            dir (os.PathLike): The directory to load the input from.
        """
        with open(os.path.join(dir, id), 'rb') as f:
           return pickle.loads(f.read())


class Input(Base):
    pass

        
class Output(Base):
    pass
            
    
class Component(ABC, Generic[InputType, OutputType]):
    name: Optional[str] = None
    version: Optional[str] = None
    input_model: Type[Input] = None
    output_model: Type[Output] = None
    
    _bounded_run_id: str = None
    
    
    def bound_run(self, run_id: str) -> None:
        self._bounded_run_id = run_id
    
    
    def log(self, message: str) -> None:
        """Logs a message.
        
        Args:
            message (str): The message to log.
        """
        pass 
    
    
    def get_use_dir(self) -> os.PathLike:
        """Returns the directory to use for storing files.
        
        Returns:
            os.PathLike: The directory to use for storing files.
        """
        base = None
        if self.name is None:
            base = self.__class__.__name__
        else:
            base = self.name
        
        return os.path.join(os.getcwd(), ".pipeline", base)
    

    def run(self, __input: input_model=None, return_inputs=False, **kwargs) -> output_model:
        """Runs the component.
        
        This method should not be overridden by subclasses. It calls the
        _run method and applies the input and output models.
        
        Args:
            inputs (input_model): The input to the component.
            
        Returns:
            output_model: The output of the component.
        """
        if self.input_model is not None:
            try:
                if __input is None:
                    inputs =  self.input_model(run_id=self._bounded_run_id, **kwargs)
                else:
                    inputs = __input
                    inputs.run_id = self._bounded_run_id
                
            except Exception as e:
                raise ValueError(
                    "Inputs of the component should follow the input model. "
                    f"Please, provide following fields {fieldsof(self.input_model)}") from e
        else:
            if __input is None:
                inputs = kwargs + { 'run_id': self._bounded_run_id } 
            else:
                inputs = __input
                inputs.run_id = self._bounded_run_id
                            
        outputs = self._run(inputs)
                
        if self.output_model is not None:
            try:
                if isinstance(outputs, dict):
                    outputs = self.output_model(**outputs)
                elif isinstance(outputs, pydantic.BaseModel):
                    outputs = self.output_model(**outputs.model_dump())
                else:
                    raise ValueError(
                        f"Could not validate an output of the component {self.name}. "
                        f"Expected a dict or pydantic.BaseModel, got {type(outputs)}") 
                     
            except Exception as e: 
                raise ValueError(
                    f"Outputs of the component {self.name} should follow the output model. "
                    "Please, provide a dict with"
                    f" following fields {fieldsof(self.output_model)}") from e
                
        else:
            outputs = outputs or {'run_id': self._bounded_run_id}
        
        outputs.run_id = self._bounded_run_id
                
        if return_inputs:
            return inputs, outputs
        return outputs
    
        
    def prepare(self, use_dir: os.PathLike=None, use_gpu: bool = False, *args, **kwargs) -> None:
        """Prepares a component for use.
        
        Does any necessary setup for the component. 
        Downloads models, creates directories, initializes variables, etc.
        
        Args:
            use_dir (os.PathLike, optional): 
                In case the component needs to use filesystem to download/store
                files, this is the directory to use. In case of None, the component
                should use a directory obtained by
            use_gpu (bool, optional): Whether to use GPU. Defaults to False.
        """
        pass
    
    
    @abstractmethod
    def _run(self, input: input_model | dict) -> output_model | dict:
        """Runs the component.
        
        This method should be implemented by subclasses. It should do the actual
        work of the component. 
        
        Args:
            inputs (Any): The input to the component.
        
        Returns:
            Any: The output of the component.
        """
        pass
            
            
    
                          
