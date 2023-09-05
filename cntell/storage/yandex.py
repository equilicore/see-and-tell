import os
import uuid
import boto3
import pickle
from typing import Any

class YandexS3:
    def __init__(self, *, service_key_id=None, service_key=None, bucket_name=None):
        if service_key is None:
            service_key = os.environ.get("YANDEX_SERVICE_KEY", None)
            
        if bucket_name is None:
            bucket_name = os.environ.get("YANDEX_BUCKET_NAME", None)
            
        if service_key_id is None:
            service_key_id = os.environ.get("YANDEX_SERVICE_KEY_ID", None)
            
        if service_key is None or bucket_name is None:
            raise ValueError("S3 initialization error: no service key or bucket name provided")
        
        self._client = boto3.client(
            "s3",
            endpoint_url="https://storage.yandexcloud.net",
            aws_access_key_id=service_key_id,
            aws_secret_access_key=service_key,
        )
        
        self._bucket_name = bucket_name
        self._session = None
        
    
    def new_session(self, session_id=None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())
        else:
            # Check whether the session exists
            if self._client.list_objects_v2(Bucket=self._bucket_name, Prefix=f"{session_id}/")["KeyCount"] > 0:
                raise ValueError(f"Session {session_id} already exists")
            
        self._session = session_id
        self._client.put_object(Bucket=self._bucket_name, Key=f"{self._session}/")
        
        return self._session
        
    
    def put_object(self, object: Any, s3_path: str) -> (bytes, str):
        if self._session is None:
            raise ValueError("No session is active")
        
        pickl = pickle.dumps(object)
        self._client.put_object(Bucket=self._bucket_name, Key=f"{self._session}/{s3_path}", Body=pickl)
        
        return pickl, f"{self._session}/{s3_path}"
    
    
    def put_file(self, file_path: str, s3_path: str) -> (bytes, str):
        if self._session is None:
            raise ValueError("No session is active")
        
        with open(file_path, "rb") as f:
            data = f.read()
            
        self._client.put_object(Bucket=self._bucket_name, Key=f"{self._session}/{s3_path}", Body=data)
        
        return data, f"{self._session}/{s3_path}"
    
    
    def get_object(self, s3_path: str) -> Any:
        if self._session is None:
            raise ValueError("No session is active")
        
        object_bytes = self._client.get_object(Bucket=self._bucket_name, Key=f"{self._session}/{s3_path}")["Body"].read()
        return pickle.loads(object_bytes)
    
    
    def get_file(self, s3_path: str, file_path: str) -> None:
        if self._session is None:
            raise ValueError("No session is active")
        
        with open(file_path, "wb") as f:
            self._client.download_fileobj(self._bucket_name, f"{self._session}/{s3_path}", f)
            
            
    def delete_object(self, s3_path: str) -> None:
        if self._session is None:
            raise ValueError("No session is active")
        
        self._client.delete_object(Bucket=self._bucket_name, Key=f"{self._session}/{s3_path}")
        
        
    def detach_session(self) -> None:
        self._session = None
        
        
    
        
    
    