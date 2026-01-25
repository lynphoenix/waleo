"""
协议定义

定义 RPC 通信的消息类型、错误码和消息格式
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    ERROR = "error"


class ErrorCode(Enum):
    """错误码"""
    OK = 0
    NOT_FOUND = 1
    INVALID_REQUEST = 2
    INTERNAL_ERROR = 3
    TIMEOUT = 4
    SERIALIZATION_ERROR = 5
    DESERIALIZATION_ERROR = 6


@dataclass
class RPCMessage:
    """RPC 消息格式"""
    msg_type: MessageType
    msg_id: str
    method: Optional[str] = None
    args: Optional[Tuple] = None
    kwargs: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[ErrorCode] = None
    error_msg: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "msg_type": self.msg_type.value,
            "msg_id": self.msg_id,
            "method": self.method,
            "args": self.args,
            "kwargs": self.kwargs,
            "result": self.result,
            "error": self.error.value if self.error else None,
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RPCMessage":
        """从字典创建"""
        return cls(
            msg_type=MessageType(data["msg_type"]),
            msg_id=data["msg_id"],
            method=data.get("method"),
            args=data.get("args"),
            kwargs=data.get("kwargs"),
            result=data.get("result"),
            error=ErrorCode(data["error"]) if data.get("error") else None,
            error_msg=data.get("error_msg"),
        )

    def is_error(self) -> bool:
        """是否为错误消息"""
        return self.error is not None and self.error != ErrorCode.OK

    def create_response(self, result: Any = None, error: ErrorCode = None, error_msg: str = None) -> "RPCMessage":
        """创建响应消息"""
        return RPCMessage(
            msg_type=MessageType.RESPONSE,
            msg_id=self.msg_id,
            result=result,
            error=error,
            error_msg=error_msg,
        )

    def create_error(self, error: ErrorCode, error_msg: str) -> "RPCMessage":
        """创建错误消息"""
        return RPCMessage(
            msg_type=MessageType.ERROR,
            msg_id=self.msg_id,
            error=error,
            error_msg=error_msg,
        )
