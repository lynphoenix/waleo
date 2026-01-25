"""
RPC 端点基类

定义 RPC 通信的端点抽象接口
"""

from abc import ABC, abstractmethod
from typing import Optional


class RPCEndpoint(ABC):
    """RPC 端点基类

    定义 RPC 服务和客户端的通用接口
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 5.0,
    ):
        """
        Args:
            host: 主机地址
            port: 端口号
            timeout: 超时时间（秒）
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self._running = False

    @property
    def address(self) -> str:
        """获取地址字符串"""
        return f"{self.host}:{self.port}"

    @abstractmethod
    def start(self) -> None:
        """启动服务/连接

        Raises:
            RuntimeError: 如果启动失败
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止服务/断开连接"""
        pass

    @property
    def is_running(self) -> bool:
        """检查是否运行中"""
        return self._running

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
