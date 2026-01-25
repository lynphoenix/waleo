"""
RPC 客户端基类

提供同步和异步远程调用能力
"""

import asyncio
import socket
import uuid
from typing import Any, Iterator, Optional
from waleo.utils.communication.base import RPCEndpoint
from waleo.utils.communication.protocol import RPCMessage, MessageType, ErrorCode
from waleo.utils.communication.serialization import serialize, deserialize


class RPCClient(RPCEndpoint):
    """RPC 客户端基类

    提供同步/异步远程调用能力
    """

    def __init__(
        self,
        host: str,
        port: int = 5000,
        timeout: float = 5.0,
    ):
        """
        Args:
            host: 服务端地址
            port: 服务端端口
            timeout: 超时时间（秒）
        """
        super().__init__(host, port, timeout)
        self._socket: Optional[socket.socket] = None

    def connect(self) -> bool:
        """连接到服务端

        Returns:
            bool: 是否连接成功

        Raises:
            RuntimeError: 如果连接失败
        """
        if self._running:
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            self._running = True
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to connect to {self.address}: {e}")

    def disconnect(self) -> None:
        """断开连接"""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            finally:
                self._socket = None
        self._running = False

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._running and self._socket is not None

    def call(self, method: str, *args, **kwargs) -> Any:
        """同步调用远程方法

        Args:
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Any: 方法返回值

        Raises:
            RuntimeError: 如果连接失败或调用失败
        """
        if not self.is_connected():
            if not self.connect():
                raise RuntimeError("Not connected to server")

        # 创建请求消息
        msg = RPCMessage(
            msg_type=MessageType.REQUEST,
            msg_id=str(uuid.uuid4()),
            method=method,
            args=args,
            kwargs=kwargs,
        )

        try:
            # 发送请求
            self._send_message(msg)

            # 接收响应
            response = self._receive_message()

            # 检查错误
            if response.is_error():
                raise RuntimeError(f"RPC error ({response.error}): {response.error_msg}")

            return response.result

        except Exception as e:
            raise RuntimeError(f"RPC call failed: {e}")

    def call_async(self, method: str, *args, **kwargs) -> Any:
        """异步调用远程方法

        Args:
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Any: 方法返回值（协程）

        Note:
            此方法需要在异步上下文中使用
        """
        # 创建异步客户端并执行调用
        async def _async_call():
            client = AsyncRPCClient(self.host, self.port, self.timeout)
            async with client:
                return await client.call(method, *args, **kwargs)

        # 运行协程
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(_async_call())

    def call_stream(self, method: str, *args, **kwargs) -> Iterator[Any]:
        """流式调用（返回迭代器）

        Args:
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数

        Yields:
            Any: 流式数据块

        Raises:
            RuntimeError: 如果连接失败或调用失败
        """
        if not self.is_connected():
            if not self.connect():
                raise RuntimeError("Not connected to server")

        # 创建流式请求消息
        msg = RPCMessage(
            msg_type=MessageType.REQUEST,
            msg_id=str(uuid.uuid4()),
            method=method,
            args=args,
            kwargs=kwargs,
        )

        try:
            # 发送请求
            self._send_message(msg)

            # 接收流式响应
            while True:
                response = self._receive_message()

                if response.msg_type == MessageType.STREAM_END:
                    break
                elif response.msg_type == MessageType.ERROR:
                    raise RuntimeError(f"RPC error: {response.error_msg}")
                elif response.msg_type == MessageType.STREAM_CHUNK:
                    yield response.result
                else:
                    yield response.result
                    break

        except Exception as e:
            raise RuntimeError(f"RPC stream call failed: {e}")

    def _send_message(self, msg: RPCMessage) -> None:
        """发送消息

        Args:
            msg: 消息对象
        """
        data = serialize(msg)
        # 先发送数据长度
        length = len(data).to_bytes(4, byteorder="big")
        self._socket.sendall(length)
        # 再发送数据
        self._socket.sendall(data)

    def _receive_message(self) -> RPCMessage:
        """接收消息

        Returns:
            RPCMessage: 消息对象
        """
        # 先接收数据长度
        length_bytes = self._socket.recv(4)
        if len(length_bytes) < 4:
            raise RuntimeError("Connection closed")
        length = int.from_bytes(length_bytes, byteorder="big")

        # 再接收数据
        data = b""
        while len(data) < length:
            chunk = self._socket.recv(length - len(data))
            if not chunk:
                raise RuntimeError("Connection closed")
            data += chunk

        # 反序列化
        return deserialize(data)

    def start(self) -> None:
        """启动连接（别名）"""
        self.connect()

    def stop(self) -> None:
        """停止连接（别名）"""
        self.disconnect()


class AsyncRPCClient(RPCEndpoint):
    """异步 RPC 客户端

    使用 asyncio 实现异步通信
    """

    def __init__(
        self,
        host: str,
        port: int = 5000,
        timeout: float = 5.0,
    ):
        super().__init__(host, port, timeout)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> bool:
        """连接到服务端

        Returns:
            bool: 是否连接成功
        """
        if self._running:
            return True

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout,
            )
            self._running = True
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to connect to {self.address}: {e}")

    async def disconnect(self) -> None:
        """断开连接"""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            finally:
                self._writer = None
                self._reader = None
        self._running = False

    async def call(self, method: str, *args, **kwargs) -> Any:
        """异步调用远程方法

        Args:
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Any: 方法返回值
        """
        if not self._running:
            await self.connect()

        # 创建请求消息
        msg = RPCMessage(
            msg_type=MessageType.REQUEST,
            msg_id=str(uuid.uuid4()),
            method=method,
            args=args,
            kwargs=kwargs,
        )

        try:
            # 发送请求
            await self._send_message(msg)

            # 接收响应
            response = await self._receive_message()

            # 检查错误
            if response.is_error():
                raise RuntimeError(f"RPC error: {response.error_msg}")

            return response.result

        except Exception as e:
            raise RuntimeError(f"RPC call failed: {e}")

    async def _send_message(self, msg: RPCMessage) -> None:
        """发送消息"""
        data = serialize(msg)
        # 先发送数据长度
        length = len(data).to_bytes(4, byteorder="big")
        self._writer.write(length)
        # 再发送数据
        self._writer.write(data)
        await self._writer.drain()

    async def _receive_message(self) -> RPCMessage:
        """接收消息"""
        # 先接收数据长度
        length_bytes = await self._reader.read(4)
        if len(length_bytes) < 4:
            raise RuntimeError("Connection closed")
        length = int.from_bytes(length_bytes, byteorder="big")

        # 再接收数据
        data = await self._reader.read(length)
        if len(data) < length:
            raise RuntimeError("Connection closed")

        # 反序列化
        return deserialize(data)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    def start(self) -> None:
        """启动连接"""
        # 对于异步客户端，这个方法不适用
        raise RuntimeError("Use 'await connect()' for async client")

    def stop(self) -> None:
        """停止连接"""
        # 对于异步客户端，这个方法不适用
        raise RuntimeError("Use 'await disconnect()' for async client")
