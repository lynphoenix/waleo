"""
RPC 服务端基类

提供远程方法注册和调用功能
"""

import socket
import threading
import uuid
from typing import Callable, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from waleo_utils.communication.base import RPCEndpoint
from waleo_utils.communication.protocol import RPCMessage, MessageType, ErrorCode
from waleo_utils.communication.serialization import serialize, deserialize


class RPCServer(RPCEndpoint):
    """RPC 服务端基类

    提供远程方法注册和调用功能
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        max_workers: int = 4,
    ):
        """
        Args:
            host: 绑定地址
            port: 绑定端口
            max_workers: 最大工作线程数
        """
        super().__init__(host, port)
        self.max_workers = max_workers
        self._methods: Dict[str, Callable] = {}
        self._server_socket: Optional[socket.socket] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._accept_thread: Optional[threading.Thread] = None

    def register_method(self, name: str, func: Callable) -> None:
        """注册可远程调用的方法

        Args:
            name: 方法名
            func: 方法函数

        Raises:
            ValueError: 如果方法名已存在
        """
        if name in self._methods:
            raise ValueError(f"Method '{name}' already registered")
        self._methods[name] = func

    def unregister_method(self, name: str) -> None:
        """取消注册方法

        Args:
            name: 方法名
        """
        if name in self._methods:
            del self._methods[name]

    def list_methods(self) -> list[str]:
        """列出所有已注册的方法

        Returns:
            list[str]: 方法名列表
        """
        return list(self._methods.keys())

    def start(self, block: bool = True) -> None:
        """启动服务

        Args:
            block: 是否阻塞运行

        Raises:
            RuntimeError: 如果启动失败
        """
        if self._running:
            return

        try:
            # 创建服务器 socket
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(5)

            self._running = True
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

            if block:
                self._run_server()
            else:
                # 在后台线程运行
                self._accept_thread = threading.Thread(target=self._run_server, daemon=True)
                self._accept_thread.start()

        except Exception as e:
            self._running = False
            raise RuntimeError(f"Failed to start server: {e}")

    def stop(self) -> None:
        """停止服务"""
        if not self._running:
            return

        self._running = False

        # 关闭服务器 socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

        # 等待接受线程结束
        if self._accept_thread:
            try:
                self._accept_thread.join(timeout=5)
            except Exception:
                pass

        # 关闭线程池
        if self._executor:
            self._executor.shutdown(wait=True)

    def _run_server(self) -> None:
        """运行服务器主循环"""
        while self._running:
            try:
                # 接受连接
                client_socket, address = self._server_socket.accept()
                print(f"Accepted connection from {address}")

                # 在线程池中处理客户端请求
                if self._executor:
                    self._executor.submit(self._handle_client, client_socket, address)
                else:
                    # 同步处理
                    self._handle_client(client_socket, address)

            except Exception as e:
                if self._running:
                    print(f"Error accepting connection: {e}")

    def _handle_client(self, client_socket: socket.socket, address: tuple) -> None:
        """处理客户端请求

        Args:
            client_socket: 客户端 socket
            address: 客户端地址
        """
        try:
            while self._running:
                # 接收请求
                request = self._receive_message(client_socket)

                if request is None:
                    break

                # 处理请求
                response = self._process_request(request)

                # 发送响应
                self._send_message(client_socket, response)

                # 如果不是流式请求，关闭连接
                if request.msg_type != MessageType.STREAM_START:
                    break

        except Exception as e:
            print(f"Error handling client {address}: {e}")

        finally:
            client_socket.close()

    def _process_request(self, request: RPCMessage) -> RPCMessage:
        """处理请求

        Args:
            request: 请求消息

        Returns:
            RPCMessage: 响应消息
        """
        try:
            # 检查方法是否存在
            if request.method not in self._methods:
                return request.create_error(
                    ErrorCode.NOT_FOUND,
                    f"Method '{request.method}' not found"
                )

            # 调用方法
            method = self._methods[request.method]

            # 处理流式请求
            if request.msg_type == MessageType.STREAM_START:
                # 流式调用
                return self._process_stream_request(request, method)
            else:
                # 普通调用
                if request.args and request.kwargs:
                    result = method(*request.args, **request.kwargs)
                elif request.args:
                    result = method(*request.args)
                elif request.kwargs:
                    result = method(**request.kwargs)
                else:
                    result = method()

                return request.create_response(result=result)

        except Exception as e:
            return request.create_error(
                ErrorCode.INTERNAL_ERROR,
                str(e)
            )

    def _process_stream_request(self, request: RPCMessage, method: Callable) -> RPCMessage:
        """处理流式请求

        Args:
            request: 请求消息
            method: 方法函数

        Returns:
            RPCMessage: 第一个响应（STREAM_START）
        """
        # 在线程池中执行流式生成
        def _generate_stream():
            try:
                if request.args and request.kwargs:
                    generator = method(*request.args, **request.kwargs)
                elif request.args:
                    generator = method(*request.args)
                elif request.kwargs:
                    generator = method(**request.kwargs)
                else:
                    generator = method()

                for item in generator:
                    yield item

            except Exception as e:
                print(f"Error in stream: {e}")

        # 返回流式开始的响应
        # 注意：实际的流式数据需要在 _handle_client 中特殊处理
        return request.create_response(result=None)

    def _receive_message(self, sock: socket.socket) -> Optional[RPCMessage]:
        """从 socket 接收消息

        Args:
            sock: socket 对象

        Returns:
            RPCMessage: 消息对象，如果连接关闭返回 None
        """
        try:
            # 先接收数据长度
            length_bytes = sock.recv(4)
            if len(length_bytes) < 4:
                return None
            length = int.from_bytes(length_bytes, byteorder="big")

            # 再接收数据
            data = b""
            while len(data) < length:
                chunk = sock.recv(length - len(data))
                if not chunk:
                    return None
                data += chunk

            # 反序列化
            return deserialize(data)

        except Exception as e:
            print(f"Error receiving message: {e}")
            return None

    def _send_message(self, sock: socket.socket, msg: RPCMessage) -> None:
        """向 socket 发送消息

        Args:
            sock: socket 对象
            msg: 消息对象
        """
        try:
            data = serialize(msg)
            # 先发送数据长度
            length = len(data).to_bytes(4, byteorder="big")
            sock.sendall(length)
            # 再发送数据
            sock.sendall(data)

        except Exception as e:
            print(f"Error sending message: {e}")
