# RPC 通信 API

## 类

### RPCServer

```python
class RPCServer(RPCEndpoint):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        max_workers: int = 4,
    )
```

RPC 服务端。

#### 方法

##### register_method

```python
def register_method(self, name: str, func: Callable) -> None
```

注册可远程调用的方法。

**异常**：
- `ValueError` - 如果方法名已存在

##### unregister_method

```python
def unregister_method(self, name: str) -> None
```

取消注册方法。

##### list_methods

```python
def list_methods(self) -> list[str]
```

列出所有已注册的方法。

##### start

```python
def start(self, block: bool = True) -> None
```

启动服务。

##### stop

```python
def stop(self) -> None
```

停止服务。

---

### RPCClient

```python
class RPCClient(RPCEndpoint):
    def __init__(
        self,
        host: str,
        port: int = 5000,
        timeout: float = 5.0,
    )
```

同步 RPC 客户端。

#### 方法

##### connect

```python
def connect(self) -> bool
```

连接到服务端。

##### disconnect

```python
def disconnect(self) -> None
```

断开连接。

##### is_connected

```python
def is_connected(self) -> bool
```

检查连接状态。

##### call

```python
def call(self, method: str, *args, **kwargs) -> Any
```

同步调用远程方法。

##### call_stream

```python
def call_stream(self, method: str, *args, **kwargs) -> Iterator[Any]
```

流式调用（返回迭代器）。

##### call_async

```python
def call_async(self, method: str, *args, **kwargs) -> Any
```

异步调用远程方法。

---

### AsyncRPCClient

```python
class AsyncRPCClient(RPCEndpoint):
    def __init__(
        self,
        host: str,
        port: int = 5000,
        timeout: float = 5.0,
    )
```

异步 RPC 客户端。

#### 方法

##### connect

```python
async def connect(self) -> bool
```

连接到服务端。

##### disconnect

```python
async def disconnect(self) -> None
```

断开连接。

##### call

```python
async def call(self, method: str, *args, **kwargs) -> Any
```

异步调用远程方法。

---

## 数据类

### MessageType

```python
class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    ERROR = "error"
```

消息类型。

---

### ErrorCode

```python
class ErrorCode(Enum):
    OK = 0
    NOT_FOUND = 1
    INVALID_REQUEST = 2
    INTERNAL_ERROR = 3
    TIMEOUT = 4
    SERIALIZATION_ERROR = 5
    DESERIALIZATION_ERROR = 6
```

错误码。

---

### RPCMessage

```python
@dataclass
class RPCMessage:
    msg_type: MessageType
    msg_id: str
    method: Optional[str] = None
    args: Optional[Tuple] = None
    kwargs: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[ErrorCode] = None
    error_msg: Optional[str] = None
```

RPC 消息格式。

#### 方法

##### to_dict

```python
def to_dict(self) -> dict
```

转换为字典。

##### from_dict

```python
@classmethod
def from_dict(cls, data: dict) -> RPCMessage
```

从字典创建。

##### is_error

```python
def is_error(self) -> bool
```

是否为错误消息。

##### create_response

```python
def create_response(self, result: Any = None, error: ErrorCode = None, error_msg: str = None) -> RPCMessage
```

创建响应消息。

##### create_error

```python
def create_error(self, error: ErrorCode, error_msg: str) -> RPCMessage
```

创建错误消息。
