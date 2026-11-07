"""项目内部可安全展示给用户的异常类型。"""


class ServiceError(RuntimeError):
    """带中文说明的服务异常；消息可以直接展示，不包含密钥或请求正文。"""
