from abc import ABC, abstractmethod
from enum import Enum
import json
from loguru import Message, logger
import os
import sys
import traceback
from typing import Callable, Dict, List

class BaseAlert(ABC):
    @abstractmethod
    def send_error_msg(self, content: str):
        raise NotImplementedError()

class Extra(Enum):
    Serialized = "serialized"
    TraceId = "trace_id"

_alerts: List[BaseAlert] = []

def _call_alerts(message: Message):
    serialized_text = _serialize_output(message.record)
    for alert in _alerts:
        try:
            alert.send_error_msg(serialized_text)
        except:
            logger.warning(traceback.format_exc())

def _get_formatter(
    serialize: Callable[[Dict[str, any]], str]
) -> Callable[[Dict[str, any]], str]:
    def formatter(record: Dict[str, any]) -> str:
        record["extra"]["serialized"] = serialize(record)
        return "{extra[serialized]}\n"

    return formatter

def _serialize_extra(extra: Dict[str, any]) -> str:
    info = json.dumps({
        k: v for k, v in extra.items()
        if k not in [Extra.Serialized.value, Extra.TraceId.value]
    }, ensure_ascii=False)
    return "{} ".format(info) if info != "{}" else ""

def _serialize_output(record: Dict[str, any]) -> str:
    extra = _serialize_extra(record["extra"])
    request_id = record["extra"].get(Extra.TraceId.value, "")

    return "[{},{}] {}{}:{} in {}\t{}".format(
        request_id,
        record["level"].name,
        extra,
        record["file"].path,
        record["line"],
        record["function"],
        record["message"],
    )

def init_logger(log_file: str, alerts: List[BaseAlert] = []):
    global _alerts
    logger.remove(handler_id=None)
    log_path = os.path.join(os.environ.get("LOGPATH", "."), log_file)
    logger.add(
        log_path,
        rotation="50 MB",
        encoding="utf-8",
        enqueue=True,
        retention="7 days",
        level="DEBUG",
        format=_get_formatter(_serialize_output),
    )
    logger.add(_call_alerts, level="ERROR")
    logger.add(
        sys.stdout,
        enqueue=True,
        colorize=True,
        level="INFO",
        format=_get_formatter(_serialize_output),
    )
