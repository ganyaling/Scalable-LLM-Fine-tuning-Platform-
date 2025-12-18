"""
WebSocket 日志推送管理器
实时推送训练日志给前端，而不是轮询
"""
from typing import Dict, List, Set
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class LogMessage:
    """日志消息数据模型"""
    run_id: str
    timestamp: str
    level: str  # 'INFO', 'WARNING', 'ERROR'
    message: str
    line_number: int
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(asdict(self))


class WebSocketLogManager:
    """WebSocket 日志管理器"""
    
    def __init__(self):
        """初始化日志管理器"""
        # 存储每个任务的 WebSocket 连接
        self.connections: Dict[str, Set] = {}
        
        # 存储每个任务的日志历史（最后 100 行）
        self.log_history: Dict[str, List[LogMessage]] = {}
        
        # 记录每个任务的日志行号
        self.line_numbers: Dict[str, int] = {}
    
    def register_connection(self, run_id: str, websocket):
        """注册 WebSocket 连接"""
        if run_id not in self.connections:
            self.connections[run_id] = set()
            self.log_history[run_id] = []
            self.line_numbers[run_id] = 0
        
        self.connections[run_id].add(websocket)
        logger.info(f"WebSocket 连接已注册: {run_id} (总连接数: {len(self.connections[run_id])})")
    
    def unregister_connection(self, run_id: str, websocket):
        """注销 WebSocket 连接"""
        if run_id in self.connections:
            self.connections[run_id].discard(websocket)
            if not self.connections[run_id]:
                del self.connections[run_id]
                logger.info(f"任务的所有连接已断开: {run_id}")
    
    async def broadcast_log(
        self,
        run_id: str,
        message: str,
        level: str = "INFO"
    ) -> None:
        """
        广播日志消息给所有连接的客户端
        
        Args:
            run_id: 任务 ID
            message: 日志消息
            level: 日志级别 (INFO, WARNING, ERROR)
        """
        if run_id not in self.connections:
            return
        
        # 创建日志消息对象
        line_num = self.line_numbers.get(run_id, 0) + 1
        self.line_numbers[run_id] = line_num
        
        log_msg = LogMessage(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            line_number=line_num
        )
        
        # 保存到历史（最多 100 条）
        if run_id not in self.log_history:
            self.log_history[run_id] = []
        
        self.log_history[run_id].append(log_msg)
        if len(self.log_history[run_id]) > 100:
            self.log_history[run_id].pop(0)
        
        # 广播给所有连接
        disconnected = set()
        for websocket in self.connections.get(run_id, set()):
            try:
                await websocket.send_json({
                    "type": "log",
                    "data": asdict(log_msg)
                })
            except Exception as e:
                logger.warning(f"发送日志失败: {e}")
                disconnected.add(websocket)
        
        # 清理断开的连接
        for ws in disconnected:
            self.unregister_connection(run_id, ws)
    
    async def broadcast_progress(
        self,
        run_id: str,
        progress: int,
        message: str = ""
    ) -> None:
        """
        广播进度更新
        
        Args:
            run_id: 任务 ID
            progress: 进度百分比 (0-100)
            message: 可选的进度消息
        """
        if run_id not in self.connections:
            return
        
        disconnected = set()
        for websocket in self.connections.get(run_id, set()):
            try:
                await websocket.send_json({
                    "type": "progress",
                    "data": {
                        "run_id": run_id,
                        "progress": progress,
                        "message": message,
                        "timestamp": datetime.now().isoformat()
                    }
                })
            except Exception as e:
                logger.warning(f"发送进度失败: {e}")
                disconnected.add(websocket)
        
        for ws in disconnected:
            self.unregister_connection(run_id, ws)
    
    async def broadcast_status(
        self,
        run_id: str,
        status: str,
        message: str = ""
    ) -> None:
        """
        广播状态更新
        
        Args:
            run_id: 任务 ID
            status: 状态 (running, completed, failed, stopped)
            message: 状态消息
        """
        if run_id not in self.connections:
            return
        
        disconnected = set()
        for websocket in self.connections.get(run_id, set()):
            try:
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "run_id": run_id,
                        "status": status,
                        "message": message,
                        "timestamp": datetime.now().isoformat()
                    }
                })
            except Exception as e:
                logger.warning(f"发送状态失败: {e}")
                disconnected.add(websocket)
        
        for ws in disconnected:
            self.unregister_connection(run_id, ws)
    
    def get_log_history(self, run_id: str) -> List[Dict]:
        """获取日志历史"""
        if run_id not in self.log_history:
            return []
        return [asdict(log) for log in self.log_history[run_id]]
    
    def clear_logs(self, run_id: str) -> None:
        """清除日志历史"""
        if run_id in self.log_history:
            self.log_history[run_id] = []
            self.line_numbers[run_id] = 0
    
    def get_connection_count(self, run_id: str) -> int:
        """获取连接数"""
        return len(self.connections.get(run_id, set()))
    
    def get_all_connections_count(self) -> int:
        """获取全部连接数"""
        return sum(len(conns) for conns in self.connections.values())


# 全局日志管理器实例
log_manager = WebSocketLogManager()
