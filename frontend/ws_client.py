"""
WebSocket 客户端 - 用于实时日志流式传输
支持在 Streamlit 中通过 HTTP 轮询获取实时日志
"""
import json
from typing import Optional
from datetime import datetime
try:
    import requests
except ImportError:
    requests = None






class StreamlitWebSocketClient:
    """
    Streamlit 优化版 WebSocket 客户端
    使用轮询方式从后端获取日志，避免 Streamlit 的限制
    """
    
    def __init__(self, backend_url: str, run_id: str):
        """
        初始化客户端
        
        Args:
            backend_url: 后端 URL (例如: http://localhost:8000)
            run_id: 训练任务 ID
        """
        self.backend_url = backend_url
        self.run_id = run_id
        self.logs = []
        self.last_log_count = 0
    
    def get_logs(self) -> list:
        """从后端获取最新日志"""
        try:
            import requests
            response = requests.get(
                f"{self.backend_url}/logs/{self.run_id}/history",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                self.logs = data.get("logs", [])
                return self.logs
        except Exception as e:
            print(f"获取日志失败: {e}")
        return self.logs
    
    def get_new_logs(self) -> list:
        """获取仅新增的日志"""
        self.get_logs()
        new_logs = self.logs[self.last_log_count:]
        self.last_log_count = len(self.logs)
        return new_logs
    
    def get_status(self) -> dict:
        """获取任务状态"""
        try:
            import requests
            response = requests.get(
                f"{self.backend_url}/status/{self.run_id}",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"获取状态失败: {e}")
        return {}
    
    def clear_logs(self):
        """清除服务器上的日志"""
        try:
            import requests
            requests.delete(
                f"{self.backend_url}/logs/{self.run_id}",
                timeout=5
            )
        except Exception as e:
            print(f"清除日志失败: {e}")
    
    def get_all_logs_status(self) -> dict:
        """获取所有日志的全局状态"""
        try:
            import requests
            response = requests.get(
                f"{self.backend_url}/logs/status",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"获取日志状态失败: {e}")
        return {}


def format_log_message(log_item: dict) -> str:
    """
    格式化日志消息用于显示
    
    Args:
        log_item: 日志项目（dict 或 str）
    
    Returns:
        格式化的日志字符串
    """
    if isinstance(log_item, str):
        return log_item
    
    if isinstance(log_item, dict):
        timestamp = log_item.get("timestamp", "")
        level = log_item.get("level", "INFO")
        message = log_item.get("message", "")
        
        if timestamp:
            # 解析时间戳
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = timestamp[:8]
            
            return f"[{time_str}] {level}: {message}"
        else:
            return f"{level}: {message}"
    
    return str(log_item)
