"""
Mini LLM Studio - Streamlit Web UI
用于 LLM 微调和对话的简单界面
"""
import streamlit as st
import requests
import json
import time
from pathlib import Path
from ws_client import StreamlitWebSocketClient, format_log_message

# 配置
BACKEND_URL = "http://localhost:8000"
INFERENCE_URL = "http://localhost:8001"
RAG_URL = "http://localhost:8002"

# 页面配置
st.set_page_config(
    page_title="Mini LLM Studio",
    page_icon="🤖",
    layout="wide"
)

# 初始化 session state
if 'current_run_id' not in st.session_state:
    st.session_state.current_run_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None


def check_backend_health():
    """检查后端服务状态"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def check_inference_health():
    """检查推理服务状态"""
    try:
        response = requests.get(f"{INFERENCE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_training(data_file, dataset_name, params):
    """启动训练任务"""
    try:
        files = None
        data = {
            "base_model": params["base_model"],
            "lora_r": params["lora_r"],
            "lora_alpha": params["lora_alpha"],
            "num_epochs": params["num_epochs"],
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
        }
        
        # 如果使用 HuggingFace 数据集
        if dataset_name:
            data["dataset_name"] = dataset_name
            data["num_samples"] = params.get("num_samples")
        
        # 如果上传了文件
        if data_file is not None:
            files = {"data_file": data_file}
        
        response = requests.post(
            f"{BACKEND_URL}/start_finetune",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


def get_task_status(run_id):
    """获取任务状态"""
    try:
        response = requests.get(f"{BACKEND_URL}/status/{run_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_all_tasks():
    """获取所有任务"""
    try:
        response = requests.get(f"{BACKEND_URL}/tasks")
        if response.status_code == 200:
            return response.json()["tasks"]
        return []
    except:
        return []


def get_logs(run_id, log_type="stderr"):
    """获取日志"""
    try:
        response = requests.get(f"{BACKEND_URL}/logs/{run_id}?log_type={log_type}")
        if response.status_code == 200:
            return response.json()["logs"]
        return "无法获取日志"
    except:
        return "无法获取日志"


def stop_task(run_id):
    """停止任务"""
    try:
        response = requests.post(f"{BACKEND_URL}/stop/{run_id}")
        return response.status_code == 200
    except:
        return False


def get_available_models():
    """获取可用的推理模型"""
    try:
        response = requests.get(f"{INFERENCE_URL}/models")
        if response.status_code == 200:
            return response.json()["models"]
        return []
    except:
        return []


def load_inference_model(model_id):
    """加载推理模型"""
    try:
        response = requests.post(f"{INFERENCE_URL}/load_model", params={"model_id": model_id})
        return response.status_code == 200
    except:
        return False


def chat_with_model(model_id, message, temperature=0.7, max_length=256):
    """与模型对话"""
    try:
        payload = {
            "model_id": model_id,
            "message": message,
            "temperature": temperature,
            "max_length": max_length
        }
        response = requests.post(f"{INFERENCE_URL}/chat", json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        return None
    except:
        return None


def chat_with_rag(model_id, message, temperature=0.7, max_length=256, top_k=3):
    """使用 RAG 增强的对话"""
    try:
        payload = {
            "model_id": model_id,
            "message": message,
            "temperature": temperature,
            "max_length": max_length,
            "use_rag": True,
            "rag_top_k": top_k
        }
        response = requests.post(f"{INFERENCE_URL}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["response"], data.get("rag_sources", [])
        return None, []
    except:
        return None, []


def upload_documents_to_rag(files):
    """上传文档到 RAG 知识库"""
    try:
        files_data = [("files", (file.name, file, "application/octet-stream")) for file in files]
        response = requests.post(f"{RAG_URL}/upload_files", files=files_data)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_rag_stats():
    """获取 RAG 知识库统计"""
    try:
        response = requests.get(f"{RAG_URL}/stats")
        return response.json() if response.status_code == 200 else None
    except:
        return None


def clear_rag_knowledge_base():
    """清空 RAG 知识库"""
    try:
        response = requests.post(f"{RAG_URL}/clear")
        return response.status_code == 200
    except:
        return False


# ==================== 主界面 ====================

st.title("🤖 Mini LLM Studio")
st.markdown("轻量级 LLM 微调平台")

# 检查后端状态
if not check_backend_health():
    st.error("⚠️ 后端服务未运行！请先启动: `python api.py`")
    st.stop()

st.success("✅ 后端服务正常")

# 创建标签页
tab1, tab2, tab3, tab4 = st.tabs(["📚 训练微调", "📊 任务管理", "📁 RAG 知识库", "💬 模型对话"])

# ==================== Tab 1: 训练微调 ====================
with tab1:
    st.header("1. 配置训练任务")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("数据源")
        data_source = st.radio(
            "选择数据来源",
            ["HuggingFace 数据集", "上传 JSONL 文件"],
            help="可以使用 HuggingFace 上的数据集或上传自己的数据"
        )
        
        dataset_name = None
        data_file = None
        
        if data_source == "HuggingFace 数据集":
            dataset_name = st.text_input(
                "数据集名称",
                value="tatsu-lab/alpaca",
                help="例如: tatsu-lab/alpaca"
            )
            num_samples = st.number_input(
                "使用样本数",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="留空使用全部数据，建议先用少量数据测试"
            )
        else:
            data_file = st.file_uploader(
                "上传训练数据",
                type=["jsonl"],
                help="JSONL 格式，每行一个 JSON 对象"
            )
            num_samples = None
            
            if data_file:
                st.info(f"✅ 已选择文件: {data_file.name}")
    
    with col2:
        st.subheader("模型配置")
        base_model = st.text_input(
            "基础模型",
            value="Qwen/Qwen2-1.5B-Instruct",
            help="HuggingFace 模型名称"
        )
        
        st.subheader("LoRA 参数")
        col_a, col_b = st.columns(2)
        with col_a:
            lora_r = st.number_input("LoRA Rank", 4, 64, 16, 4)
        with col_b:
            lora_alpha = st.number_input("LoRA Alpha", 8, 128, 32, 8)
    
    st.subheader("训练参数")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        num_epochs = st.number_input("训练轮数", 1, 10, 3, 1)
    with col4:
        batch_size = st.number_input("批次大小", 1, 16, 4, 1)
    with col5:
        learning_rate = st.number_input(
            "学习率",
            min_value=0.00001,
            max_value=0.001,
            value=0.0002,
            step=0.00001,
            format="%.5f"
        )
    
    # 启动训练按钮
    st.markdown("---")
    if st.button("🚀 开始训练", type="primary", use_container_width=True):
        # 验证输入
        if data_source == "上传 JSONL 文件" and data_file is None:
            st.error("❌ 请先上传训练数据文件")
        elif data_source == "HuggingFace 数据集" and not dataset_name:
            st.error("❌ 请输入数据集名称")
        else:
            with st.spinner("正在启动训练任务..."):
                params = {
                    "base_model": base_model,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_samples": num_samples,
                }
                
                result = start_training(data_file, dataset_name, params)
                
                if "error" in result:
                    st.error(f"❌ 启动失败: {result['error']}")
                else:
                    st.success(f"✅ 训练已启动！Run ID: {result['run_id']}")
                    st.session_state.current_run_id = result['run_id']
                    time.sleep(1)
                    st.rerun()

# ==================== Tab 2: 任务管理 ====================
with tab2:
    st.header("📊 训练任务管理")
    
    # 刷新按钮
    if st.button("🔄 刷新", key="refresh_tasks"):
        st.rerun()
    
    # 获取所有任务
    tasks = get_all_tasks()
    
    if not tasks:
        st.info("暂无训练任务")
    else:
        st.markdown(f"**共 {len(tasks)} 个任务**")
        
        # 显示任务列表
        for task in tasks:
            run_id = task["run_id"]
            status = task["status"]
            progress = task.get("progress", 0)
            
            # 状态图标
            status_icon = {
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "stopped": "⏸️",
                "starting": "⏳"
            }.get(status, "❓")
            
            with st.expander(f"{status_icon} {run_id[:8]}... - {status} ({progress}%)"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**数据源:** {task.get('data_source', 'N/A')}")
                    st.write(f"**基础模型:** {task.get('base_model', 'N/A')}")
                    st.write(f"**创建时间:** {task.get('created_at', 'N/A')}")
                    st.write(f"**进度:** {progress}%")
                    
                    # 显示参数
                    params = task.get('params', {})
                    if params:
                        st.write("**参数:**")
                        st.json(params)
                
                with col2:
                    # 操作按钮
                    if status == "running":
                        if st.button("⏸️ 停止", key=f"stop_{run_id}"):
                            if stop_task(run_id):
                                st.success("已发送停止命令")
                                time.sleep(1)
                                st.rerun()
                    
                    if st.button("📋 查看详情", key=f"detail_{run_id}"):
                        st.session_state.current_run_id = run_id
                
                # 显示日志
                if st.checkbox("显示日志", key=f"logs_{run_id}"):
                    # 选择日志模式
                    log_mode = st.radio(
                        "日志模式",
                        ["实时 WebSocket", "轮询获取"],
                        key=f"log_mode_{run_id}",
                        horizontal=True
                    )
                    
                    if log_mode == "实时 WebSocket":
                        # 实时 WebSocket 日志显示
                        st.subheader("🔴 实时日志流")
                        
                        # 初始化 WebSocket 客户端
                        ws_key = f"ws_client_{run_id}"
                        if ws_key not in st.session_state:
                            st.session_state[ws_key] = StreamlitWebSocketClient(BACKEND_URL, run_id)
                        
                        ws_client = st.session_state[ws_key]
                        
                        # 日志显示容器
                        log_container = st.container(border=True)
                        
                        # 更新按钮和刷新间隔
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            refresh_interval = st.slider(
                                "刷新间隔 (秒)",
                                1, 10, 2,
                                key=f"refresh_interval_{run_id}"
                            )
                        with col_b:
                            if st.button("🔄 手动刷新", key=f"manual_refresh_{run_id}"):
                                st.rerun()
                        
                        # 获取日志
                        with log_container:
                            logs = ws_client.get_logs()
                            if logs:
                                log_text = "\n".join([
                                    format_log_message(log) if isinstance(log, dict) else str(log)
                                    for log in logs[-100:]  # 显示最后 100 行
                                ])
                                st.code(log_text, language="text")
                                st.caption(f"📊 共 {len(logs)} 条日志")
                            else:
                                st.info("暂无日志数据")
                        
                        # 显示当前状态
                        status_info = ws_client.get_status()
                        if status_info:
                            col_x, col_y = st.columns(2)
                            with col_x:
                                st.metric("任务状态", status_info.get("status", "unknown"))
                            with col_y:
                                st.metric("进度", f"{status_info.get('progress', 0)}%")
                        
                        # 自动刷新（使用 Streamlit 的 rerun 功能）
                        st.session_state[f"last_refresh_{run_id}"] = time.time()
                    
                    else:
                        # 传统轮询方式
                        st.subheader("📋 日志内容")
                        log_type = st.radio(
                            "日志类型",
                            ["stderr", "stdout", "command"],
                            key=f"log_type_{run_id}",
                            horizontal=True
                        )
                        logs = get_logs(run_id, log_type)
                        st.code(logs, language="text")
                
                # 如果训练失败，显示错误
                if status == "failed" and "error" in task:
                    st.error(f"**错误:** {task['error']}")
                
                # 如果完成，显示模型路径
                if status == "completed" and "model_path" in task:
                    st.info(f"**模型路径:** {task['model_path']}")

# ==================== Tab 3: RAG 知识库 ====================
with tab3:
    st.header("📁 RAG 知识库管理")
    
    st.markdown("""
    RAG (检索增强生成) 允许模型基于你上传的文档来回答问题。
    """)
    
    # 知识库统计
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("上传文档")
        
        uploaded_files = st.file_uploader(
            "选择文件",
            type=["txt", "json", "jsonl"],
            accept_multiple_files=True,
            help="支持 .txt, .json, .jsonl 格式"
        )
        
        if uploaded_files:
            if st.button("📤 上传到知识库", type="primary"):
                with st.spinner("正在处理文档..."):
                    result = upload_documents_to_rag(uploaded_files)
                    
                    if result:
                        st.success(f"✅ {result['message']}")
                        st.json(result['stats'])
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 上传失败")
    
    with col2:
        st.subheader("知识库状态")
        
        stats = get_rag_stats()
        if stats:
            st.metric("文档块数", stats['total_chunks'])
            st.metric("文档数", stats['total_documents'])
            st.metric("向量维度", stats['embedding_dim'])
        else:
            st.warning("⚠️ RAG 服务未运行")
        
        if st.button("🗑️ 清空知识库", type="secondary"):
            if clear_rag_knowledge_base():
                st.success("✅ 知识库已清空")
                time.sleep(1)
                st.rerun()
    
    st.markdown("---")
    
    # 测试检索
    st.subheader("🔍 测试检索")
    
    test_query = st.text_input("输入测试查询", placeholder="例如：什么是机器学习？")
    
    if st.button("🔍 搜索") and test_query:
        try:
            response = requests.post(
                f"{RAG_URL}/search",
                json={"query": test_query, "top_k": 5}
            )
            
            if response.status_code == 200:
                results = response.json()["results"]
                
                if results:
                    st.write(f"找到 {len(results)} 个相关文档：")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"结果 {i} (相似度分数: {result['score']:.4f})"):
                            st.write("**内容:**")
                            st.write(result['content'])
                            
                            if result.get('metadata'):
                                st.write("**元数据:**")
                                st.json(result['metadata'])
                else:
                    st.info("未找到相关文档")
        except:
            st.error("❌ 检索失败，请确保 RAG 服务正在运行")

# ==================== Tab 4: 模型对话 ====================
with tab4:
    st.header("💬 与训练好的模型对话")
    
    # 检查推理服务
    if not check_inference_health():
        st.warning("⚠️ 推理服务未运行！请先启动: `python inference_api.py --port 8001`")
        st.info("推理服务运行在端口 8001，用于加载模型并提供聊天功能")
    else:
        st.success("✅ 推理服务正常")
        
        # 获取可用模型
        available_models = get_available_models()
        
        if not available_models:
            st.info("暂无可用模型，请先完成训练任务")
        else:
            # 模型选择
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_options = [
                    f"{m['model_id'][:8]}... ({m['base_model']})" 
                    for m in available_models
                ]
                selected_idx = st.selectbox(
                    "选择模型",
                    range(len(model_options)),
                    format_func=lambda x: model_options[x],
                    key="model_selector"
                )
                
                selected_model = available_models[selected_idx]
                model_id = selected_model['model_id']
                
                # 显示模型信息
                st.write(f"**模型 ID:** {model_id}")
                st.write(f"**基础模型:** {selected_model['base_model']}")
                st.write(f"**状态:** {'🟢 已加载' if selected_model['loaded'] else '⚪ 未加载'}")
            
            with col2:
                # 加载/卸载按钮
                if not selected_model['loaded']:
                    if st.button("📦 加载模型", type="primary", use_container_width=True):
                        with st.spinner("正在加载模型..."):
                            if load_inference_model(model_id):
                                st.success("✅ 模型加载成功")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ 模型加载失败")
                else:
                    st.success("✅ 模型已就绪")
                
                # RAG 开关
                use_rag = st.checkbox("🔍 使用 RAG", help="从知识库检索相关信息")
                
                if use_rag:
                    rag_top_k = st.slider("检索文档数", 1, 10, 3)
                else:
                    rag_top_k = 3
                
                # 参数设置
                temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1, key="temp")
                max_length = st.slider("最大长度", 64, 512, 256, 64, key="max_len")
            
            st.markdown("---")
            
            # 聊天界面
            if selected_model['loaded'] or st.session_state.get('force_chat'):
                st.subheader("💬 开始对话")
                
                # 显示聊天历史
                chat_container = st.container()
                with chat_container:
                    for msg in st.session_state.chat_history:
                        if msg['role'] == 'user':
                            st.chat_message("user").write(msg['content'])
                        else:
                            st.chat_message("assistant").write(msg['content'])
                
                # 输入框
                user_input = st.chat_input("输入你的问题...")
                
                if user_input:
                    # 添加用户消息
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    # 显示用户消息
                    with chat_container:
                        st.chat_message("user").write(user_input)
                    
                    # 生成回复
                    with st.spinner("正在思考..."):
                        if use_rag:
                            response, rag_sources = chat_with_rag(
                                model_id,
                                user_input,
                                temperature=temperature,
                                max_length=max_length,
                                top_k=rag_top_k
                            )
                        else:
                            response = chat_with_model(
                                model_id,
                                user_input,
                                temperature=temperature,
                                max_length=max_length
                            )
                            rag_sources = []
                    
                    if response:
                        # 添加 AI 回复
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response,
                            'rag_sources': rag_sources if use_rag else None
                        })
                        
                        # 显示 AI 回复
                        with chat_container:
                            st.chat_message("assistant").write(response)
                            
                            # 如果使用了 RAG，显示引用来源
                            if use_rag and rag_sources:
                                with st.expander(f"📚 引用了 {len(rag_sources)} 个来源"):
                                    for i, source in enumerate(rag_sources, 1):
                                        st.write(f"**来源 {i}:**")
                                        st.text(source['content'][:200] + "...")
                                        st.caption(f"相似度: {source['score']:.4f}")
                    else:
                        st.error("❌ 生成回复失败")
                
                # 清空对话
                col_a, col_b, col_c = st.columns([1, 1, 2])
                with col_a:
                    if st.button("🗑️ 清空对话", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
                
                with col_b:
                    if st.button("💾 导出对话", use_container_width=True):
                        conversation = json.dumps(
                            st.session_state.chat_history,
                            ensure_ascii=False,
                            indent=2
                        )
                        st.download_button(
                            "📥 下载 JSON",
                            conversation,
                            file_name="conversation.json",
                            mime="application/json"
                        )
            else:
                st.info("👆 请先加载模型以开始对话")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("ℹ️ 系统信息")
    
    # 后端状态
    health = check_backend_health()
    if health:
        st.success("✅ 后端服务: 正常")
    else:
        st.error("❌ 后端服务: 离线")
    
    st.markdown(f"**后端地址:** {BACKEND_URL}")
    st.markdown(f"**推理服务:** {INFERENCE_URL}")
    
    # 服务状态
    inference_health = check_inference_health()
    if inference_health:
        st.success("✅ 推理服务: 正常")
    else:
        st.warning("⚠️ 推理服务: 离线")
    
    # 当前任务
    if st.session_state.current_run_id:
        st.markdown("---")
        st.subheader("📍 当前任务")
        run_id = st.session_state.current_run_id
        status = get_task_status(run_id)
        
        if status:
            st.write(f"**Run ID:** {run_id[:8]}...")
            st.write(f"**状态:** {status['status']}")
            st.progress(status.get('progress', 0) / 100)
            
            if status['status'] == 'running':
                if st.button("🔄 自动刷新", key="auto_refresh"):
                    time.sleep(2)
                    st.rerun()
    
    # 快捷操作
    st.markdown("---")
    st.subheader("🔗 快捷链接")
    st.markdown("[📊 MLflow UI](http://localhost:5000)")
    st.markdown("[📖 API 文档](http://localhost:8000/docs)")
    
    # 使用说明
    st.markdown("---")
    st.subheader("📚 使用说明")
    st.markdown("""
    1. **训练微调**: 上传数据或选择 HF 数据集
    2. **任务管理**: 查看所有训练任务状态
    3. **模型对话**: 测试训练好的模型
    
    **提示**: 首次训练建议使用较少样本测试
    """)