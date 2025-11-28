"""
Mini LLM Studio - Streamlit Web UI
用于 LLM 微调和对话的简单界面
"""
import streamlit as st
import requests
import json
import time
from pathlib import Path

# api endpoints
BACKEND_URL = "http://localhost:8000"
INFERENCE_URL = "http://localhost:8001"

# page settings
st.set_page_config(
    page_title="Mini LLM Studio",
    page_icon="🤖",
    layout="wide"
)

# initialize session state
if 'current_run_id' not in st.session_state:
    st.session_state.current_run_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None


def check_backend_health():
    """Check backend service health"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def check_inference_health():
    """Check inference service health"""
    try:
        response = requests.get(f"{INFERENCE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_training(data_file, dataset_name, params):
    """Start training task"""
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
        
        # If using HuggingFace dataset
        if dataset_name:
            data["dataset_name"] = dataset_name
            data["num_samples"] = params.get("num_samples")
        
        # If uploading a file
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
    """Get task status"""
    try:
        response = requests.get(f"{BACKEND_URL}/status/{run_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_all_tasks():
    """Get all tasks"""
    try:
        response = requests.get(f"{BACKEND_URL}/tasks")
        if response.status_code == 200:
            return response.json()["tasks"]
        return []
    except:
        return []


def get_logs(run_id, log_type="stderr"):
    """Get logs"""
    try:
        response = requests.get(f"{BACKEND_URL}/logs/{run_id}?log_type={log_type}")
        if response.status_code == 200:
            return response.json()["logs"]
        return "Failed to get logs"
    except:
        return "Failed to get logs"


def stop_task(run_id):
    """Stop task"""
    try:
        response = requests.post(f"{BACKEND_URL}/stop/{run_id}")
        return response.status_code == 200
    except:
        return False


def get_available_models():
    """Get available inference models"""
    try:
        response = requests.get(f"{INFERENCE_URL}/models")
        if response.status_code == 200:
            return response.json()["models"]
        return []
    except:
        return []


def load_inference_model(model_id):
    """Load inference model"""
    try:
        response = requests.post(f"{INFERENCE_URL}/load_model", params={"model_id": model_id})
        return response.status_code == 200
    except:
        return False


def chat_with_model(model_id, message, temperature=0.7, max_length=256):
    """Chat with model"""
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


# ==================== 主界面 ====================

st.title("🤖 Mini LLM Studio")
st.markdown("Lightweight LLM Fine-tuning Platform")

# Check backend health
if not check_backend_health():
    st.error("⚠️ Backend service is not running! Please start it first: `python api.py`")
    st.stop()

st.success("✅ Backend service is running")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📚 Fine-tune Training", "📊 Task Management", "💬 Model Chat"])

# ==================== Tab 1: Fine-tune Training ====================
with tab1:
    st.header("1. Configure Training Task")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Source")
        data_source = st.radio(
            "Select Data Source",
            ["HuggingFace Dataset", "Upload JSONL File"],
            help="You can use datasets from HuggingFace or upload your own data in JSONL format"
        )
        
        dataset_name = None
        data_file = None
        
        if data_source == "HuggingFace Dataset":
            dataset_name = st.text_input(
                "Dataset Name",
                value="tatsu-lab/alpaca",
                help="e.g., tatsu-lab/alpaca"
            )
            num_samples = st.number_input(
                "Number of Samples to Use",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="Leave empty to use all data. It's recommended to test with a small amount first."
            )
        else:
            data_file = st.file_uploader(
                "Upload Training Data (JSONL Format)",
                type=["jsonl"],
                help="JSONL format, one JSON object per line with 'instruction' and 'response' fields"
            )
            num_samples = None
            
            if data_file:
                st.info(f"✅ Selected file: {data_file.name}")
    
    with col2:
        st.subheader("Model Configuration")
        base_model = st.text_input(
            "Base Model",
            value="Qwen/Qwen2-1.5B-Instruct",
            help="HuggingFace model name"
        )
        
        st.subheader("LoRA Parameters")
        col_a, col_b = st.columns(2)
        with col_a:
            lora_r = st.number_input("LoRA Rank", 4, 64, 16, 4)
        with col_b:
            lora_alpha = st.number_input("LoRA Alpha", 8, 128, 32, 8)
    
    st.subheader("Training Parameters")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        num_epochs = st.number_input("Number of Epochs", 1, 10, 3, 1)
    with col4:
        batch_size = st.number_input("Batch Size", 1, 16, 4, 1)
    with col5:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=0.001,
            value=0.0002,
            step=0.00001,
            format="%.5f"
        )
    
    # Start Training Button
    st.markdown("---")
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Validate input
        if data_source == "Upload JSONL File" and data_file is None:
            st.error("❌ Please upload the training data file first")
        elif data_source == "HuggingFace Dataset" and not dataset_name:
            st.error("❌ Please enter the dataset name")
        else:
            with st.spinner("Starting training task..."):
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
                    st.error(f"❌ Failed to start: {result['error']}")
                else:
                    st.success(f"✅ Training started! Run ID: {result['run_id']}")
                    st.session_state.current_run_id = result['run_id']
                    time.sleep(1)
                    st.rerun()

# ==================== Tab 2: Task Management ====================
with tab2:
    st.header("📊 Task Management")
    
    # refresh button
    if st.button("🔄 refresh", key="refresh_tasks"):
        st.rerun()
    
    # Get all tasks
    tasks = get_all_tasks()
    
    if not tasks:
        st.info("No training tasks available. Start a new training task in the 'Fine-tune Training' tab.")
    else:
        st.markdown(f"**Total {len(tasks)} tasks**")
        
        # Display task list
        for task in tasks:
            run_id = task["run_id"]
            status = task["status"]
            progress = task.get("progress", 0)
            
            # Status icon
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
                    st.write(f"**Data Source:** {task.get('data_source', 'N/A')}")
                    st.write(f"**Base Model:** {task.get('base_model', 'N/A')}")
                    st.write(f"**Created At:** {task.get('created_at', 'N/A')}")
                    st.write(f"**Progress:** {progress}%")
                    
                    # Display parameters
                    params = task.get('params', {})
                    if params:
                        st.write("**Parameters:**")
                        st.json(params)
                
                with col2:
                    # Action buttons
                    if status == "running":
                        if st.button("⏸️ Stop", key=f"stop_{run_id}"):
                            if stop_task(run_id):
                                st.success("Stop command sent")
                                time.sleep(1)
                                st.rerun()
                    
                    if st.button("📋 View Details", key=f"detail_{run_id}"):
                        st.session_state.current_run_id = run_id
                
                # Display logs
                if st.checkbox("Display Logs", key=f"logs_{run_id}"):
                    log_type = st.radio(
                        "Log Type",
                        ["stderr", "stdout", "command"],
                        key=f"log_type_{run_id}",
                        horizontal=True
                    )
                    logs = get_logs(run_id, log_type)
                    st.code(logs, language="text")
                
                # If training failed, display error
                if status == "failed" and "error" in task:
                    st.error(f"**Error:** {task['error']}")
                
                # If completed, display model path
                if status == "completed" and "model_path" in task:
                    st.info(f"**Model Path:** {task['model_path']}")

# ==================== Tab 3: Model Chat ====================
with tab3:
    st.header("💬 Chat with Trained Models")
    
    # Check inference service health
    if not check_inference_health():
        st.warning("⚠️ Inference service is not running! Please start it first: `python inference_api.py --port 8001`")
        st.info("The inference service runs on port 8001 and is used to load models and provide chat functionality")
    else:
        st.success("✅ Inference service is healthy")
        
        # Get available models
        available_models = get_available_models()
        
        if not available_models:
            st.info("No available models. Please complete a training task first.")
        else:
            # Model selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_options = [
                    f"{m['model_id'][:8]}... ({m['base_model']})" 
                    for m in available_models
                ]
                selected_idx = st.selectbox(
                    "Select Model",
                    range(len(model_options)),
                    format_func=lambda x: model_options[x],
                    key="model_selector"
                )
                
                selected_model = available_models[selected_idx]
                model_id = selected_model['model_id']
                
                # Display model information
                st.write(f"**Model ID:** {model_id}")
                st.write(f"**Base Model:** {selected_model['base_model']}")
                st.write(f"**Status:** {'🟢 Loaded' if selected_model['loaded'] else '⚪ Not Loaded'}")
            
            with col2:
                # Load/Unload buttons
                if not selected_model['loaded']:
                    if st.button("📦 Load Model", type="primary", use_container_width=True):
                        with st.spinner("Loading model..."):
                            if load_inference_model(model_id):
                                st.success("✅ Model loaded successfully")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Model loading failed")
                else:
                    st.success("✅ Model is ready for chat")
                
                # Parameter settings
                temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1, key="temp")
                max_length = st.slider("Max Length", 64, 512, 256, 64, key="max_len")
            
            st.markdown("---")
            
            # Chat interface
            if selected_model['loaded'] or st.session_state.get('force_chat'):
                st.subheader("💬 Start Chatting")
                
                # Display chat history
                chat_container = st.container()
                with chat_container:
                    for msg in st.session_state.chat_history:
                        if msg['role'] == 'user':
                            st.chat_message("user").write(msg['content'])
                        else:
                            st.chat_message("assistant").write(msg['content'])
                
                # Input box
                user_input = st.chat_input("Enter your question...")
                
                if user_input:
                    # Add user message to history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    # Display user message
                    with chat_container:
                        st.chat_message("user").write(user_input)
                    
                    # get model response
                    with st.spinner("Thinking..."):
                        response = chat_with_model(
                            model_id, 
                            user_input,
                            temperature=temperature,
                            max_length=max_length
                        )
                    
                    if response:
                        # Add AI response
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                        
                        # Display AI response
                        with chat_container:
                            st.chat_message("assistant").write(response)
                    else:
                        st.error("❌ Failed to generate response from model")
                
                # Clear conversation and Export buttons
                col_a, col_b, col_c = st.columns([1, 1, 2])
                with col_a:
                    if st.button("🗑️ Clear Conversation", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
                
                with col_b:
                    if st.button("💾 Export Conversation", use_container_width=True):
                        conversation = json.dumps(
                            st.session_state.chat_history,
                            ensure_ascii=False,
                            indent=2
                        )
                        st.download_button(
                            "📥 Download JSON",
                            conversation,
                            file_name="conversation.json",
                            mime="application/json"
                        )
            else:
                st.info("👆 Please load a model to start chatting")

# ==================== Sidebar ====================
with st.sidebar:
    st.header("ℹ️ System Information")
    
    # Backend status
    health = check_backend_health()
    if health:
        st.success("✅ Backend Service: Online")
    else:
        st.error("❌ Backend Service: Offline")
    
    st.markdown(f"**Backend URL:** {BACKEND_URL}")
    st.markdown(f"**Inference Service:** {INFERENCE_URL}")
    
    # Inference status
    inference_health = check_inference_health()
    if inference_health:
        st.success("✅ Inference Service: Online")
    else:
        st.warning("⚠️ Inference Service: Offline")
    
    # Current Task Status
    if st.session_state.current_run_id:
        st.markdown("---")
        st.subheader("📍 Current Task")
        run_id = st.session_state.current_run_id
        status = get_task_status(run_id)
        
        if status:
            st.write(f"**Run ID:** {run_id[:8]}...")
            st.write(f"**Status:** {status['status']}")
            st.progress(status.get('progress', 0) / 100)
            
            if status['status'] == 'running':
                if st.button("🔄 Auto Refresh", key="auto_refresh"):
                    time.sleep(2)
                    st.rerun()
    
    # Quick Actions
    st.markdown("---")
    st.subheader("🔗 Quick Links")
    st.markdown("[📊 MLflow UI](http://localhost:5000)")
    st.markdown("[📖 API Documentation](http://localhost:8000/docs)")
    
    # Usage Instructions
    st.markdown("---")
    st.subheader("📚 Usage Instructions")
    st.markdown("""
    1. **Fine-tune Training**: Upload data or select HF dataset               
    2. **Task Management**: View status of all training tasks               
    3. **Model Chat**: Test trained models
    
    **Note**: It is recommended to use fewer samples for initial training tests
    """)