import streamlit as st
import os
import numpy as np
import dashscope
from pypdf import PdfReader
from hello_agents import HelloAgentsLLM

# ==================== 配置 ====================
dashscope.api_key = st.secrets.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

llm = HelloAgentsLLM(
    model=st.secrets.get("LLM_MODEL") or os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"),
    api_key=st.secrets.get("LLM_API_KEY") or os.getenv("LLM_API_KEY"),
    base_url=st.secrets.get("LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1")
)


# ==================== 核心逻辑（加缓存）====================
@st.cache_data(show_spinner=False)
def process_pdf(file_bytes):
    """读取PDF并切片，带缓存避免重复处理"""
    reader = PdfReader(file_bytes)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = []
    start = 0
    text_len = len(full_text)
    while start < text_len:
        end = start + 500
        chunk = full_text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
        start = end - 50
    return chunks

    def is_toc(chunk):
        # 目录块的特征：包含"目录"字样，或大量"..."，或大量"数字+点"的章节行
        if "目录" in chunk:
            return True
        if chunk.count("...") > 3:
            return True
        # 统计"数字+点"的行数（如"1.xxx......1"）
        import re
        toc_lines = len(re.findall(r'\d+\.\s.*?\.{3,}', chunk))
        if toc_lines > 3:
            return True
        return False

    chunks = [c for c in chunks if not is_toc(c)]
    # 👆 过滤结束

    return chunks

    chunks = [c for c in chunks if not is_toc(c)]
    # 👆 过滤结束

    return chunks
@st.cache_data(show_spinner=False)

def build_vector_store(chunks):
    """向量化所有切片，带缓存"""
    vectors = []
    for chunk in chunks:
        try:
            resp = dashscope.TextEmbedding.call(model="text-embedding-v3", input=chunk)
            if resp.status_code == 200:
                vectors.append(np.array(resp.output["embeddings"][0]["embedding"]))
            else:
                vectors.append(np.zeros(1024))
        except:
            vectors.append(np.zeros(1024))
    return vectors


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve_by_vector(query, chunks, vectors, top_k=3):
    resp = dashscope.TextEmbedding.call(model="text-embedding-v3", input=query)
    if resp.status_code != 200:
        return []
    query_vec = np.array(resp.output["embeddings"][0]["embedding"])

    scores = []
    for i, vec in enumerate(vectors):
        sim = cosine_similarity(query_vec, vec)
        scores.append((sim, chunks[i]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:top_k]]


# ==================== 界面 ====================
st.set_page_config(page_title="校规智能问答助手", page_icon="📚", layout="centered")
st.title("📚 校规智能问答助手")
st.caption("上传校规 PDF，用自然语言提问，帮你找到相关条款并给出依据。")

# 侧边栏上传
with st.sidebar:
    st.header("📄 上传校规")
    uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"])
    if uploaded_file:
        st.success(f"已上传：{uploaded_file.name}")
        st.info("提示：首次上传需要向量化所有切片，请耐心等待约 30 秒。")

# 主聊天区
if uploaded_file:
    # 1. 处理 PDF
    with st.spinner("正在解析 PDF 并切片..."):
        chunks = process_pdf(uploaded_file)

    # 2. 构建向量库
    with st.spinner("正在向量化切片（首次处理约30秒）..."):
        vectors = build_vector_store(chunks)

    st.success(f"✅ 处理完成！共 {len(chunks)} 个切片，已建立向量索引。")

    # 3. 聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 4. 用户输入
    if prompt := st.chat_input("请输入你的问题，例如：体测不合格影响毕业吗？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在检索和思考..."):
                relevant = retrieve_by_vector(prompt, chunks, vectors, top_k=3)
                context = "\n\n---\n\n".join(relevant)

                rag_prompt = f"""你是一个校规问答助手。请根据以下校规条款，回答用户的问题。
如果条款中没有明确答案，请如实说明“校规中没有找到相关规定”，不要编造。

【校规条款】
{context}

【用户问题】
{prompt}

请给出准确、有依据的回答，并注明依据来自哪一条款。"""

                response = llm.invoke([{"role": "user", "content": rag_prompt}])
                st.markdown(response)

                # 展示依据来源
                with st.expander("📖 查看检索到的原始条款"):
                    for i, chunk in enumerate(relevant, 1):
                        st.markdown(f"**切片 {i}**\n\n{chunk}")

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👈 请先在左侧上传一份校规 PDF 文件，即可开始问答。")
