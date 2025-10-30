from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
import operator
import re, json

from .system_prompt import get_reasoning_prompt, get_summary_prompt
from .model import chatModels
from .agent import retrieve_documents  # 🔥 直接导入 retriever，不需要初始化
import requests
import random
import string
from typing import Dict, Any


# ===========================
# 1. 定义状态
# ===========================
class AgentState(TypedDict):
    """Agent 的状态定义"""
    messages: Annotated[list, operator.add]
    rag_context: str
    next_action: str


class MockMode:
    """控制是否使用 Mock 数据"""
    ENABLED = True


@tool
def get_company_portrait(company_name: str) -> Dict[str, Any]:
    """查询企业画像信息"""
    if MockMode.ENABLED:
        return {
            company_name: {
                '高管属性': '法代一人多企',
                '经营属性': '同业从业人员多,主营业务',
                '基础属性': None,
                '监管属性': None,
                '能力属性': None,
                '社会影响特征': None
            }
        }

    url = "http://127.0.0.1:8090/get_portrait"
    params = {"query": company_name}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"查询失败: {str(e)}"}


@tool
def search_penalty_basis(illegal_behavior: str, req_type: int = 3) -> list:
    """查询裁量基准"""
    if MockMode.ENABLED:
        return [{
            '违法行为': {'名称': illegal_behavior},
            '裁量等级': {
                '裁量等级': '一般',
                '裁量基准': '1.没收违法所得；2.没收违法生产经营的食品...',
                '违法行为危害程度': '产品已经销售且货值金额3000元至7000元的',
                '违法行为危害后果': '造成轻微财产损失，但不构成食品安全事故的'
            },
            '法律条例': {
                '内容': '第一百二十二条第一款 违反本法规定...'
            }
        }]

    url = "http://127.0.0.1:18002/search_penalty_basis"
    req_id = "".join(random.sample(string.ascii_letters + string.digits, 10))
    params = {
        "query": illegal_behavior,
        "req_type": req_type,
        "req_id": req_id
    }
    try:
        response = requests.post(url, json=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return [{"error": f"查询失败: {str(e)}"}]


@tool
def search_law_content(law_clause: str) -> str:
    """查询法律条款内容"""
    if MockMode.ENABLED:
        return f"第二十一条 食品安全风险评估结果是制定、修订食品安全标准和实施食品安全监督管理的科学依据。（Mock数据）"

    url = "http://127.0.0.1:18002/search_law_wc"
    req_id = "".join(random.sample(string.ascii_letters + string.digits, 10))
    params = {
        "query": law_clause,
        "req_type": 1,
        "req_id": req_id
    }
    try:
        response = requests.post(url, json=params, timeout=10)
        response.raise_for_status()
        return response.text or "未找到相关法律条款"
    except Exception as e:
        return f"查询失败: {str(e)}"


@tool
def recommend_similar_cases(
        industry: str,
        domain: str,
        abstract: str,
        fact: str
) -> Dict[str, str]:
    """推荐类似案件"""
    if MockMode.ENABLED:
        return {
            'fact1': '案件号123,案件来源,2019-10-29,立案号456,违法广告,上海某公司,虚假宣传,...',
            'fact2': '案件号789,案件来源,2020-01-15,立案号012,违法广告,上海某公司2,虚假宣传,...',
            'fact3': '案件号345,案件来源,2019-07-04,立案号678,违法广告,上海某公司3,虚假宣传,...'
        }

    url = "http://127.0.0.1:8021/get_leian_v2"
    params = {
        "hangye": industry,
        "domain": domain,
        "abstract": abstract,
        "fact": fact
    }
    try:
        response = requests.post(url, json=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"查询失败: {str(e)}"}


@tool
def recommend_illegal_behavior(
        industry: str,
        domain: str,
        abstract: str,
        fact: str
) -> Dict[str, str]:
    """推荐违法行为标签"""
    if MockMode.ENABLED:
        return {
            'act1': '违反本法第二十八条第二款第（二）项规定，发布虚假广告的',
            'act2': '违反本法第二十八条第二款第（五）项规定，发布虚假广告的',
            'act3': '违反本法第十六条第一款第一项规定发布医疗、药品、医疗器械广告的'
        }

    url = "http://127.0.0.1:8021/get_xingwei_v2"
    params = {
        "hangye": industry,
        "domain": domain,
        "abstract": abstract,
        "fact": fact
    }
    try:
        response = requests.post(url, json=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"查询失败: {str(e)}"}


TOOLS = [
    get_company_portrait,
    search_penalty_basis,
    search_law_content,
    recommend_similar_cases,
    recommend_illegal_behavior,
]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# ===========================
# 3. 节点定义
# ===========================

def retrieve_node(state: AgentState) -> dict:
    """节点1：RAG 检索"""
    print("\n" + "=" * 80)
    print("🔍 [检索节点] 开始检索（使用 similarities RAG）")
    print("=" * 80)

    # 获取最后一条用户消息
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)

    print(f"\n📝 用户问题: {query}\n")

    # 🔥 使用外部 RAG 检索
    rag_context = retrieve_documents(query)

    print(f"\n✅ 检索完成")
    print("=" * 80)
    print(f"📊 上下文长度: {len(rag_context)} 字符")
    print("=" * 80 + "\n")

    return {
        "rag_context": rag_context,
        "messages": []
    }


def reasoning_node(state: AgentState) -> dict:
    """节点2: 推理节点"""
    print("\n🤔 [推理节点] LLM 思考中...")

    last_user_msg = state["messages"][-1].content

    # 构建系统提示词
    system_prompt = get_reasoning_prompt(
        user_question=last_user_msg,
        rag_context=state['rag_context']
    )

    llm_with_tools = chatModels.bind_tools(TOOLS)

    response = llm_with_tools.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_msg)
    ])

    # 处理非标准格式的 tool_call
    if not response.tool_calls and "<tool_call>" in response.content:
        print("⚠️  检测到非标准格式 tool_call，手动解析中...")

        try:
            pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
            matches = re.findall(pattern, response.content, re.DOTALL)

            if matches:
                tool_calls = []
                for i, match in enumerate(matches):
                    tool_data = json.loads(match)
                    tool_calls.append({
                        'name': tool_data.get('name'),
                        'args': tool_data.get('arguments', {}),
                        'id': f'manual_call_{i}',
                        'type': 'tool_call'
                    })

                response.tool_calls = tool_calls
                print(f"✓ 手动解析成功: {[tc['name'] for tc in tool_calls]}")

        except Exception as e:
            print(f"✗ 解析失败: {e}")

    if response.tool_calls:
        print(f"✓ LLM 决定调用工具: {[tc['name'] for tc in response.tool_calls]}")
        next_action = "call_tools"
    else:
        print("✓ LLM 直接回答（无需工具）")
        next_action = "end"

    return {
        "messages": [response],
        "next_action": next_action
    }


def tools_node(state: AgentState) -> dict:
    """节点3：执行工具调用"""
    print("\n🔧 [工具节点] 执行工具...")

    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"  ├─ 调用 {tool_name}({tool_args})")

        tool_func = TOOL_MAP[tool_name]
        result = tool_func.invoke(tool_args)

        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": tool_results,
        "next_action": "synthesize"
    }


def synthesize_node(state: AgentState) -> dict:
    """节点4：综合工具结果 + RAG 上下文生成最终答案"""
    print("\n📝 [综合节点] 生成最终答案...")

    # 找到最后一个用户消息
    user_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_question = msg.content

    # 收集所有工具返回
    tool_messages = [
        msg for msg in state["messages"]
        if isinstance(msg, ToolMessage)
    ]

    # 构建工具结果文本（不截断）
    tool_results_text = "\n\n".join([
        f"【工具返回 {i + 1}】\n{msg.content}"
        for i, msg in enumerate(tool_messages)
    ])

    # 构建最终提示词
    final_prompt = get_summary_prompt(
        user_question=user_question,
        rag_context=state['rag_context'],
        tool_results=tool_results_text
    )

    response = chatModels.invoke([
        SystemMessage(content=final_prompt)
    ])

    print("✓ 最终答案已生成")

    return {
        "messages": [response],
        "next_action": "end"
    }


# ===========================
# 4. 路由函数
# ===========================
def route_after_reasoning(state: AgentState) -> Literal["call_tools", "end"]:
    """决定推理后的路由"""
    return state.get("next_action", "end")


def route_after_tools(state: AgentState) -> Literal["synthesize"]:
    """工具调用后必定进入综合节点"""
    return "synthesize"


# ===========================
# 5. 构建图
# ===========================
def create_agent_graph():
    """创建 Agent 图"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("call_tools", tools_node)
    workflow.add_node("synthesize", synthesize_node)

    # 添加边
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "reasoning")

    # 推理后的条件路由
    workflow.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {
            "call_tools": "call_tools",
            "end": END
        }
    )

    # 工具调用后 → 综合节点
    workflow.add_conditional_edges(
        "call_tools",
        route_after_tools,
        {"synthesize": "synthesize"}
    )

    # 综合节点 → 结束
    workflow.add_edge("synthesize", END)

    # 编译（支持多轮对话）
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ===========================
# 6. 创建全局 Agent Graph
# ===========================
agent_graph = create_agent_graph()


# ===========================
# 7. 主函数（供外部调用）
# ===========================
def build_agent(query: str, thread_id: str = "default"):
    """
    执行 Agent（供 main.py 调用）

    Args:
        query: 用户问题
        thread_id: 会话 ID（用于多轮对话）

    Returns:
        最终答案字符串
    """
    import time

    print("\n" + "=" * 60)
    print(f"开始执行 Agent (会话ID: {thread_id})")
    print("=" * 60)

    start_time = time.time()

    # 执行图
    config = {"configurable": {"thread_id": thread_id}}

    for step, event in enumerate(agent_graph.stream(
            {
                "messages": [HumanMessage(content=query)],
                "rag_context": "",
                "next_action": ""
            },
            config=config,
            stream_mode="updates"
    ), start=1):
        print(f"\n{'─' * 60}")
        print(f"Step {step}: {list(event.keys())}")

    # 获取最终状态
    final_state = agent_graph.get_state(config)
    final_message = final_state.values["messages"][-1]

    print("\n" + "=" * 60)
    print("🎯 最终回答:")
    print("=" * 60)
    print(final_message.content)
    print(f"\n⏱️ 总耗时: {time.time() - start_time:.2f}秒")

    return final_message.content


# ===========================
# 8. 多轮对话示例（可选）
# ===========================
def chat_loop():
    """多轮对话模式（测试用）"""
    import uuid
    thread_id = str(uuid.uuid4())

    print("🤖 法律助手已启动（输入 'quit' 退出）\n")

    while True:
        query = input("👤 你: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break

        if not query:
            continue

        try:
            answer = build_agent(query, thread_id)
            print()
        except Exception as e:
            print(f"❌ 出错了: {e}\n")


# ===========================
# 9. 测试入口
# ===========================
if __name__ == "__main__":
    # 单次测试
    build_agent("个人独资企业名称与登记不符，罚款上限是多少？")

    # 多轮对话测试
    # chat_loop()
