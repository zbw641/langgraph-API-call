"""测试入口"""
from source.agent import build_vector_store
from source.graph import create_agent_graph
from langchain_core.messages import HumanMessage
import time


def test_single_query(file: str, query: str):
    """单次问答测试"""
    print("\n" + "=" * 60)
    print("🚀 单次问答模式")
    print("=" * 60)

    # 1. 初始化向量库
    build_vector_store(file)

    # 2. 创建 Agent 图
    agent_graph = create_agent_graph()

    # 3. 执行查询
    start_time = time.time()
    config = {"configurable": {"thread_id": "test_single"}}

    print(f"\n📝 问题: {query}\n")

    for step, event in enumerate(agent_graph.stream(
            {
                "messages": [HumanMessage(content=query)],
                "rag_context": "",
                "next_action": ""
            },
            config=config,
            stream_mode="updates"
    ), start=1):
        node_name = list(event.keys())[0]
        print(f"Step {step}: 执行节点 [{node_name}]")

    # 4. 获取最终答案
    final_state = agent_graph.get_state(config)
    final_message = final_state.values["messages"][-1]

    print("\n" + "=" * 60)
    print("🎯 最终回答:")
    print("=" * 60)
    print(final_message.content)
    print(f"\n⏱️ 总耗时: {time.time() - start_time:.2f}秒\n")


def test_multi_turn(file: str):
    """多轮对话测试"""
    import uuid

    print("\n" + "=" * 60)
    print("🤖 多轮对话模式（输入 'quit' 退出）")
    print("=" * 60)

    # 1. 初始化向量库
    build_vector_store(file)

    # 2. 创建 Agent 图
    agent_graph = create_agent_graph()

    # 3. 生成会话ID（保持上下文）
    thread_id = str(uuid.uuid4())

    while True:
        query = input("\n👤 你: ").strip()
        if query.lower() in ['quit', 'exit', 'q', '退出']:
            print("👋 再见！")
            break

        if not query:
            continue

        # 执行查询
        start_time = time.time()
        config = {"configurable": {"thread_id": thread_id}}

        for event in agent_graph.stream(
                {
                    "messages": [HumanMessage(content=query)],
                    "rag_context": "",
                    "next_action": ""
                },
                config=config,
                stream_mode="updates"
        ):
            pass  # 静默执行

        # 获取答案
        final_state = agent_graph.get_state(config)
        final_message = final_state.values["messages"][-1]

        print(f"\n🤖 助手: {final_message.content}")
        print(f"   (耗时: {time.time() - start_time:.2f}秒)")


if __name__ == "__main__":
    # ===== 测试1: 单次问答 =====
    if __name__ == "__main__":
        questions = [
            "关于统一计量制度的命令是什么时候发布的",
            "法定计量单位包括哪些内容？",
            "这个命令的发布日期是多少？"
        ]

        for q in questions:
            test_single_query("data/国务院关于在我国统一实行法定计量单位的命令.txt", q)

    # ===== 测试2: 多轮对话（取消注释使用） =====
    # test_multi_turn("国务院关于在我国统一实行法定计量单位的命令.txt")

    # ===== 测试3: 批量测试多个问题 =====
    # questions = [
    #     "关于统一计量制度的命令是什么时候发布的",
    #     "法定计量单位包括哪些内容？",
    #     "这个命令由哪个部门发布？"
    # ]
    # for q in questions:
    #     test_single_query("国务院关于在我国统一实行法定计量单位的命令.txt", q)