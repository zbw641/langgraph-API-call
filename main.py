# -*- coding: utf-8 -*-
"""
@description: 基于 Flask 的产品质量智能问答系统
@author: AI Assistant
@version: 1.0
"""
import os
import uuid
from flask import Flask, render_template, request, jsonify
from loguru import logger
from source.graph import build_agent
from source.agent import initialize_rag  # 🔥 导入初始化函数

app = Flask(__name__)


class RagSystem:
    """RAG 系统封装类"""

    def __init__(self):
        """初始化系统"""
        logger.info("初始化 RAG 系统")
        # 不需要做任何事，agent.py 已经初始化
        logger.info("✅ RAG 系统初始化完成")

    def predict(self, query: str, thread_id: str = None) -> str:
        """
        执行预测，返回完整答案

        Args:
            query: 用户问题
            thread_id: 会话ID（可选）
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        try:
            logger.info(f"开始处理查询: {query}")

            # 调用 graph.build_agent
            answer = build_agent(query=query, thread_id=thread_id)

            logger.info(f"✅ 获取到答案，长度: {len(answer)}")
            return answer

        except Exception as e:
            logger.error(f"❌ 预测出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"系统错误: {str(e)}"


# 全局模型实例
model = None


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'success': False, 'error': '消息不能为空'})

        logger.info(f"收到问题: {message}")

        # 调用模型
        answer = model.predict(message)

        logger.info(f"返回答案: {answer[:50]}...")

        return jsonify({
            'success': True,
            'answer': answer
        })

    except Exception as e:
        logger.error(f"处理请求出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


def main():
    """主函数"""
    global model

    # 🔥 自动加载 data/ 目录下的所有文件
    data_dir = "data/"
    data_files = []

    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        # 获取所有文件（排除隐藏文件和目录）
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                data_files.append(file_path)

        logger.info(f"找到 {len(data_files)} 个数据文件: {data_files}")
    else:
        logger.warning(f"数据目录不存在: {data_dir}")
        data_files = []

    if not data_files:
        logger.error("❌ 未找到任何数据文件，请检查 data/ 目录")
        return

    # 🔥 初始化 RAG 检索器
    logger.info("正在初始化 RAG 检索器...")
    success = initialize_rag(
        corpus_files=data_files,  # 传入找到的所有文件
        chunk_size=250,  # 分块大小
        num_expand_context=0,  # 上下文扩展
        chunk_overlap=100,
        similarity_top_k=10,
        use_rerank=False  # 暂时禁用 Rerank
    )

    if not success:
        logger.error("❌ RAG 初始化失败，退出")
        return

    # 初始化模型
    logger.info("正在初始化模型...")
    model = RagSystem()

    # 预热
    logger.info("正在预热模型...")
    try:
        test_answer = model.predict("测试")
        logger.info(f"✅ 模型预热完成，测试答案: {test_answer[:50]}...")
    except Exception as e:
        logger.warning(f"⚠️  模型预热失败: {e}")

    # 启动 Flask 服务
    host = "0.0.0.0"
    port = 6050

    logger.info(f"🚀 启动服务: http://{host}:{port}")
    logger.info(f"📁 已加载 {len(data_files)} 个数据文件")

    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()