"""团团校园问答：python -m streamlit run app.py"""
from html import escape
from datetime import datetime

import streamlit as st
from config import HISTORY_ROUNDS, MAX_QUESTION_CHARS, missing_api_keys
from campusmind.qa_service import ask, STAGE_LABELS
from campusmind.session_manager import ChatSession

WELCOME = "同学，你好呀！我叫团团，是一个校园规章制度问答系统，请问有什么可以帮助你的吗？"
MASCOT = '''<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="团团">
<circle cx="60" cy="60" r="59" fill="#e3f2df"/><path d="M63 31Q57 12 78 14Q81 30 63 31" fill="#6aab77"/>
<rect x="24" y="48" width="75" height="48" rx="20" fill="#e8c479"/>
<path d="M23 71Q21 31 60 32Q99 31 97 71Q96 102 60 102Q25 102 23 71" fill="#fff9e8" stroke="#d0dfc0" stroke-width="2"/>
<circle cx="46" cy="64" r="4" fill="#345846"/><circle cx="75" cy="64" r="4" fill="#345846"/>
<ellipse cx="37" cy="75" rx="7" ry="4" fill="#efbdab"/><ellipse cx="84" cy="75" rx="7" ry="4" fill="#efbdab"/>
<path d="M54 77Q61 84 68 77" fill="none" stroke="#345846" stroke-width="3" stroke-linecap="round"/>
<path d="M44 94L60 99L77 94L77 109L60 113L44 108Z" fill="#76a78c"/>
<path d="M60 100V112" stroke="#fff9e8" stroke-width="2"/></svg>'''


def initialize_session():
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(history_rounds=HISTORY_ROUNDS)
    st.session_state.setdefault("display_turns", [])
    st.session_state.setdefault("confirm_new", False)
    return st.session_state.chat_session


def handle_new_chat():
    st.session_state.chat_session = ChatSession(history_rounds=HISTORY_ROUNDS)
    st.session_state.display_turns = []
    st.session_state.confirm_new = False
    st.session_state.history_search = ""
    st.session_state.question_input = ""


def request_new_chat():
    if st.session_state.display_turns:
        st.session_state.confirm_new = True
    else:
        handle_new_chat()


def fill_question(question):
    st.session_state.question_input = question


def apply_styles():
    st.markdown('''<style>
    :root {--ink:#294c40;--muted:#6b8076;--green:#34745b;--line:#e0e9dc;}
    .stApp {background:#faf9f3;color:var(--ink);}
    [data-testid="stHeader"] {background:rgba(250,249,243,.94);}
    [data-testid="stSidebar"] {background:#edf4e9;border-right:1px solid #dce7d7;}
    [data-testid="stSidebar"] * {color:var(--ink);}
    [data-testid="stSidebar"] {min-width:285px;max-width:310px;}
    .stMainBlockContainer {max-width:1000px;padding-top:2.5rem;padding-bottom:2rem;}
    h1,h2,h3 {color:var(--ink)!important;letter-spacing:-.035em;}
    .brand {display:flex;align-items:center;gap:12px;margin:4px 0 24px;}
    .brand svg {width:58px;height:58px;flex-shrink:0;}
    .brand strong {font-size:23px;letter-spacing:-1px;}
    .brand small {display:block;font-size:12px;color:var(--muted);margin-top:4px;}
    .eyebrow {font-size:11px;font-weight:700;letter-spacing:2px;color:#5b8068;margin-bottom:12px;}
    .hero {display:flex;align-items:center;justify-content:space-between;gap:20px;padding:28px 32px;
      background:linear-gradient(110deg,#edf4e4,#fff6de);border:1px solid #e6ead7;border-radius:24px;margin:8px 0 22px;}
    .hero h1 {font-size:31px;margin:0;padding:0 0 12px;line-height:1.4;}
    .hero p {font-size:14px;color:#667c6c;margin:0;line-height:1.8;}
    .hero svg {width:115px;min-width:90px;transform:rotate(5deg);}
    .history-label {font-size:12px;color:#6b8076;margin:20px 0 12px;}
    .history-link {display:block;text-decoration:none!important;color:#365746!important;
      padding:12px 14px;border-radius:12px;margin:5px 0;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .history-link:hover,.history-link:focus {background:#dcebd7;outline:2px solid #96b49b;}
    .history-link.latest {background:#dfeeda;font-weight:600;}
    .side-note {font-size:12px;color:#718173;line-height:1.8;margin-top:24px;}
    [data-testid="stChatMessage"] {background:#fffef9;border:1px solid var(--line);border-radius:20px;padding:22px;margin-bottom:18px;}
    [data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li {line-height:1.85;color:var(--ink);}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {background:#edf4e9;border-color:#dce8d6;}
    .stButton>button,.stDownloadButton>button {border-radius:12px;border:1px solid #cbdcc8;background:#fffef9;color:#345745;min-height:44px;}
    .stButton>button:hover,.stDownloadButton>button:hover {border-color:#78a482;background:#e5f0de;color:#28533e;}
    .stButton>button[kind="primary"] {background:#3f765b;color:white;border-color:#3f765b;}
    [data-testid="stTextInput"] input {background:#fffef9;color:#294c40;}
    [data-testid="stExpander"] {background:#f7f9f1;border-color:#e0e9dc;border-radius:12px;color:var(--ink);}
    [data-testid="stBottom"],[data-testid="stBottom"]>div {background:#faf9f3;}
    [data-testid="stChatInput"] {border:1px solid #a9c4a3;border-radius:20px;background:#fffef9;box-shadow:0 5px 24px #426c4910;}
    [data-testid="stChatInput"] textarea {color:#294c40;background:#fffef9;}
    .turn-anchor {scroll-margin-top:90px;}
    @media(max-width:700px){.hero{padding:22px 18px}.hero h1{font-size:23px}.hero svg{width:76px;min-width:60px}.stMainBlockContainer{padding-top:1.5rem}[data-testid="stChatMessage"]{padding:15px}}
    @media(prefers-reduced-motion: no-preference){html{scroll-behavior:smooth}}
    </style>''', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(f'<div class="brand">{MASCOT}<div><strong>团团 · CampusMind</strong><small>你的校园规章小助手</small></div></div>', unsafe_allow_html=True)
        st.button("＋ 开启新对话", key="new_chat", type="primary", use_container_width=True, on_click=request_new_chat)
        if st.session_state.confirm_new:
            st.warning("开启新对话会清空本次聊天记录，确定吗？")
            yes, no = st.columns(2)
            yes.button("确定开启", key="confirm_clear", on_click=handle_new_chat)
            if no.button("再聊一会", key="cancel_clear"):
                st.session_state.confirm_new = False
                st.rerun()
        search = st.text_input("搜索历史提问", placeholder="找找之前问过什么…", key="history_search")
        turns = st.session_state.display_turns
        st.markdown(f'<div class="history-label">当前对话 · {len(turns)} 次提问</div>', unsafe_allow_html=True)
        matches = [(i, t) for i, t in enumerate(turns) if search.casefold() in t['question'].casefold()]
        if not turns:
            st.caption("还没有提问，团团在等你呀 🌱")
        elif not matches:
            st.caption("没有找到相关提问，换个词试试～")
        for i, turn in reversed(matches):
            label = escape(turn['question'], quote=True)
            style = ' latest' if i == len(turns) - 1 else ''
            st.markdown(f'<a class="history-link{style}" href="#turn-{i}" target="_self" title="{label}">{i+1:02d} · {label}</a>', unsafe_allow_html=True)
        if turns:
            transcript = '\n\n'.join(f"同学：{t['question']}\n\n团团：{t['result'].get('answer') or t['result'].get('message', '')}" for t in turns)
            st.download_button("↓ 保存聊天文字", transcript, file_name="团团聊天记录.txt", mime="text/plain", use_container_width=True)
        st.markdown('<div class="side-note">📖 每一份依据，都可以回到原文核对。<br>点击历史提问可定位对应回答。<br><br>记录仅保留在当前浏览器会话中，刷新或关闭页面后可能丢失。</div>', unsafe_allow_html=True)


def render_sources(sources):
    if sources:
        st.caption(f"📖 看看原文依据 · {len(sources)} 条")
    for source in sources:
        start = source.get("page_start") or source.get("page_number")
        end = source.get("page_end") or start
        pages = "页码未标注" if start is None else f"PDF 第 {start} 页" if start == end else f"PDF 第 {start}–{end} 页"
        with st.expander(f"[{source['citation_id']}] {source.get('file_name', '参考资料')} · {pages}"):
            if source.get("section_title"):
                st.caption(source["section_title"])
            st.text(source.get("content", "暂无原文。"))


def render_result(result):
    status = result.get("status", "error")
    if status == "answered":
        st.markdown(result["answer"])
    elif status == "insufficient":
        st.info("同学，团团目前还没找到足够的资料，暂时不能确定。你可以补充具体事项，我再帮你查一查～")
        if result.get("insufficient_reason") == "citation_unverified":
            st.caption("这次生成的回答没有通过引用核对，团团先不给出未核实的结论。可以换个问法或补充具体事项再试。")
    elif status == "needs_clarification":
        st.info("同学，团团需要再确认一下：" + (result.get("answer") or result.get("message", "请补充具体事项。")))
        st.caption("把具体类别或事项补充到完整问题里，再发给团团吧～")
    else:
        st.error(result.get("message") or "哎呀，这次查询没有完成，请稍后再试一下～")
    render_sources(result.get("sources", []))
    timings = result.get("timings_ms", {})
    if timings:
        with st.expander("查看处理详情"):
            for stage, elapsed in timings.items():
                if elapsed is not None:
                    st.text(f"{'总计' if stage == 'total' else STAGE_LABELS.get(stage, stage)}：{elapsed / 1000:.2f} 秒")


def render_chat_history():
    with st.chat_message("assistant", avatar="🌱"):
        st.caption("团团 · 校园规章小助手")
        st.markdown(WELCOME)
    for index, turn in enumerate(st.session_state.display_turns):
        st.markdown(f'<div id="turn-{index}" class="turn-anchor"></div>', unsafe_allow_html=True)
        with st.chat_message("user", avatar="🎒"):
            st.markdown(turn["question"])
        with st.chat_message("assistant", avatar="🌱"):
            st.caption("团团" + (f" · {turn['time']}" if turn.get('time') else ''))
            render_result(turn["result"])


def handle_question(question):
    if not question.strip():
        return
    with st.chat_message("user", avatar="🎒"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🌱"):
        with st.status("团团正在翻阅学生手册，稍等一下～", expanded=False) as progress:
            try:
                result = ask(question, st.session_state.chat_session)
            except Exception:
                result = {"status": "error", "message": "哎呀，这次查询没有完成，请稍后再试一下～", "sources": []}
            progress.update(label="这次查询没有完成" if result['status'] == 'error' else "团团整理好啦", state="error" if result['status'] == 'error' else "complete")
    st.session_state.display_turns.append({"question": question, "result": result, "time": datetime.now().strftime("%H:%M")})
    st.rerun()  # 更新左侧记录；问答调用只发生于提交，不在重绘时再次执行。


def main():
    st.set_page_config(page_title="团团 · CampusMind", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")
    initialize_session()
    apply_styles()
    render_sidebar()
    st.markdown(f'<div class="hero"><div><div class="eyebrow">CAMPUSMIND · CAMPUS COMPANION</div><h1>校园里的小问号，<br>和团团一起找答案。</h1><p>奖学金 · 学籍 · 考试规章<br>从学生手册出发，让每一次解答有据可查。</p></div>{MASCOT}</div>', unsafe_allow_html=True)
    missing = missing_api_keys()
    if missing:
        st.warning("服务还差一点准备：请在项目根目录 .env 中填写 " + "、".join(missing) + "，然后重启应用。")
    render_chat_history()
    if not st.session_state.display_turns:
        st.caption("不知道从哪里问起？点一个问题，修改后发给团团 ↓")
        examples = [("🎓 奖学金申请", "国家奖学金申请需要满足哪些条件？"), ("📚 缓考规定", "申请缓考有什么要求？"), ("📝 申请材料", "国家励志奖学金需要哪些申请材料？")]
        for index, (column, (label, question)) in enumerate(zip(st.columns(3), examples)):
            column.button(label, key=f"example_{index}", on_click=fill_question, args=(question,), disabled=bool(missing), use_container_width=True)
    st.caption("🌿 团团依据当前资料回答，具体规定请以原文为准。")
    question = st.chat_input("同学，有什么校园规定想问团团？", key="question_input", max_chars=MAX_QUESTION_CHARS, disabled=bool(missing))
    if question:
        handle_question(question)


if __name__ == "__main__":
    main()
