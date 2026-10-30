from __future__ import annotations


FRONTEND_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FinAgent 金融决策工作台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7f8;
      --rail: #eef3f4;
      --panel: #ffffff;
      --panel-soft: #f8fafb;
      --line: #d8e1e4;
      --line-strong: #b9c8ce;
      --text: #172126;
      --text-strong: #071116;
      --muted: #617179;
      --muted-soft: #87959b;
      --accent: #0b766e;
      --accent-strong: #075d57;
      --accent-soft: #dff4f0;
      --indigo: #4057a8;
      --indigo-soft: #e8ecfb;
      --amber: #a15c08;
      --amber-soft: #fff3d7;
      --danger: #b42318;
      --danger-soft: #fee4e2;
      --success: #087443;
      --success-soft: #dcfce7;
      --code: #101826;
      --code-line: #273247;
      --radius: 8px;
      --radius-sm: 6px;
      --shadow-soft: 0 6px 18px rgba(20, 39, 32, .07);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-width: 320px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      background: linear-gradient(90deg, rgba(64, 87, 168, .06), transparent 34%), var(--bg);
      color: var(--text);
    }

    button, input, textarea, select {
      font: inherit;
      letter-spacing: 0;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 10;
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 12px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, .94);
      backdrop-filter: blur(18px);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-mark {
      width: 38px;
      height: 38px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      border: 1px solid #b8d5ce;
      border-radius: var(--radius);
      background: var(--accent-soft);
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .72);
    }

    .brand-mark span {
      width: 20px;
      height: 20px;
      display: block;
      border-left: 4px solid var(--accent);
      border-bottom: 4px solid var(--accent);
      border-radius: 2px;
      transform: skew(-8deg);
      box-shadow: 8px -7px 0 -3px var(--indigo);
    }

    .brand-copy {
      display: grid;
      gap: 3px;
      min-width: 0;
    }

    h1 {
      margin: 0;
      color: var(--text-strong);
      font-size: 20px;
      font-weight: 760;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .subtitle {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      white-space: nowrap;
      overflow-wrap: anywhere;
    }

    .status-pill {
      min-height: 34px;
      max-width: 100%;
      padding: 6px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      box-shadow: var(--shadow-soft);
    }

    .dot {
      width: 9px;
      height: 9px;
      flex: 0 0 auto;
      border-radius: 50%;
      background: var(--amber);
      box-shadow: 0 0 0 4px rgba(161, 92, 8, .12);
    }

    .dot.ok {
      background: var(--success);
      box-shadow: 0 0 0 4px rgba(8, 116, 67, .12);
    }

    .dot.not-ready {
      background: var(--amber);
      box-shadow: 0 0 0 4px rgba(161, 92, 8, .12);
    }

    .app-shell {
      display: grid;
      grid-template-columns: minmax(310px, 382px) minmax(0, 1fr);
      min-height: calc(100vh - 72px);
    }

    aside {
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff, var(--rail));
      padding: 22px;
    }

    section {
      min-width: 0;
      padding: 22px;
    }

    .side-header,
    .pane-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 16px;
    }

    .eyebrow {
      margin: 0 0 5px;
      color: var(--accent-strong);
      font-size: 11px;
      font-weight: 800;
      line-height: 1.2;
      text-transform: uppercase;
    }

    .panel-title {
      margin: 0;
      color: var(--text-strong);
      font-size: 16px;
      font-weight: 760;
      line-height: 1.35;
    }

    .panel-note {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }

    form {
      display: grid;
      gap: 14px;
    }

    .field-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    label {
      display: grid;
      gap: 7px;
      min-width: 0;
      color: var(--muted);
      font-size: 13px;
      font-weight: 680;
    }

    .field-text {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-width: 0;
    }

    .hint {
      color: var(--muted-soft);
      font-size: 12px;
      font-weight: 560;
      white-space: nowrap;
    }

    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
      color: var(--text);
      background: rgba(255, 255, 255, .9);
      outline: none;
      transition: border-color .16s ease, box-shadow .16s ease, background .16s ease;
    }

    input:hover, textarea:hover, select:hover {
      border-color: var(--line-strong);
      background: #ffffff;
    }

    input:focus, textarea:focus, select:focus {
      border-color: var(--accent);
      background: #ffffff;
      box-shadow: 0 0 0 3px rgba(11, 118, 110, .13);
    }

    select {
      min-height: 42px;
      cursor: pointer;
    }

    textarea {
      min-height: 76px;
      resize: vertical;
      line-height: 1.55;
    }

    button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-width: 0;
      border: 0;
      border-radius: var(--radius-sm);
      padding: 10px 13px;
      background: var(--accent);
      color: white;
      font-weight: 740;
      line-height: 1.25;
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(11, 118, 110, .18);
      transition: background .16s ease, transform .16s ease, box-shadow .16s ease, opacity .16s ease;
    }

    button:hover:not(:disabled) {
      background: var(--accent-strong);
      box-shadow: 0 10px 22px rgba(7, 93, 87, .22);
    }

    button:active:not(:disabled) {
      transform: translateY(1px);
    }

    button:disabled {
      opacity: .62;
      cursor: default;
      box-shadow: none;
    }

    .primary-action {
      min-height: 46px;
      margin-top: 2px;
      width: 100%;
    }

    .ghost {
      min-height: 36px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      box-shadow: none;
    }

    .ghost:hover:not(:disabled) {
      border-color: var(--line-strong);
      background: var(--panel-soft);
      box-shadow: none;
    }

    .session-actions {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      align-items: end;
    }

    .settings-stack {
      display: grid;
      gap: 14px;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
      max-width: 1180px;
      margin: 0 auto;
    }

    .chat-pane,
    .trace-pane {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, .78);
      box-shadow: var(--shadow-soft);
      overflow: hidden;
    }

    .chat-pane {
      display: grid;
      grid-template-rows: auto minmax(360px, calc(100vh - 306px)) auto;
    }

    .trace-pane {
      display: grid;
      grid-template-rows: auto auto minmax(180px, 320px);
    }

    .pane-header {
      margin: 0;
      padding: 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    .session-chip {
      max-width: 220px;
      min-height: 28px;
      padding: 5px 9px;
      border: 1px solid #b8d5ce;
      border-radius: 999px;
      color: var(--accent-strong);
      background: var(--accent-soft);
      font-size: 12px;
      font-weight: 760;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .chat-thread {
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
      padding: 18px;
      overflow: auto;
      background: linear-gradient(180deg, #fbfcfd, #f4f7f8);
    }

    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 118px;
      gap: 10px;
      align-items: end;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }

    .composer label {
      gap: 6px;
    }

    .composer textarea {
      min-height: 76px;
      max-height: 220px;
    }

    .composer .primary-action {
      margin: 0;
      min-height: 76px;
    }

    .message {
      display: grid;
      gap: 5px;
      max-width: min(760px, 88%);
      min-width: 0;
    }

    .message-user {
      align-self: end;
      justify-items: end;
    }

    .message-assistant,
    .message-status,
    .message-error {
      align-self: start;
      justify-items: start;
    }

    .message-meta {
      color: var(--muted-soft);
      font-size: 11px;
      font-weight: 740;
      line-height: 1.2;
    }

    .message-bubble {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 11px 13px;
      background: var(--panel);
      color: var(--text);
      line-height: 1.65;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      box-shadow: 0 1px 2px rgba(20, 39, 32, .05);
    }

    .message-user .message-bubble {
      border-color: #aad9d1;
      background: #e7f6f1;
      color: #08211d;
    }

    .message-status .message-bubble {
      border-color: #d7c189;
      background: var(--amber-soft);
      color: #553102;
    }

    .message-error .message-bubble {
      border-color: #f3b8b4;
      background: var(--danger-soft);
      color: var(--danger);
    }

    .trace-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }

    .trace-meta .status {
      min-width: 0;
      white-space: normal;
    }

    .steps {
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-height: 0;
      padding: 14px;
      overflow: auto;
    }

    .step {
      position: relative;
      flex: 0 0 auto;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(20, 39, 32, .05);
    }

    .step::before {
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      background: var(--accent);
    }

    .step.step-tool::before { background: var(--indigo); }
    .step.step-error::before { background: var(--danger); }
    .step.step-final::before { background: var(--success); }

    .step-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 12px;
      align-items: center;
      padding: 12px 14px 12px 17px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
      cursor: pointer;
    }

    .step-title {
      min-width: 0;
      color: var(--text-strong);
      font-size: 13px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }

    .step-toggle {
      min-height: 28px;
      padding: 4px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      box-shadow: none;
    }

    .step-toggle:hover:not(:disabled) {
      border-color: var(--line-strong);
      background: #ffffff;
      color: var(--text);
      box-shadow: none;
    }

    .step-summary {
      padding: 10px 14px 10px 17px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
      background: #ffffff;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 760;
      white-space: nowrap;
    }

    .badge.tool {
      background: var(--indigo-soft);
      color: var(--indigo);
    }

    .badge.error {
      background: var(--danger-soft);
      color: var(--danger);
    }

    .badge.final {
      background: var(--success-soft);
      color: var(--success);
    }

    .step-body {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 0;
      border-top: 1px solid var(--line);
    }

    .step-body[hidden] {
      display: none;
    }

    .io {
      min-width: 0;
      padding: 12px 14px 14px 17px;
    }

    .io + .io {
      border-left: 1px solid var(--line);
    }

    .io h2 {
      margin: 0 0 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      letter-spacing: 0;
    }

    pre {
      margin: 0;
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--code-line);
      border-radius: var(--radius-sm);
      padding: 11px;
      background: var(--code);
      color: #e8edf3;
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .empty {
      display: grid;
      place-items: center;
      min-height: 180px;
      border: 1px dashed var(--line-strong);
      border-radius: var(--radius);
      padding: 24px;
      color: var(--muted);
      background: rgba(255, 255, 255, .68);
      text-align: center;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .72);
    }

    @media (max-width: 1180px) {
      .chat-pane,
      .trace-pane {
        grid-template-rows: auto auto;
      }
    }

    @media (max-width: 980px) {
      .app-shell {
        grid-template-columns: 1fr;
      }

      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
    }

    @media (max-width: 760px) {
      header {
        position: static;
        align-items: flex-start;
        flex-direction: column;
        padding: 14px 18px;
      }

      .status-pill {
        width: 100%;
      }

      .status {
        white-space: normal;
      }

      aside,
      section {
        padding: 16px;
      }

      .field-grid,
      .session-actions,
      .composer,
      .step-body {
        grid-template-columns: 1fr;
      }

      .ghost {
        width: 100%;
      }

      .pane-header,
      .trace-meta {
        align-items: stretch;
        flex-direction: column;
      }

      .session-chip {
        max-width: 100%;
      }

      .message {
        max-width: 96%;
      }

      .io + .io {
        border-left: 0;
        border-top: 1px solid var(--line);
      }
    }
  </style>
</head>
<body>
  <header>
      <div class="brand">
      <div class="brand-mark" aria-hidden="true"><span></span></div>
      <div class="brand-copy">
        <h1>FinAgent 多轮金融对话</h1>
        <div class="subtitle">基于会话记忆、工具调用与风控审核的连续分析</div>
      </div>
    </div>
    <div class="status status-pill"><span id="readyDot" class="dot"></span><span id="readyText">正在检查</span></div>
  </header>
  <main class="app-shell">
    <aside>
      <div class="side-header">
        <div>
          <p class="eyebrow">Session</p>
          <h2 class="panel-title">会话设置</h2>
          <p class="panel-note">只保留多轮对话入口，可载入历史或新建会话</p>
        </div>
      </div>
      <div class="settings-stack">
        <div class="field-grid">
          <label><span class="field-text">租户 ID</span>
            <input id="tenantId" value="tenant_default" autocomplete="off">
          </label>
          <label><span class="field-text">用户 ID</span>
            <input id="userId" value="user_default" autocomplete="off">
          </label>
        </div>
        <label><span class="field-text">会话 ID</span>
          <span class="session-actions">
            <input id="sessionId" value="session_ops" autocomplete="off">
            <button id="loadButton" class="ghost" type="button">载入</button>
            <button id="newButton" class="ghost" type="button">新建</button>
          </span>
        </label>
        <label><span class="field-text">账户范围 <span class="hint">可选</span></span>
          <select id="accountScope">
            <option value="">不指定</option>
            <option value="default" selected>default</option>
          </select>
        </label>
        <label><span class="field-text">风险等级</span>
          <select id="riskLevel">
            <option value="conservative">保守</option>
            <option value="balanced" selected>均衡</option>
            <option value="aggressive">激进</option>
          </select>
        </label>
      </div>
    </aside>
    <section>
      <div class="workspace">
        <div class="chat-pane">
          <div class="pane-header">
            <div>
              <p class="eyebrow">Conversation</p>
              <h2 class="panel-title">多轮对话</h2>
              <p id="chatCount" class="panel-note">0 条消息</p>
            </div>
            <span id="sessionChip" class="session-chip">session_ops</span>
          </div>
          <div id="chatMessages" class="chat-thread">
            <div class="empty">当前会话暂无消息</div>
          </div>
          <form id="chatForm" class="composer">
            <label><span class="field-text">继续提问</span>
              <textarea id="message" placeholder="输入下一轮问题，例如：继续结合我的风险偏好分析。"></textarea>
            </label>
            <button id="runButton" class="primary-action" type="submit">发送</button>
          </form>
        </div>
        <div class="trace-pane">
          <div class="pane-header">
            <div>
              <p class="eyebrow">Trace</p>
              <h2 class="panel-title">执行轨迹</h2>
              <p class="panel-note">最近一轮任务事件、工具调用与风控审核</p>
            </div>
          </div>
          <div class="trace-meta">
            <span id="taskMeta" class="status">尚未开始任务</span>
            <button id="clearButton" class="ghost" type="button">清空轨迹</button>
          </div>
          <div id="steps" class="steps">
            <div class="empty">等待消息</div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById("chatForm");
    const steps = document.getElementById("steps");
    const chatMessages = document.getElementById("chatMessages");
    const runButton = document.getElementById("runButton");
    const clearButton = document.getElementById("clearButton");
    const loadButton = document.getElementById("loadButton");
    const newButton = document.getElementById("newButton");
    const taskMeta = document.getElementById("taskMeta");
    const readyDot = document.getElementById("readyDot");
    const readyText = document.getElementById("readyText");
    const sessionChip = document.getElementById("sessionChip");
    const chatCount = document.getElementById("chatCount");
    const messageInput = document.getElementById("message");

    const STORAGE_KEY = "finagent.console.settings.v1";
    const state = { taskId: null, sequence: 0, messageCount: 0, pendingAssistant: null, terminalReceived: false, abortController: null, tokenBuffer: "", requestId: 0, seenEvents: new Set(), renderedMessageIds: new Set(), tokenRenderTimer: null };
    const text = {
      emptyTrace: "等待消息",
      emptyChat: "当前会话暂无消息",
      noTask: "尚未开始任务",
      running: "分析中",
      run: "发送",
      assistantPending: "正在分析...",
      assistantName: "FinAgent",
      userName: "用户"
    };
    const eventLabels = {
      status: "状态",
      tool_result: "工具结果",
      risk_review: "风控审核",
      final: "最终结果",
      error: "错误"
    };
    const stageLabels = {
      received: "接收请求",
      memory_retrieval: "记忆检索",
      intent_routing: "意图路由",
      macro_agent: "宏观智能体",
      quant_agent: "量化智能体",
      market_agent: "市场智能体",
      report_agent: "研报智能体",
      memory_agent: "记忆智能体",
      router: "路由器",
      candidate_response: "候选答复",
      tool_result: "工具调用",
      risk_review: "风控审核",
      final: "最终结果",
      status: "状态更新",
      error: "错误",
      tool_progress: "工具进度",
      client_error: "客户端错误"
    };
    const statusLabels = {
      ready: "就绪",
      not_ready: "未就绪",
      ok: "正常",
      unavailable: "不可用"
    };

    function labelFor(map, value) {
      return map[value] || value || "-";
    }

    function asJson(value) {
      if (value === undefined) return "{}";
      return JSON.stringify(value, null, 2);
    }

    function authHeaders() {
      return {
        "X-Tenant-ID": document.getElementById("tenantId").value.trim(),
        "X-User-ID": document.getElementById("userId").value.trim()
      };
    }

    function currentSessionId() {
      return document.getElementById("sessionId").value.trim();
    }

    function saveClientSettings() {
      const settings = {
        tenantId: document.getElementById("tenantId").value.trim(),
        userId: document.getElementById("userId").value.trim(),
        sessionId: currentSessionId(),
        accountScope: document.getElementById("accountScope").value.trim()
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    }

    function restoreClientSettings() {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
        if (saved.tenantId) document.getElementById("tenantId").value = saved.tenantId;
        if (saved.userId) document.getElementById("userId").value = saved.userId;
        if (saved.sessionId) document.getElementById("sessionId").value = saved.sessionId;
        if (saved.accountScope !== undefined) document.getElementById("accountScope").value = saved.accountScope;
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
    }

    function renderAccountScopeOptions(items, preferredValue) {
      const select = document.getElementById("accountScope");
      const currentValue = preferredValue !== undefined ? preferredValue : select.value;
      const seen = new Set([""]);
      const options = [{ value: "", label: "不指定" }];
      for (const item of items || []) {
        const value = String(item.account_id || "").trim();
        if (!value || seen.has(value)) continue;
        seen.add(value);
        const currency = item.currency ? ` · ${item.currency}` : "";
        options.push({ value, label: `${value}${currency}` });
      }
      if (currentValue && !seen.has(currentValue)) {
        options.push({ value: currentValue, label: currentValue });
      }
      if (!seen.has("default")) {
        options.push({ value: "default", label: "default" });
      }
      select.innerHTML = "";
      for (const option of options) {
        const node = document.createElement("option");
        node.value = option.value;
        node.textContent = option.label;
        select.appendChild(node);
      }
      select.value = options.some(option => option.value === currentValue) ? currentValue : "default";
    }

    async function loadAccountScopes(preferredValue) {
      const select = document.getElementById("accountScope");
      const selectedValue = preferredValue !== undefined ? preferredValue : select.value;
      try {
        const response = await fetch("/v1/portfolio/accounts", { headers: authHeaders() });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        renderAccountScopeOptions(data.items || [], selectedValue);
      } catch {
        renderAccountScopeOptions([], selectedValue);
      }
    }

    function updateSessionChip() {
      sessionChip.textContent = currentSessionId() || "-";
    }

    function updateChatCount() {
      chatCount.textContent = `${state.messageCount} 条消息`;
    }

    function clearTrace() {
      steps.innerHTML = `<div class="empty">${text.emptyTrace}</div>`;
      taskMeta.textContent = text.noTask;
      state.taskId = null;
      state.sequence = 0;
      state.terminalReceived = false;
      state.seenEvents.clear();
    }

    function resetChat() {
      chatMessages.innerHTML = `<div class="empty">${text.emptyChat}</div>`;
      state.messageCount = 0;
      state.pendingAssistant = null;
      state.renderedMessageIds.clear();
      updateChatCount();
    }

    function stepTitle(event) {
      const payload = event.payload || {};
      const stage = payload.stage || event.event_type || "event";
      return `${event.sequence}. ${labelFor(stageLabels, stage)}`;
    }

    function badgeClassFor(event, payload) {
      if (payload.status === "error" || event.event_type === "error") return "badge error";
      if (event.event_type === "tool_result") return "badge tool";
      if (event.event_type === "final") return "badge final";
      return "badge";
    }

    function stepClassFor(event, payload) {
      if (payload.status === "error" || event.event_type === "error") return "step step-error";
      if (event.event_type === "tool_result") return "step step-tool";
      if (event.event_type === "final") return "step step-final";
      return "step";
    }

    function summarizeStep(event, input, output) {
      if (event.event_type === "final") {
        const answer = output.answer || (event.payload && event.payload.answer);
        return answer ? `最终回答：${String(answer).slice(0, 120)}` : "最终回答已生成";
      }
      if (event.event_type === "error") {
        return output.message || output.error || output.code || "请求失败";
      }
      if (event.event_type === "tool_result") {
        const toolName = output.tool_name || input.tool_name || "工具";
        return `${toolName}：${output.status || "已返回结果"}`;
      }
      if (event.event_type === "risk_review") {
        const verdict = output.verdict || (output.compliance_tool && output.compliance_tool.data && output.compliance_tool.data.verdict);
        return verdict ? `风控结论：${verdict}` : "风控审核完成";
      }
      return output.message || output.summary || output.intent || output.status || labelFor(stageLabels, (event.payload || {}).stage);
    }

    function appendStep(event) {
      if (steps.querySelector(".empty")) steps.innerHTML = "";
      const payload = event.payload || {};
      const input = payload.input !== undefined ? payload.input : payload;
      const output = payload.output !== undefined ? payload.output : payload;
      const step = document.createElement("article");
      step.className = stepClassFor(event, payload);
      const badgeClass = badgeClassFor(event, payload);
      const stepId = `step_body_${event.sequence || Date.now()}_${Math.random().toString(36).slice(2)}`;
      const expanded = event.event_type === "error";

      const head = document.createElement("div");
      head.className = "step-head";
      const title = document.createElement("div");
      title.className = "step-title";
      title.textContent = stepTitle(event);
      const badge = document.createElement("span");
      badge.className = badgeClass;
      badge.textContent = labelFor(eventLabels, event.event_type);
      const toggle = document.createElement("button");
      toggle.className = "step-toggle";
      toggle.type = "button";
      toggle.setAttribute("aria-controls", stepId);
      head.append(title, badge, toggle);

      const summary = document.createElement("div");
      summary.className = "step-summary";
      summary.textContent = summarizeStep(event, input, output);

      const body = document.createElement("div");
      body.id = stepId;
      body.className = "step-body";
      body.append(createTraceIo("输入", input), createTraceIo("输出", output));

      const setExpanded = nextExpanded => {
        body.hidden = !nextExpanded;
        toggle.setAttribute("aria-expanded", nextExpanded ? "true" : "false");
        toggle.textContent = nextExpanded ? "收起" : "展开";
      };
      const toggleStep = () => setExpanded(body.hidden);
      setExpanded(expanded);
      toggle.addEventListener("click", event => {
        event.stopPropagation();
        toggleStep();
      });
      head.addEventListener("click", toggleStep);
      step.append(head, summary, body);
      steps.appendChild(step);
      step.scrollIntoView({ block: "nearest" });
    }

    function createTraceIo(title, value) {
      const item = document.createElement("div");
      item.className = "io";
      const heading = document.createElement("h2");
      heading.textContent = title;
      const pre = document.createElement("pre");
      pre.textContent = asJson(value);
      item.append(heading, pre);
      return item;
    }

    function appendChatMessage(role, content, options = {}) {
      if (options.messageId && state.renderedMessageIds.has(options.messageId)) return null;
      if (chatMessages.querySelector(".empty")) chatMessages.innerHTML = "";
      const item = document.createElement("article");
      item.className = `message message-${role}`;
      const label = options.label || (role === "user" ? text.userName : text.assistantName);
      const timestamp = options.timestamp || "";
      const metaExtra = timestamp ? ` · ${timestamp}` : "";
      item.innerHTML = `
        <div class="message-meta">${escapeHtml(label)}${escapeHtml(metaExtra)}</div>
        <div class="message-bubble">${escapeHtml(content)}</div>`;
      chatMessages.appendChild(item);
      if (!options.transient) {
        if (options.messageId) state.renderedMessageIds.add(options.messageId);
        state.messageCount += 1;
        updateChatCount();
      }
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return item;
    }

    function updateChatMessage(item, role, content, label) {
      if (!item) return;
      item.className = `message message-${role}`;
      item.querySelector(".message-meta").textContent = label || text.assistantName;
      item.querySelector(".message-bubble").textContent = content;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function flushTokenRender() {
      if (state.tokenRenderTimer !== null) {
        clearTimeout(state.tokenRenderTimer);
        state.tokenRenderTimer = null;
      }
      if (state.pendingAssistant && state.tokenBuffer) {
        updateChatMessage(state.pendingAssistant, "assistant", state.tokenBuffer, text.assistantName);
      }
    }

    function renderFinal(event) {
      const payload = event.payload || {};
      const output = payload.output || payload;
      const answer = output.answer || payload.answer || "";
      if (!answer) {
        renderError("服务没有返回可展示的最终回答，请查看执行轨迹。");
        return;
      }
      const disclaimerText = output.risk_disclaimer || payload.risk_disclaimer || "";
      const disclaimer = disclaimerText ? `\\n\\n${disclaimerText}` : "";
      if (state.pendingAssistant) {
        updateChatMessage(state.pendingAssistant, "assistant", answer + disclaimer, text.assistantName);
        state.pendingAssistant = null;
        state.messageCount += 1;
        updateChatCount();
      } else {
        appendChatMessage("assistant", answer + disclaimer);
      }
    }

    function renderError(message) {
      if (state.pendingAssistant) {
        updateChatMessage(state.pendingAssistant, "error", message, "错误");
        state.pendingAssistant = null;
        state.messageCount += 1;
        updateChatCount();
      } else {
        appendChatMessage("error", message, { label: "错误" });
      }
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function readStream(response) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\\n\\n");
        buffer = chunks.pop() || "";
        for (const chunk of chunks) {
          const eventLine = chunk.split("\\n").find(item => item.startsWith("event: "));
          const sseEventType = eventLine ? eventLine.slice(7).trim() : "";
          const dataLine = chunk.split("\\n").find(item => item.startsWith("data: "));
          if (!dataLine) continue;
          const parsed = JSON.parse(dataLine.slice(6));
          let event;
          if (parsed && parsed.event_type) {
            event = parsed;
          } else if (sseEventType) {
            event = { ...parsed, event_type: sseEventType, sequence: parsed.sequence || state.sequence + 1, task_id: parsed.task_id || state.taskId || "" };
          } else {
            const code = parsed && parsed.code;
            const message = parsed && parsed.message;
            event = {
              event_type: "error",
              sequence: state.sequence + 1,
              task_id: state.taskId || "",
              event_id: "",
              checkpoint_id: "",
              payload: code && message
                ? { stage: "error", input: {}, output: { code, message }, status: "error" }
                : { stage: "error", input: {}, output: parsed || { message: "流式响应解析失败" }, status: "error" }
            };
          }
          if (state.seenEvents.has(event.event_id || `${event.sequence}:${event.event_type}`)) continue;
          if (event.event_id) state.seenEvents.add(event.event_id);
          else if (event.sequence) state.seenEvents.add(`${event.sequence}:${event.event_type}`);
          state.taskId = event.task_id || state.taskId;
          state.sequence = event.sequence || state.sequence;
          taskMeta.textContent = state.taskId ? `任务 ${state.taskId} · 序列 ${state.sequence}` : text.noTask;
          if (event.event_type === "token") {
            state.tokenBuffer += (event.payload && event.payload.token) || "";
            if (state.tokenRenderTimer === null) {
              state.tokenRenderTimer = setTimeout(flushTokenRender, 30);
            }
            continue;
          }
          flushTokenRender();
          appendStep(event);
          if (event.event_type === "final") {
            state.terminalReceived = true;
            state.tokenBuffer = "";
            renderFinal(event);
          }
          if (event.event_type === "error") {
            state.terminalReceived = true;
            state.tokenBuffer = "";
            const payload = event.payload || {};
            const output = payload.output !== undefined ? payload.output : payload;
            renderError(output.message || output.error || output.code || "请求失败");
          }
        }
      }
      flushTokenRender();
      if (state.pendingAssistant && !state.terminalReceived) {
        renderError("服务连接已结束，但没有收到最终回答，请稍后重试。");
      }
    }

    async function run(event) {
      event.preventDefault();
      const message = messageInput.value.trim();
      if (!message) return;
      if (state.abortController) {
        state.abortController.abort();
      }
      state.requestId += 1;
      const requestId = state.requestId;
      clearTrace();
      state.tokenBuffer = "";
      updateSessionChip();
      saveClientSettings();
      appendChatMessage("user", message);
      state.pendingAssistant = appendChatMessage("status", text.assistantPending, {
        label: text.assistantName,
        transient: true
      });
      runButton.disabled = true;
      runButton.textContent = text.running;
      state.abortController = new AbortController();
      try {
        const body = {
          session_id: currentSessionId(),
          message,
          account_scope: document.getElementById("accountScope").value.trim() || null,
          risk_level: document.getElementById("riskLevel").value
        };
        const response = await fetch("/v1/chat/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...authHeaders()
          },
          body: JSON.stringify(body),
          signal: state.abortController.signal
        });
        if (!response.ok || !response.body) throw new Error(`请求失败，HTTP 状态码 ${response.status}`);
        await readStream(response);
        if (state.requestId === requestId) {
          messageInput.value = "";
        }
      } catch (error) {
        if (state.requestId !== requestId) return;
        if (error && error.name === "AbortError") {
          state.terminalReceived = true;
          state.tokenBuffer = "";
          renderError("请求已取消。");
        } else {
          const msg = formatClientError(error);
          appendStep({
            event_type: "error",
            sequence: state.sequence + 1,
            payload: {
              stage: "client_error",
              input: {},
              output: { message: msg },
              status: "error"
            }
          });
          renderError(msg);
        }
      } finally {
        if (state.requestId !== requestId) return;
        runButton.disabled = false;
        runButton.textContent = text.run;
        state.abortController = null;
        messageInput.focus();
      }
    }

    async function loadHistory() {
      updateSessionChip();
      saveClientSettings();
      resetChat();
      clearTrace();
      const params = new URLSearchParams({ session_id: currentSessionId(), limit: "80" });
      try {
        const response = await fetch(`/v1/chat/history?${params.toString()}`, {
          headers: authHeaders()
        });
        if (!response.ok) throw new Error(`请求失败，HTTP 状态码 ${response.status}`);
        const data = await response.json();
        const messages = data.messages || [];
        for (const message of messages) {
          if (message.role === "user" || message.role === "assistant") {
            const stamp = message.created_at ? new Date(message.created_at).toLocaleString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
            appendChatMessage(message.role, message.content, { messageId: message.message_id, timestamp: stamp });
          }
        }
      } catch (error) {
        renderError(formatClientError(error));
      }
    }

    function formatClientError(error) {
      const raw = String(error || "请求失败");
      const message = raw.startsWith("Error: ") ? raw.slice(7) : raw;
      if (message.includes("Failed to fetch")) return "网络请求失败，请检查服务是否已启动。";
      if (message.includes("NetworkError")) return "网络请求失败，请检查服务连接。";
      return message || "请求失败";
    }

    function newSession() {
      const id = `session_${Date.now().toString(36)}`;
      document.getElementById("sessionId").value = id;
      updateSessionChip();
      saveClientSettings();
      resetChat();
      clearTrace();
      messageInput.focus();
    }

    async function loadReady() {
      try {
        const response = await fetch("/ready");
        const data = await response.json();
        readyText.textContent = `${labelFor(statusLabels, data.status)} · ${data.runtime_mode} · ${data.market_data_provider}`;
        readyDot.classList.toggle("ok", data.status === "ready");
        readyDot.classList.toggle("not-ready", data.status !== "ready");
      } catch {
        readyText.textContent = statusLabels.unavailable;
        readyDot.classList.remove("ok");
        readyDot.classList.add("not-ready");
      }
    }

    form.addEventListener("submit", run);
    clearButton.addEventListener("click", clearTrace);
    loadButton.addEventListener("click", loadHistory);
    newButton.addEventListener("click", newSession);
    for (const id of ["tenantId", "userId", "sessionId", "accountScope"]) {
      document.getElementById(id).addEventListener("change", () => {
        updateSessionChip();
        saveClientSettings();
        if (id === "tenantId" || id === "userId") loadAccountScopes("default");
      });
    }
    restoreClientSettings();
    updateSessionChip();
    loadAccountScopes();
    loadReady();
    loadHistory();
  </script>
</body>
</html>
"""
