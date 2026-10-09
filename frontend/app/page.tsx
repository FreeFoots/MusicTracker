"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Metric = {
  name: string;
  label: string;
  status: string;
  value: number | null;
  display: string | null;
  reason: string | null;
  collected_at: string;
  source_url: string;
  observation_id: string;
  comparison?: {
    delta: number;
    label: string;
    from: string;
    to: string;
  } | null;
};
type Song = {
  platform: string;
  platform_id: string;
  title: string;
  artists: string[];
  album: string;
  url: string;
  version: string;
  relation: string;
  relation_reason: string;
  origin_status: string;
  metrics: Metric[];
  fetched_at: string;
  release_date?: string;
  entity_id?: string;
};
type Group = {
  title: string;
  artists: string[];
  version: string;
  relation: string;
  origin_status: string;
  songs: Song[];
};
type Report = {
  title?: string;
  summary?: string;
  origin?: string;
  groups?: Group[];
  songs?: Song[];
  failures?: Record<string, { code: string; message: string }>;
  limitations?: string[];
  original?: { artists: string[]; url: string; basis: string } | null;
  window?: { start: string; end: string };
  selection_basis?: string;
  reused_from_query_id?: string;
  output?: unknown;
  tool?: string;
  execution_mode?: string;
  model?: string;
  model_error?: string;
  agent_usage?: { rounds: number; tool_calls: number; budget_tokens: number };
  answer?: {
    status: string;
    reason?: string;
    blocks: {
      heading: string;
      lines: { text: string; observation_id?: string; collected_at?: string }[];
    }[];
    citations: {
      observation_id: string;
      source_url: string;
      collected_at: string;
    }[];
    next_action?: string;
  };
};
type Conversation = {
  conversation_id: string;
  title: string;
  turns: {
    query_id: string;
    input: string;
    execution_status: string;
    question?: string;
    report: Report;
  }[];
};
type AgentStatus = {
  configured: boolean;
  model: string;
  max_rounds: number;
  max_tools: number;
  privacy: string;
};
type Query = {
  query_id: string;
  last_event_id: number;
  conversation_id: string | null;
  request: { execution_mode?: string };
  agent_steps: { step_id: number; kind: string; payload: object }[];
  input: string;
  execution_status: string;
  persistence_status: string;
  created_at: string;
  plan: { title?: string };
  report: Report;
  checkpoint: {
    answer_only_retry?: boolean;
    question?: string;
    choices?: {
      key: string;
      artists: string[];
      title: string;
      album: string;
      url: string;
    }[];
  };
  results: Song[];
  error?: string;
  tool_calls: {
    tool: string;
    status: string;
    duration_ms: number;
    error_code?: string;
  }[];
};
type History = {
  query_id: string;
  input: string;
  execution_status: string;
  created_at: string;
  result_count: number;
};
type Tool = {
  name: string;
  description: string;
  version: string;
  input_schema: object;
};
const API = "/api/backend/api/v1";
const terminal = new Set([
  "completed",
  "partial",
  "failed",
  "cancelled",
  "waiting_input",
]);
const platformNames: Record<string, string> = {
  qq: "QQ 音乐",
  netease: "网易云音乐",
};
const reasonNames: Record<string, string> = {
  timeout: "平台请求超时",
  upstream_error: "平台入口返回异常",
  authorization_required: "平台要求授权",
  rate_limited: "平台限流",
  schema_changed: "平台结构变化",
  field_missing: "平台未返回此字段",
  field_missing_or_invalid: "字段缺失或无效",
};
const statuses: Record<string, string> = {
  queued: "等待执行",
  resolving: "解析与发现",
  fetching: "获取平台数据",
  normalizing: "整理数据",
  completed: "查询完成",
  partial: "部分完成",
  failed: "获取失败",
  cancelled: "已取消",
  waiting_input: "需要确认作品",
};
const versions: Record<string, string> = {
  studio: "录音室",
  live: "现场 Live",
  remix: "Remix",
  instrumental: "伴奏",
  excerpt: "片段",
};
function date(text: string, withTime = false) {
  return new Date(text).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  });
}
function number(value: number | null) {
  return value === null ? "—" : new Intl.NumberFormat("zh-CN").format(value);
}
async function api(path: string, body?: object, key?: string) {
  const response = await fetch(API + path, {
    method: body ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
      ...(key ? { "Idempotency-Key": key } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "请求参数无效，请检查输入与查询范围",
    );
  return data;
}
function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 4 4" />
      </>
    ),
    layers: (
      <>
        <path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5" />
      </>
    ),
    history: (
      <>
        <path d="M3 11a9 9 0 1 1 2.5 7M3 4v7h7" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    tool: (
      <>
        <path d="m14 5 5 5M3 21l7-7M8 3l13 13-5 5L3 8Z" />
      </>
    ),
    arrow: (
      <>
        <path d="M4 12h16m-6-6 6 6-6 6" />
      </>
    ),
    refresh: (
      <>
        <path d="M20 8a8 8 0 0 0-14-3L3 8m0-5v5h5M4 16a8 8 0 0 0 14 3l3-3m0 5v-5h-5" />
      </>
    ),
    link: (
      <>
        <path d="M14 3h7v7M21 3l-9 9" />
        <path d="M10 3H3v18h18v-7" />
      </>
    ),
    download: (
      <>
        <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />
      </>
    ),
    check: <path d="m5 12 4 4 10-10" />,
    music: (
      <>
        <path d="M9 18V5l11-2v12M9 8l11-2" />
        <ellipse cx="5.5" cy="18" rx="3.5" ry="3" />
        <ellipse cx="16.5" cy="15" rx="3.5" ry="3" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.music}
    </svg>
  );
}

export default function Workspace() {
  const [view, setView] = useState("search");
  const [input, setInput] = useState("我希望获取《空心》的近期数据");
  const [days, setDays] = useState(0);
  const [platforms, setPlatforms] = useState(["qq", "netease"]);
  const [mode, setMode] = useState("live");
  const [executionMode, setExecutionMode] = useState("agent");
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [followup, setFollowup] = useState("");
  const [connectionResult, setConnectionResult] = useState("");
  const [history, setHistory] = useState<History[]>([]);
  const [stats, setStats] = useState({
    queries: 0,
    entities: 0,
    observations: 0,
    database: "",
  });
  const [connected, setConnected] = useState(false);
  const [active, setActive] = useState<Query | null>(null);
  const [events, setEvents] = useState<
    { type: string; payload?: { message?: string; tool?: string } }[]
  >([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tools, setTools] = useState<Tool[]>([]);
  const [library, setLibrary] = useState<Song[]>([]);
  const [toolName, setToolName] = useState("netease.search_songs");
  const [toolInput, setToolInput] = useState("空心");
  const [toolOutput, setToolOutput] = useState<unknown>(null);
  const stream = useRef<EventSource | null>(null);
  const activeId = useRef<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const [h, s] = await Promise.all([api("/queries"), api("/stats")]);
      setHistory(h.queries);
      setStats(s);
      setConnected(true);
      setAgentStatus(await api("/agent/status"));
    } catch {
      setConnected(false);
    }
  }, []);
  useEffect(() => {
    reload();
    api("/tools")
      .then((r) => setTools(r.tools))
      .catch(() => {});
    return () => stream.current?.close();
  }, [reload]);

  const loadConversation = useCallback(async (cid: string | null) => {
    setConversation(cid ? await api("/conversations/" + cid) : null);
  }, []);

  const open = useCallback(
    async (id: string) => {
      stream.current?.close();
      activeId.current = id;
      setEvents([]);
      setView("search");
      window.history.replaceState(null, "", `?query=${encodeURIComponent(id)}`);
      const query: Query = await api("/queries/" + id);
      setActive(query);
      await loadConversation(query.conversation_id);
      setBusy(!terminal.has(query.execution_status));
      if (terminal.has(query.execution_status)) {
        reload();
        return;
      }
      const es = new EventSource(
        `${API}/queries/${id}/events?after=${query.last_event_id || 0}`,
      );
      stream.current = es;
      es.onmessage = async (message) => {
        if (activeId.current !== id) {
          es.close();
          return;
        }
        const event = JSON.parse(message.data);
        setEvents((prior) => [...prior.slice(-24), event]);
        if (
          event.type === "finished" ||
          event.type === "needs_review" ||
          event.type === "stream_end"
        ) {
          es.close();
          const next = await api("/queries/" + id);
          if (activeId.current === id) {
            setActive(next);
            setBusy(false);
            await loadConversation(next.conversation_id);
          }
          reload();
        } else if (event.type === "database_unavailable") {
          es.close();
          setError("数据库暂不可用，保存状态未确认");
          setBusy(false);
        } else if (event.type === "platform_saved" || event.type === "plan") {
          const next = await api("/queries/" + id);
          if (activeId.current === id) setActive(next);
        }
      };
      es.onerror = () => {
        setError("进度连接中断，后台查询不会因此取消；可从历史重新打开");
      };
    },
    [reload, loadConversation],
  );

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("query");
    if (id && /^[a-f0-9]{32}$/.test(id))
      open(id).catch((e) => setError(e.message));
  }, [open]);

  async function submit() {
    if (busy || !input.trim() || !platforms.length) return;
    if (executionMode === "agent" && !agentStatus?.configured) {
      setError(
        "请先在本机 .env 配置 DEEPSEEK_API_KEY 并重启；也可明确选择规则模式。",
      );
      return;
    }
    setBusy(true);
    setError("");
    setActive(null);
    try {
      const body = {
        input,
        ...(days ? { days } : {}),
        platforms,
        max_versions: 4,
        mode,
        execution_mode: executionMode,
        ...(mode === "prefer_cache" ? { max_age_seconds: 3600 } : {}),
      };
      const cid =
        executionMode === "agent"
          ? (await api("/conversations", {})).conversation_id
          : null;
      const result = await api(
        cid ? `/conversations/${cid}/messages` : "/queries",
        body,
        crypto.randomUUID(),
      );
      await open(result.query_id);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  async function sendFollowup() {
    if (!active || !followup.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const next = await api(
        `/queries/${active.query_id}/follow-up`,
        {
          input: followup,
          execution_mode: "agent",
          platforms,
          ...(days ? { days } : {}),
          mode: "live",
          max_versions: 4,
        },
        crypto.randomUUID(),
      );
      setFollowup("");
      await open(next.query_id);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  async function checkConnection() {
    setConnectionResult("正在请求 DeepSeek，结果将保存到历史…");
    try {
      const result = await api("/agent/check", {});
      setConnectionResult(
        result.connected
          ? `真实连接成功 · ${result.model}；音乐任务尚需评测。`
          : result.error,
      );
      reload();
    } catch (e) {
      setConnectionResult((e as Error).message);
    }
  }
  async function action(kind: string, body: object = {}) {
    if (!active) return;
    setError("");
    try {
      const result = await api(`/queries/${active.query_id}/${kind}`, body);
      await open(result.query_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function changeView(next: string) {
    setView(next);
    setError("");
    if (next === "library")
      try {
        setLibrary((await api("/library")).songs);
      } catch (e) {
        setError((e as Error).message);
      }
    if (next === "history") reload();
  }
  async function invoke() {
    setBusy(true);
    setError("");
    try {
      const args = toolName.endsWith("search_songs")
        ? { keyword: toolInput, limit: 12 }
        : {
            ids: toolInput
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean),
          };
      const output = await api("/tools/invoke", {
        name: toolName,
        arguments: args,
      });
      setToolOutput(output);
      reload();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const report = active?.report;
  const shownGroups = report?.groups || [];
  const lastEvent = events.filter((e) => e.payload?.message).at(-1)
    ?.payload?.message;

  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="MusicTracker 首页">
          <span className="brand-mark">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>
            Music<span className="muted">Tracker</span>
          </span>
        </a>
        <div className="workspace-label">
          <span className="avatar">M</span>
          <div>
            音乐研究工作区<small>LOCAL WORKSPACE</small>
          </div>
          <span className="tiny-dot" />
        </div>
        <div className="nav-caption">工作台</div>
        <nav>
          {[
            ["search", "search", "全景查询"],
            ["history", "history", "查询历史"],
            ["library", "layers", "音乐资料库"],
            ["tools", "tool", "平台工具"],
          ].map(([key, icon, text]) => (
            <button
              key={key}
              className={view === key ? "nav-item selected" : "nav-item"}
              onClick={() => changeView(key)}
            >
              <Icon name={icon} />
              {text}
              {key === "history" && (
                <span className="nav-count">{stats.queries}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-history">
          <div className="nav-caption">
            最近查询 <span>↗</span>
          </div>
          {history.slice(0, 5).map((h) => (
            <button
              key={h.query_id}
              onClick={() => open(h.query_id).catch((e) => setError(e.message))}
            >
              <span className="history-dot" />
              <span>{h.input}</span>
            </button>
          ))}
          {!history.length && <p>从第一首歌开始积累。</p>}
        </div>
        <div className="sidebar-bottom">
          <div className="storage-card">
            <Icon name="layers" />
            <div>
              查询即归档<small>数据与来源，自动留存</small>
            </div>
          </div>
          <div className="local-note">
            <span className={connected ? "tiny-dot" : "tiny-dot offline"} />
            {connected ? `${stats.database} · 本地连接` : "等待后端连接"}
            <span>v0.6.1</span>
          </div>
        </div>
      </aside>
      <div className="main-column">
        <header className="topbar">
          <div>
            <span className="muted">工作区</span>
            <span className="slash">/</span>
            {
              {
                search: "全景查询",
                history: "查询历史",
                library: "音乐资料库",
                tools: "平台工具",
              }[view]
            }
          </div>
          <div className="topbar-right">
            <span className="local-badge">LOCAL FIRST</span>
            <span className="profile">M</span>
          </div>
        </header>
        <nav className="mobile-nav" aria-label="移动工作台导航">
          {[
            ["search", "查询"],
            ["history", "历史"],
            ["library", "资料库"],
            ["tools", "工具"],
          ].map(([key, label]) => (
            <button
              key={key}
              aria-current={view === key ? "page" : undefined}
              onClick={() => changeView(key)}
            >
              {label}
            </button>
          ))}
        </nav>
        <main>
          {view === "search" && (
            <>
              <section className="hero">
                <div className="eyebrow">
                  <span /> MUSIC INTELLIGENCE WORKSPACE
                </div>
                <h1>
                  把一首歌，<span>看得更完整。</span>
                </h1>
                <p>
                  从原唱到不同演绎，从一个平台到多个平台。
                  <br />
                  让每次查询，成为下一次洞察的起点。
                </p>
                <div className="hero-art" aria-hidden="true">
                  <div className="record">
                    <div className="record-label">
                      <Icon name="music" size={28} />
                    </div>
                  </div>
                  <div className="art-caption">
                    ONE SONG.
                    <br />
                    MANY PERSPECTIVES.
                  </div>
                  <span className="art-spark">✳</span>
                </div>
              </section>
              <section className="agent-strip" aria-label="DeepSeek Agent 状态">
                <div>
                  <strong>
                    DeepSeek Agent · {agentStatus?.model || "连接后端中"}
                  </strong>
                  <span
                    className={
                      agentStatus?.configured ? "saved-label" : "muted"
                    }
                  >
                    {agentStatus?.configured
                      ? "已配置 · 需真实连接核验"
                      : "未配置 API Key · 不会假装启用"}
                  </span>
                </div>
                <p>
                  模型规划 → 工具反馈循环 → 作品澄清 → 证据校验回答。最多{" "}
                  {agentStatus?.max_rounds || 10} 轮、
                  {agentStatus?.max_tools || 18} 次工具调用。
                </p>
                <p className="muted">
                  Agent 会将查询、最近会话和公开工具结果发送给
                  DeepSeek；密钥仅在服务器。
                </p>
                {!agentStatus?.configured && (
                  <details>
                    <summary>配置 DeepSeek</summary>
                    <p>
                      在项目根目录 .env 填写 DEEPSEEK_API_KEY，保存后重启
                      start.command。不要将 Key 输入查询框或发到聊天中。
                    </p>
                    <code>DEEPSEEK_MODEL=deepseek-flash</code>
                    <p>没有 Key 时，可在下方明确选择“规则查询”，不是 Agent。</p>
                  </details>
                )}
                <button
                  className="text-button"
                  disabled={!agentStatus?.configured || busy}
                  onClick={checkConnection}
                >
                  测试 DeepSeek 连接
                </button>
                {connectionResult && <p role="status">{connectionResult}</p>}
              </section>
              <section className="query-card">
                <div className="query-title">
                  <Icon name="search" />
                  <span>你想了解哪首歌？</span>
                  <span className="subtle-tag">自然语言查询</span>
                </div>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    submit();
                  }}
                >
                  <textarea
                    aria-label="音乐查询内容"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    rows={2}
                    maxLength={500}
                    placeholder="例如：我希望获取《空心》的近期数据"
                  />
                  <div className="query-controls">
                    <div className="control-group">
                      <label>平台</label>
                      {["qq", "netease"].map((p) => (
                        <button
                          type="button"
                          key={p}
                          className={
                            platforms.includes(p)
                              ? `platform-toggle on ${p}`
                              : "platform-toggle"
                          }
                          aria-pressed={platforms.includes(p)}
                          onClick={() =>
                            setPlatforms((list) =>
                              list.includes(p)
                                ? list.filter((v) => v !== p)
                                : [...list, p],
                            )
                          }
                        >
                          <span />
                          {platformNames[p]}
                        </button>
                      ))}
                    </div>
                    <div className="select-controls">
                      <select
                        aria-label="执行引擎"
                        value={executionMode}
                        onChange={(e) => setExecutionMode(e.target.value)}
                      >
                        <option value="agent">DeepSeek Agent</option>
                        <option value="rules">规则查询 · 不使用 LLM</option>
                      </select>
                      <select
                        aria-label="近期范围"
                        value={days}
                        onChange={(e) => setDays(Number(e.target.value))}
                      >
                        <option value={0}>自动 · 近期默认 7 天</option>
                        <option value={7}>近 7 天</option>
                        <option value={30}>近 30 天</option>
                        <option value={90}>近 90 天</option>
                      </select>
                      <select
                        aria-label="数据获取模式"
                        value={mode}
                        onChange={(e) => setMode(e.target.value)}
                      >
                        <option value="live">实时获取</option>
                        <option value="prefer_cache">优先缓存 · 1 小时</option>
                        <option value="cache_only">仅查历史</option>
                      </select>
                    </div>
                    <button
                      className="primary"
                      type="submit"
                      disabled={
                        busy ||
                        !platforms.length ||
                        !input.trim() ||
                        (executionMode === "agent" && !agentStatus?.configured)
                      }
                    >
                      {busy ? (
                        <>
                          <span className="spinner" />
                          正在查询
                        </>
                      ) : (
                        <>
                          {executionMode === "agent" && !agentStatus?.configured
                            ? "请先配置 DeepSeek"
                            : executionMode === "agent"
                              ? "开始 Agent 研究"
                              : "开始规则查询"}
                          <Icon name="arrow" size={18} />
                        </>
                      )}
                    </button>
                  </div>
                </form>
                <div className="query-footnote">
                  <Icon name="check" size={14} />
                  查询与结果自动入库 <span>·</span>{" "}
                  无历史时仅展示当前可得数据，不生成近期增量
                </div>
              </section>
              {error && (
                <div role="alert" className="error-box">
                  {error}
                </div>
              )}
              {busy && (
                <section className="progress-card">
                  <div>
                    <span className="spinner" />
                    <strong>{lastEvent || "查询已受理，正在等待执行"}</strong>
                  </div>
                  <div className="progress-stages">
                    {active?.checkpoint.answer_only_retry ||
                    active?.agent_steps?.some(
                      (step) => step.kind === "answer_regeneration",
                    ) ? (
                      <span>仅基于已保存证据重新生成回答</span>
                    ) : (
                      <>
                        <span>
                          {active?.request.execution_mode === "agent"
                            ? "01 模型规划"
                            : "01 解析需求"}
                        </span>
                        <span>
                          {active?.request.execution_mode === "agent"
                            ? "02 工具反馈循环"
                            : "02 发现版本"}
                        </span>
                        <span>
                          {active?.request.execution_mode === "agent"
                            ? "03 证据引用校验"
                            : "03 获取数据"}
                        </span>
                        <span>04 保存与汇总</span>
                      </>
                    )}
                  </div>
                  <button
                    className="text-button"
                    onClick={() => action("cancel")}
                  >
                    取消后续请求
                  </button>
                </section>
              )}
              {active &&
                !busy &&
                active.request.execution_mode === "agent" &&
                !!active.report.songs?.length && (
                  <button
                    className="text-button"
                    disabled={!agentStatus?.configured}
                    onClick={() => action("regenerate-answer")}
                  >
                    重新生成回答 · 不重新获取数据（使用模型额度）
                  </button>
                )}
              {active?.execution_status === "waiting_input" && (
                <section className="review-card">
                  <h2>先确认一下，你指的是哪一首？</h2>
                  <p>
                    {active.checkpoint.question ||
                      "平台发现了不同歌手的同名原创标记。我们不会将它们自动合并。"}
                  </p>
                  {active.checkpoint.choices?.map((c) => (
                    <button
                      key={c.key}
                      onClick={() => action("select", { anchor_id: c.key })}
                    >
                      <Icon name="music" />
                      <span>
                        {c.title} · {c.artists.join(" / ")}
                        <small>{c.album}</small>
                      </span>
                      <Icon name="arrow" />
                    </button>
                  ))}
                </section>
              )}
              {active &&
                !busy &&
                active.execution_status !== "waiting_input" && (
                  <section className="report-section">
                    <div className="section-heading">
                      <div>
                        <div className="eyebrow">QUERY REPORT</div>
                        <h2>
                          {report?.title
                            ? `《${report.title}》全景报告`
                            : "查询记录"}
                        </h2>
                      </div>
                      <div className="report-actions">
                        <span
                          className={`status-badge ${active.execution_status}`}
                        >
                          {statuses[active.execution_status]}
                        </span>
                        {((active.request.execution_mode !== "agent" &&
                          active.execution_status === "failed") ||
                          report?.model_error === "model_unconfigured" ||
                          (active.execution_status === "partial" &&
                            active.request.execution_mode !== "agent" &&
                            Object.keys(report?.failures || {}).length >
                              0)) && (
                          <button
                            className="icon-button"
                            aria-label="重试失败平台"
                            title="只重试失败平台，保留已提交结果"
                            onClick={() => action("retry")}
                          >
                            <Icon name="arrow" />
                          </button>
                        )}
                        <button
                          className="icon-button"
                          aria-label="实时刷新"
                          title="实时刷新，保留旧观测"
                          onClick={() => action("refresh")}
                        >
                          <Icon name="refresh" />
                        </button>
                        <a
                          className="icon-button"
                          aria-label="导出 CSV"
                          title="导出 CSV"
                          href={`${API}/queries/${active.query_id}/export?format=csv`}
                        >
                          <Icon name="download" />
                        </a>
                      </div>
                    </div>
                    {report?.answer && (
                      <article className="agent-answer">
                        <div className="eyebrow">EVIDENCE-GROUNDED ANSWER</div>
                        <h3>Agent 研究回答 · {report.model}</h3>
                        {report.answer.status !== "verified" && (
                          <p className="muted">
                            {report.answer.reason}。结构化事实仍保留。
                          </p>
                        )}
                        {report.answer.blocks.map((block, i) => (
                          <section key={i}>
                            <h4>{block.heading}</h4>
                            <ul>
                              {block.lines.map((line, j) => {
                                const citation = report.answer?.citations.find(
                                  (c) =>
                                    c.observation_id === line.observation_id,
                                );
                                return (
                                  <li key={j}>
                                    {line.text}
                                    {citation && (
                                      <>
                                        <small>
                                          {" "}
                                          · {date(citation.collected_at, true)}
                                        </small>{" "}
                                        <a
                                          href={citation.source_url}
                                          target="_blank"
                                          rel="noreferrer"
                                          title={citation.observation_id}
                                        >
                                          [来源 ·{" "}
                                          {citation.observation_id.slice(0, 8)}]
                                        </a>
                                      </>
                                    )}
                                  </li>
                                );
                              })}
                            </ul>
                          </section>
                        ))}
                        {report.answer.next_action && (
                          <p>{report.answer.next_action}</p>
                        )}
                      </article>
                    )}
                    <div className="summary-card">
                      <span className="summary-icon">
                        <Icon name="music" />
                      </span>
                      <div>
                        <p>
                          {report?.summary ||
                            active.error ||
                            "本次查询未形成完整报告；已提交结果可在下方查看。"}
                        </p>
                        <div className="summary-meta">
                          <span className="saved-label">
                            {active.persistence_status === "saved"
                              ? "✓ 已保存"
                              : `保存状态：${active.persistence_status}`}
                          </span>
                          <span>
                            {report?.origin === "cache"
                              ? "历史缓存 · 非本次实时采集"
                              : "实时获取"}
                          </span>
                          {report?.window && (
                            <span>
                              请求范围 {date(report.window.start)} —{" "}
                              {date(report.window.end)}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    {report?.original && (
                      <div className="original-line">
                        平台标注原始版本{" "}
                        <strong>{report.original.artists.join(" / ")}</strong>
                        <a
                          href={report.original.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          查看依据 <Icon name="link" size={12} />
                        </a>
                        <span>平台声明，非独立版权认证</span>
                      </div>
                    )}
                    {shownGroups.map((group, index) => (
                      <article className="version-card" key={index}>
                        <div className="version-header">
                          <div className="album-tile">
                            <Icon name="music" size={26} />
                            <span>{String(index + 1).padStart(2, "0")}</span>
                          </div>
                          <div>
                            <h3>{group.artists.join(" / ")}</h3>
                            <p>
                              {group.title}
                              <span className="small-tag">
                                {versions[group.version] || group.version}
                              </span>
                            </p>
                          </div>
                          <span className={`relation-tag ${group.relation}`}>
                            {group.relation === "anchor"
                              ? "目标原始版本"
                              : group.relation === "platform_linked_cover"
                                ? "平台关联翻唱"
                                : group.relation === "metadata_match"
                                  ? "元数据匹配"
                                  : "作品关系待核实"}
                          </span>
                        </div>
                        <div className="table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>音乐平台</th>
                                <th>评论总数</th>
                                <th>收藏 / 红心</th>
                                <th>数据时点</th>
                                <th>来源</th>
                              </tr>
                            </thead>
                            <tbody>
                              {group.songs.map((song) => (
                                <tr key={song.platform + song.platform_id}>
                                  <td>
                                    <span
                                      className={`platform-label ${song.platform}`}
                                    >
                                      <i />
                                      {platformNames[song.platform]}
                                    </span>
                                  </td>
                                  {["comment_count", "favorite_count"].map(
                                    (name) => {
                                      const m = song.metrics.find(
                                        (m) => m.name === name,
                                      );
                                      return (
                                        <td key={name}>
                                          <span
                                            title={m?.reason || undefined}
                                            className={
                                              m?.status === "exact" ||
                                              m?.status === "approximate"
                                                ? "metric-value"
                                                : "unavailable"
                                            }
                                          >
                                            {m?.status === "exact"
                                              ? number(m.value)
                                              : m?.status === "approximate"
                                                ? m.display
                                                : "暂不可获取"}
                                          </span>
                                          {m?.status === "approximate" && (
                                            <small>
                                              平台展示区间，非精确值
                                            </small>
                                          )}
                                          {m &&
                                            ["error", "unavailable"].includes(
                                              m.status,
                                            ) && (
                                              <small>
                                                {reasonNames[m.reason || ""] ||
                                                  "未取得可靠数值"}
                                              </small>
                                            )}
                                          {m?.comparison && (
                                            <small className="delta">
                                              {m.comparison.delta >= 0
                                                ? "+"
                                                : ""}
                                              {number(m.comparison.delta)} ·
                                              相对上次观测
                                            </small>
                                          )}
                                        </td>
                                      );
                                    },
                                  )}
                                  <td className="time-cell">
                                    {date(
                                      song.metrics[0]?.collected_at ||
                                        song.fetched_at,
                                      true,
                                    )}
                                    <small>当前累计 · 采集时点</small>
                                  </td>
                                  <td>
                                    <a
                                      className="source-link"
                                      href={song.url}
                                      target="_blank"
                                      rel="noreferrer"
                                    >
                                      平台详情 <Icon name="link" size={13} />
                                    </a>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </article>
                    ))}
                    {!shownGroups.length && active.results.length > 0 && (
                      <div className="limit-card">
                        已保存 {active.results.length}{" "}
                        个平台结果；本次未完成全景汇总。可导出 JSON
                        查看完整记录。
                      </div>
                    )}
                    {report?.output !== undefined && (
                      <pre className="tool-output">
                        {JSON.stringify(report.output, null, 2)}
                      </pre>
                    )}
                    {!!report?.limitations?.length && (
                      <details className="limit-card" open>
                        <summary>数据范围与限制</summary>
                        <ul>
                          {report.limitations.map((l, i) => (
                            <li key={i}>{l}</li>
                          ))}
                        </ul>
                        {report.selection_basis && (
                          <p>版本筛选：{report.selection_basis}</p>
                        )}
                      </details>
                    )}
                    <details className="trace-card">
                      <summary>
                        查看工具执行与数据证据 · {active.tool_calls.length}{" "}
                        次调用
                      </summary>
                      {active.tool_calls.map((c, i) => (
                        <div key={i}>
                          <code>{c.tool}</code>
                          <span>
                            {c.status === "completed" ? "完成" : c.error_code}
                          </span>
                          <small>{c.duration_ms} ms</small>
                        </div>
                      ))}
                      <p>
                        查询 ID：{active.query_id} ·{" "}
                        <a
                          href={`${API}/queries/${active.query_id}/export?format=json`}
                        >
                          下载完整 JSON 与观测引用
                        </a>
                      </p>
                    </details>
                  </section>
                )}
              {active && !busy && (
                <section className="conversation-card">
                  <div className="section-heading">
                    <h2>继续研究 · 多轮会话</h2>
                    <span className="muted">
                      {conversation
                        ? `${conversation.turns.length} 轮记录`
                        : "追问会创建关联会话"}
                    </span>
                  </div>
                  {conversation?.turns.map((turn) => (
                    <article
                      key={turn.query_id}
                      className={
                        turn.query_id === active.query_id
                          ? "conversation-turn current"
                          : "conversation-turn"
                      }
                    >
                      <p>
                        <strong>你：</strong>
                        {turn.input}
                      </p>
                      <p className="muted">
                        {turn.question ||
                          turn.report.summary ||
                          statuses[turn.execution_status]}
                      </p>
                      {turn.query_id !== active.query_id && (
                        <button
                          className="text-button"
                          onClick={() =>
                            open(turn.query_id).catch((e) =>
                              setError(e.message),
                            )
                          }
                        >
                          查看此轮报告
                        </button>
                      )}
                    </article>
                  ))}
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      sendFollowup();
                    }}
                  >
                    <textarea
                      aria-label="继续追问"
                      maxLength={500}
                      rows={2}
                      value={followup}
                      onChange={(e) => setFollowup(e.target.value)}
                      placeholder="例如：我指的是光泽的版本；或：刚才那首歌只看网易云的数据"
                    />
                    <button
                      className="primary"
                      disabled={
                        !followup.trim() ||
                        !agentStatus?.configured ||
                        !platforms.length
                      }
                      type="submit"
                    >
                      继续问 DeepSeek <Icon name="arrow" size={16} />
                    </button>
                  </form>
                  {!agentStatus?.configured && (
                    <p className="muted">
                      需配置 DeepSeek 后才能多轮研究，旧报告仍可查看。
                    </p>
                  )}
                </section>
              )}
              {!!active?.agent_steps?.length && !busy && (
                <details className="trace-card">
                  <summary>
                    Agent 公开轨迹与预算 · {active.agent_steps.length} 条事件
                  </summary>
                  <p>
                    仅包含公开计划、工具选择、反馈和用量，不展示或保存私有思维链。
                  </p>
                  {report?.agent_usage && (
                    <p>
                      {report.agent_usage.rounds} 轮 ·{" "}
                      {report.agent_usage.tool_calls} 次工具尝试 ·{" "}
                      {report.agent_usage.budget_tokens} 预算
                      Token（缺少提供商用量时为估计）
                    </p>
                  )}
                  {active.agent_steps.map((step) => (
                    <div key={step.step_id}>
                      <code>{step.kind}</code>
                      <pre>{JSON.stringify(step.payload, null, 2)}</pre>
                    </div>
                  ))}
                </details>
              )}
              {!active && !busy && (
                <>
                  <div className="section-heading start-heading">
                    <h2>让数据成为你的工作习惯</h2>
                    <span className="muted">每次查询，都有迹可循</span>
                  </div>
                  <div className="feature-grid">
                    <article>
                      <span className="feature-icon">
                        <Icon name="music" />
                      </span>
                      <span className="feature-index">01</span>
                      <h3>一首歌，多种演绎</h3>
                      <p>
                        发现原始版本与演唱候选，
                        <br />
                        保留版本差异，不混淆同名作品。
                      </p>
                    </article>
                    <article>
                      <span className="feature-icon">
                        <Icon name="layers" />
                      </span>
                      <span className="feature-index">02</span>
                      <h3>跨平台，有据可查</h3>
                      <p>
                        评论、收藏及公开元数据，
                        <br />
                        每个数值都带来源与时间口径。
                      </p>
                    </article>
                    <article>
                      <span className="feature-icon">
                        <Icon name="history" />
                      </span>
                      <span className="feature-index">03</span>
                      <h3>查询一次，积累一次</h3>
                      <p>
                        查询自动归档，刷新保留旧值，
                        <br />
                        从当前快照逐步积累历史。
                      </p>
                    </article>
                  </div>
                  <div className="stats-strip">
                    <div>
                      <strong>{number(stats.queries)}</strong>
                      <span>已归档查询</span>
                    </div>
                    <div>
                      <strong>{number(stats.entities)}</strong>
                      <span>平台歌曲实体</span>
                    </div>
                    <div>
                      <strong>{number(stats.observations)}</strong>
                      <span>留存指标观测</span>
                    </div>
                    <p>
                      从空数据库开始
                      <br />
                      只积累真实获取的数据
                    </p>
                  </div>
                </>
              )}
            </>
          )}
          {view === "history" && (
            <section>
              <div className="eyebrow">YOUR RESEARCH TRAIL</div>
              <h1 className="page-title">查询历史</h1>
              <p className="page-description">
                成功、部分结果与失败，都留下可追溯的记录。
              </p>
              <div className="history-list">
                {history.map((h) => (
                  <button
                    key={h.query_id}
                    onClick={() =>
                      open(h.query_id).catch((e) => setError(e.message))
                    }
                  >
                    <span className="history-icon">
                      <Icon name="history" />
                    </span>
                    <div>
                      <strong>{h.input}</strong>
                      <small>
                        {date(h.created_at, true)} · {h.result_count} 个报告结果
                      </small>
                    </div>
                    <span className={`status-badge ${h.execution_status}`}>
                      {statuses[h.execution_status]}
                    </span>
                    <Icon name="arrow" size={18} />
                  </button>
                ))}
                {!history.length && (
                  <div className="empty-state">
                    还没有查询记录。试着从一首歌开始。
                  </div>
                )}
              </div>
            </section>
          )}
          {view === "library" && (
            <section>
              <div className="eyebrow">BUILT WITH EVERY QUERY</div>
              <h1 className="page-title">音乐资料库</h1>
              <p className="page-description">
                查询后积累的平台歌曲实体。不同录音与不同平台身份独立保存。
              </p>
              <div className="library-grid">
                {library.map((s) => (
                  <article key={s.entity_id}>
                    <div className="library-art">
                      <Icon name="music" size={30} />
                      <span>{platformNames[s.platform]}</span>
                    </div>
                    <h3>{s.title}</h3>
                    <p>{s.artists.join(" / ")}</p>
                    <small>{s.album || "发行信息未提供"}</small>
                    <a
                      className="source-link"
                      href={s.url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      平台详情 <Icon name="link" size={13} />
                    </a>
                  </article>
                ))}
                {!library.length && (
                  <div className="empty-state">
                    数据库可以为空。首次查询后，音乐资料会自动出现在这里。
                  </div>
                )}
              </div>
            </section>
          )}
          {view === "tools" && (
            <section>
              <div className="eyebrow">VERIFIED PLATFORM TOOLS</div>
              <h1 className="page-title">平台工具</h1>
              <p className="page-description">
                六个受控、只读的平台工具。工作流与诊断使用同一注册契约，诊断结果也会保存。
              </p>
              <div className="tool-grid">
                {tools.map((t) => (
                  <article key={t.name}>
                    <div>
                      <Icon name="tool" />
                      <span className="small-tag">v{t.version}</span>
                    </div>
                    <h3>{t.name}</h3>
                    <p>{t.description}</p>
                    <details>
                      <summary>查看输入契约</summary>
                      <pre>{JSON.stringify(t.input_schema, null, 2)}</pre>
                    </details>
                  </article>
                ))}
              </div>
              <section className="tool-test">
                <h2>工具验证台</h2>
                <label>
                  选择工具
                  <select
                    value={toolName}
                    onChange={(e) => {
                      setToolName(e.target.value);
                      setToolInput(
                        e.target.value.endsWith("search_songs")
                          ? "空心"
                          : e.target.value.startsWith("netease")
                            ? "435288399"
                            : e.target.value.endsWith("song_metrics")
                              ? "200390803"
                              : "002WW9dA4RYnHm",
                      );
                    }}
                  >
                    {tools.map((t) => (
                      <option key={t.name}>{t.name}</option>
                    ))}
                  </select>
                </label>
                <label>
                  {toolName.endsWith("search_songs")
                    ? "搜索词"
                    : toolName === "qq.song_detail"
                      ? "歌曲 MID（多个用逗号分隔）"
                      : "歌曲数字 ID（多个用逗号分隔）"}
                  <input
                    value={toolInput}
                    onChange={(e) => setToolInput(e.target.value)}
                  />
                </label>
                <button
                  className="primary"
                  disabled={busy || !toolInput.trim()}
                  onClick={invoke}
                >
                  {busy ? "验证中…" : "调用并保存"}
                  <Icon name="arrow" size={16} />
                </button>
                {toolOutput !== null && (
                  <pre className="tool-output">
                    {JSON.stringify(toolOutput, null, 2)}
                  </pre>
                )}
              </section>
            </section>
          )}
          {view !== "search" && error && (
            <div className="error-box" role="alert">
              {error}
            </div>
          )}
          <footer>
            <span>
              MusicTracker <span className="muted">/</span> 为音乐工作者而构建
            </span>
            <span>数据有来源 · 时间有口径 · 查询有记录</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
