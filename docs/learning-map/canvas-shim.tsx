// Browser implementation of the subset of the Cursor `cursor/canvas` SDK that the
// learning map uses, so the same canvas file renders both in Cursor and on the web.
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

export { useEffect, useMemo, useRef, useState } from "react";

// Line numbers in the learning map refer to this commit, not to the tip of main.
const SOURCE_BASE = "https://github.com/AnranS/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/";
const STORAGE_PREFIX = "minisgl-learning-map:";
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const NARROW = "(max-width: 760px)";

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

type Tone = "info" | "success" | "warning" | "danger" | "neutral";

export interface CanvasHostTheme {
  kind: "light" | "dark";
  bg: { editor: string; chrome: string; elevated: string };
  text: { primary: string; secondary: string; tertiary: string; quaternary: string; link: string; onAccent: string };
  stroke: { primary: string; secondary: string; tertiary: string; focused: string };
  fill: { primary: string; secondary: string; tertiary: string; quaternary: string };
  accent: { primary: string; control: string; controlHover: string };
  tone: Record<Tone, string>;
}

const DARK: CanvasHostTheme = {
  kind: "dark",
  bg: { editor: "#1b1b1b", chrome: "#151515", elevated: "#242424" },
  text: {
    primary: "#e8e8e8",
    secondary: "#b4b4b4",
    tertiary: "#8e8e8e",
    quaternary: "#6e6e6e",
    link: "#6cb6ff",
    onAccent: "#ffffff",
  },
  stroke: {
    primary: "rgba(255,255,255,0.24)",
    secondary: "rgba(255,255,255,0.16)",
    tertiary: "rgba(255,255,255,0.09)",
    focused: "#5aa2ff",
  },
  fill: {
    primary: "rgba(255,255,255,0.14)",
    secondary: "rgba(255,255,255,0.09)",
    tertiary: "rgba(255,255,255,0.055)",
    quaternary: "rgba(255,255,255,0.03)",
  },
  accent: { primary: "#5aa2ff", control: "#2f6fe0", controlHover: "#2a63c9" },
  tone: { info: "#5aa2ff", success: "#3fb950", warning: "#d29922", danger: "#f85149", neutral: "#8e8e8e" },
};

const LIGHT: CanvasHostTheme = {
  kind: "light",
  bg: { editor: "#ffffff", chrome: "#f6f6f6", elevated: "#ffffff" },
  text: {
    primary: "#1f2328",
    secondary: "#4d535a",
    tertiary: "#6e7781",
    quaternary: "#8c959f",
    link: "#0969da",
    onAccent: "#ffffff",
  },
  stroke: {
    primary: "rgba(31,35,40,0.24)",
    secondary: "rgba(31,35,40,0.15)",
    tertiary: "rgba(31,35,40,0.09)",
    focused: "#0969da",
  },
  fill: {
    primary: "rgba(31,35,40,0.10)",
    secondary: "rgba(31,35,40,0.065)",
    tertiary: "rgba(31,35,40,0.045)",
    quaternary: "rgba(31,35,40,0.025)",
  },
  accent: { primary: "#0969da", control: "#0969da", controlHover: "#0858b9" },
  tone: { info: "#0969da", success: "#1a7f37", warning: "#9a6700", danger: "#cf222e", neutral: "#6e7781" },
};

const ThemeContext = createContext<CanvasHostTheme>(LIGHT);

function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [query]);
  return matches;
}

export function CanvasHost({ children }: { children?: ReactNode }) {
  const theme = useMedia("(prefers-color-scheme: dark)") ? DARK : LIGHT;
  useEffect(() => {
    document.body.style.background = theme.bg.editor;
    document.body.style.color = theme.text.primary;
    document.documentElement.style.colorScheme = theme.kind;
  }, [theme]);
  return <ThemeContext.Provider value={theme}>{children}</ThemeContext.Provider>;
}

export function useHostTheme(): CanvasHostTheme {
  return useContext(ThemeContext);
}

// ---------------------------------------------------------------------------
// State and actions
// ---------------------------------------------------------------------------

export type SetCanvasState<T> = (action: T | ((prev: T) => T)) => void;

export function useCanvasState<T>(key: string, defaultValue: T): [T, SetCanvasState<T>] {
  const storageKey = STORAGE_PREFIX + key;
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      return raw === null ? defaultValue : (JSON.parse(raw) as T);
    } catch {
      return defaultValue;
    }
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(value));
    } catch {
      // Private browsing or disabled storage: keep the in-memory value only.
    }
  }, [storageKey, value]);
  const update = useCallback<SetCanvasState<T>>(
    (action) => setValue((prev) => (typeof action === "function" ? (action as (p: T) => T)(prev) : action)),
    [],
  );
  return [value, update];
}

export type CanvasAction =
  | {
      type: "openFile";
      path: string;
      selection?: { startLineNumber: number; startColumn?: number; endLineNumber?: number; endColumn?: number };
    }
  | { type: "openAgent"; agentId: string }
  | { type: "newComposerChat"; userPrompt?: string };

export function useCanvasAction(): (action: CanvasAction) => void {
  return useCallback((action: CanvasAction) => {
    if (action.type !== "openFile") return;
    const relative = action.path.replace(/^.*?\/mini-sglang\//, "").replace(/^\/+/, "");
    const line = action.selection?.startLineNumber;
    window.open(SOURCE_BASE + relative + (line ? `#L${line}` : ""), "_blank", "noopener,noreferrer");
  }, []);
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

const ALIGN = { start: "flex-start", center: "center", end: "flex-end", stretch: "stretch" } as const;
const JUSTIFY = { start: "flex-start", center: "center", end: "flex-end", "space-between": "space-between" } as const;

export type StackProps = { children?: ReactNode; gap?: number; style?: CSSProperties };

export function Stack({ children, gap = 8, style }: StackProps) {
  return <div style={{ display: "flex", flexDirection: "column", gap, minWidth: 0, ...style }}>{children}</div>;
}

export type RowProps = {
  children?: ReactNode;
  gap?: number;
  align?: keyof typeof ALIGN;
  justify?: keyof typeof JUSTIFY;
  wrap?: boolean;
  style?: CSSProperties;
};

export function Row({ children, gap = 8, align = "center", justify = "start", wrap = false, style }: RowProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        gap,
        alignItems: ALIGN[align],
        justifyContent: JUSTIFY[justify],
        flexWrap: wrap ? "wrap" : "nowrap",
        minWidth: 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export type GridProps = {
  children?: ReactNode;
  columns: number | string;
  gap?: number;
  align?: keyof typeof ALIGN;
  style?: CSSProperties;
};

export function Grid({ children, columns, gap = 12, align = "stretch", style }: GridProps) {
  const narrow = useMedia(NARROW);
  const template = narrow
    ? "minmax(0, 1fr)"
    : typeof columns === "number"
      ? `repeat(${columns}, minmax(0, 1fr))`
      : columns;
  return (
    <div style={{ display: "grid", gridTemplateColumns: template, gap, alignItems: ALIGN[align], ...style }}>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Typography
// ---------------------------------------------------------------------------

const WEIGHT = { normal: 400, medium: 500, semibold: 600, bold: 700 } as const;
export type TextWeight = keyof typeof WEIGHT;

const INLINE_MARKDOWN = /`([^`]+)`|\[([^\]]+)\]\(([^)\s]+)\)/g;

function renderInline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE_MARKDOWN)) {
    const index = match.index ?? 0;
    if (index > last) out.push(text.slice(last, index));
    out.push(
      match[1] !== undefined ? (
        <Code key={index}>{match[1]}</Code>
      ) : (
        <Link key={index} href={match[3]}>
          {match[2]}
        </Link>
      ),
    );
    last = index + match[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export type TextProps = {
  children?: ReactNode;
  tone?: "primary" | "secondary" | "tertiary" | "quaternary";
  size?: "body" | "small";
  as?: "p" | "span";
  weight?: TextWeight;
  italic?: boolean;
  style?: CSSProperties;
};

export function Text({ children, tone = "primary", size = "body", as = "p", weight = "normal", italic, style }: TextProps) {
  const t = useHostTheme();
  const Tag = as;
  return (
    <Tag
      style={{
        margin: 0,
        color: t.text[tone],
        fontSize: size === "small" ? 12.5 : 14,
        lineHeight: size === "small" ? "19px" : "22px",
        fontWeight: WEIGHT[weight],
        fontStyle: italic ? "italic" : undefined,
        ...style,
      }}
    >
      {typeof children === "string" ? renderInline(children) : children}
    </Tag>
  );
}

type HeadingProps = { children?: ReactNode; style?: CSSProperties };

function heading(Tag: "h1" | "h2" | "h3", fontSize: number, lineHeight: string) {
  return function Heading({ children, style }: HeadingProps) {
    const t = useHostTheme();
    return (
      <Tag style={{ margin: 0, fontSize, lineHeight, fontWeight: 600, color: t.text.primary, ...style }}>
        {children}
      </Tag>
    );
  };
}

export const H1 = heading("h1", 24, "32px");
export const H2 = heading("h2", 18, "26px");
export const H3 = heading("h3", 15, "22px");

export function Code({ children, style }: { children?: ReactNode; style?: CSSProperties }) {
  const t = useHostTheme();
  return (
    <code
      style={{
        fontFamily: MONO,
        fontSize: "0.92em",
        padding: "1px 5px",
        borderRadius: 4,
        background: t.fill.secondary,
        color: t.text.primary,
        overflowWrap: "anywhere",
        ...style,
      }}
    >
      {children}
    </code>
  );
}

export function Link({ children, href, style }: { children?: ReactNode; href: string; style?: CSSProperties }) {
  const t = useHostTheme();
  return (
    <a href={href} target="_blank" rel="noreferrer" style={{ color: t.text.link, textDecoration: "none", ...style }}>
      {children}
    </a>
  );
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

export type ButtonProps = {
  children?: ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
  type?: "button" | "submit" | "reset";
  style?: CSSProperties;
  onClick?: () => void;
};

export function Button({ children, variant = "secondary", disabled, type = "button", style, onClick }: ButtonProps) {
  const t = useHostTheme();
  const look: CSSProperties =
    variant === "primary"
      ? { background: t.accent.control, color: t.text.onAccent, border: "1px solid transparent" }
      : variant === "ghost"
        ? { background: "transparent", color: t.text.secondary, border: "1px solid transparent" }
        : { background: t.fill.secondary, color: t.text.primary, border: `1px solid ${t.stroke.tertiary}` };
  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      style={{
        height: 26,
        padding: "0 10px",
        borderRadius: 6,
        fontSize: 12.5,
        fontFamily: "inherit",
        whiteSpace: "nowrap",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.45 : 1,
        ...look,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

export type SelectOption = { value: string; label: string; disabled?: boolean };

export type SelectProps = {
  value?: string;
  onChange?: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  disabled?: boolean;
  style?: CSSProperties;
};

export function Select({ value, onChange, options, placeholder, disabled, style }: SelectProps) {
  const t = useHostTheme();
  return (
    <select
      value={value ?? ""}
      disabled={disabled}
      onChange={(event) => onChange?.(event.target.value)}
      style={{
        height: 26,
        padding: "0 8px",
        borderRadius: 6,
        fontSize: 12.5,
        fontFamily: "inherit",
        color: t.text.primary,
        background: t.bg.elevated,
        border: `1px solid ${t.stroke.secondary}`,
        ...style,
      }}
    >
      {placeholder !== undefined && value === undefined ? (
        <option value="" disabled>
          {placeholder}
        </option>
      ) : null}
      {options.map((option) => (
        <option key={option.value} value={option.value} disabled={option.disabled}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

export type CalloutProps = {
  children?: ReactNode;
  tone?: Tone;
  title?: ReactNode;
  icon?: ReactNode;
  style?: CSSProperties;
};

export function Callout({ children, tone = "neutral", title, icon, style }: CalloutProps) {
  const t = useHostTheme();
  return (
    <div
      style={{
        display: "flex",
        gap: 10,
        padding: "12px 14px",
        borderRadius: 8,
        background: t.fill.quaternary,
        border: `1px solid ${t.stroke.tertiary}`,
        ...style,
      }}
    >
      {icon ?? (
        <span
          aria-hidden
          style={{ flex: "0 0 auto", width: 8, height: 8, marginTop: 7, borderRadius: 999, background: t.tone[tone] }}
        />
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
        {title ? <div style={{ fontSize: 13.5, fontWeight: 600, color: t.text.primary }}>{title}</div> : null}
        <div style={{ fontSize: 14, color: t.text.secondary }}>{children}</div>
      </div>
    </div>
  );
}

export type TableProps = {
  headers: ReactNode[];
  rows: ReactNode[][];
  columnAlign?: Array<"left" | "center" | "right" | undefined>;
  rowTone?: Array<Tone | undefined>;
  framed?: boolean;
  striped?: boolean;
  style?: CSSProperties;
  emptyMessage?: ReactNode;
};

export function Table({ headers, rows, columnAlign, rowTone, framed = true, striped, style, emptyMessage }: TableProps) {
  const t = useHostTheme();
  const cell = (column: number): CSSProperties => ({
    padding: "8px 12px",
    textAlign: columnAlign?.[column] ?? "left",
    verticalAlign: "top",
  });
  const table = (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, lineHeight: "20px" }}>
      <thead>
        <tr>
          {headers.map((header, column) => (
            <th
              key={column}
              style={{
                ...cell(column),
                fontSize: 12,
                fontWeight: 500,
                color: t.text.tertiary,
                whiteSpace: "nowrap",
                borderBottom: `1px solid ${t.stroke.tertiary}`,
              }}
            >
              {header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.length === 0 ? (
          <tr>
            <td colSpan={headers.length} style={{ ...cell(0), color: t.text.tertiary }}>
              {emptyMessage}
            </td>
          </tr>
        ) : (
          rows.map((row, r) => {
            const tone = rowTone?.[r];
            return (
              <tr key={r} style={{ background: striped && r % 2 === 1 ? t.fill.quaternary : undefined }}>
                {headers.map((_, column) => (
                  <td
                    key={column}
                    style={{
                      ...cell(column),
                      color: t.text.primary,
                      borderTop: r === 0 ? undefined : `1px solid ${t.stroke.tertiary}`,
                    }}
                  >
                    {column === 0 && tone ? (
                      <span
                        aria-hidden
                        style={{
                          display: "inline-block",
                          width: 7,
                          height: 7,
                          marginRight: 8,
                          borderRadius: 999,
                          verticalAlign: "middle",
                          background: t.tone[tone],
                        }}
                      />
                    ) : null}
                    {row[column]}
                  </td>
                ))}
              </tr>
            );
          })
        )}
      </tbody>
    </table>
  );
  if (!framed) return <div style={{ overflowX: "auto", ...style }}>{table}</div>;
  return (
    <div style={{ border: `1px solid ${t.stroke.tertiary}`, borderRadius: 8, overflowX: "auto", ...style }}>{table}</div>
  );
}

// ---------------------------------------------------------------------------
// Todo list
// ---------------------------------------------------------------------------

export type TodoStatus = "pending" | "in_progress" | "completed" | "cancelled";

export interface TodoItem {
  readonly id: string;
  readonly content: string;
  readonly status: TodoStatus;
}

function StatusIcon({ status }: { status: TodoStatus }) {
  const t = useHostTheme();
  const common = { width: 14, height: 14, viewBox: "0 0 14 14", style: { flex: "0 0 auto", marginTop: 3 } } as const;
  if (status === "completed") {
    return (
      <svg {...common} aria-label="已完成">
        <circle cx={7} cy={7} r={6.5} fill={t.tone.success} />
        <path d="M4 7.2 L6.1 9.2 L10 5" fill="none" stroke={t.text.onAccent} strokeWidth={1.6} strokeLinecap="round" />
      </svg>
    );
  }
  if (status === "in_progress") {
    return (
      <svg {...common} aria-label="进行中">
        <circle cx={7} cy={7} r={6} fill="none" stroke={t.accent.primary} strokeWidth={1.5} />
        <path d="M7 3 A4 4 0 0 1 7 11 Z" fill={t.accent.primary} />
      </svg>
    );
  }
  if (status === "cancelled") {
    return (
      <svg {...common} aria-label="已取消">
        <circle cx={7} cy={7} r={6} fill="none" stroke={t.text.tertiary} strokeWidth={1.2} />
        <path d="M4.5 9.5 L9.5 4.5" stroke={t.text.tertiary} strokeWidth={1.2} />
      </svg>
    );
  }
  return (
    <svg {...common} aria-label="未开始">
      <circle cx={7} cy={7} r={6} fill="none" stroke={t.stroke.primary} strokeWidth={1.2} />
    </svg>
  );
}

export type TodoListProps = {
  todos: readonly TodoItem[];
  dimmedTodoIds?: ReadonlySet<string>;
  onTodoClick?: (todo: TodoItem) => void;
  style?: CSSProperties;
};

export function TodoList({ todos, dimmedTodoIds, onTodoClick, style }: TodoListProps) {
  const t = useHostTheme();
  if (todos.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", padding: "4px 0", ...style }}>
      {todos.map((todo) => (
        <button
          key={todo.id}
          type="button"
          onClick={() => onTodoClick?.(todo)}
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 8,
            padding: "6px 12px",
            border: "none",
            background: "transparent",
            textAlign: "left",
            fontFamily: "inherit",
            cursor: onTodoClick ? "pointer" : "default",
            opacity: dimmedTodoIds?.has(todo.id) ? 0.5 : 1,
          }}
        >
          <StatusIcon status={todo.status} />
          <span
            style={{
              fontSize: 13,
              lineHeight: "20px",
              color: todo.status === "completed" ? t.text.tertiary : t.text.primary,
              textDecoration: todo.status === "cancelled" ? "line-through" : undefined,
            }}
          >
            {todo.content}
          </span>
        </button>
      ))}
    </div>
  );
}

export type TodoListCardProps = TodoListProps & { defaultExpanded?: boolean };

export function TodoListCard({ todos, dimmedTodoIds, defaultExpanded = false, onTodoClick, style }: TodoListCardProps) {
  const t = useHostTheme();
  const [open, setOpen] = useState(defaultExpanded);
  if (todos.length === 0) return null;
  const done = todos.filter((todo) => todo.status === "completed").length;
  return (
    <div style={{ border: `1px solid ${t.stroke.tertiary}`, borderRadius: 8, overflow: "hidden", ...style }}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        style={{
          display: "flex",
          width: "100%",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 12px",
          border: "none",
          background: t.fill.quaternary,
          color: t.text.secondary,
          fontSize: 12.5,
          fontFamily: "inherit",
          cursor: "pointer",
        }}
      >
        <span>{`已完成 ${done} / ${todos.length}`}</span>
        <svg width={10} height={10} viewBox="0 0 10 10" aria-hidden style={{ transform: open ? "rotate(180deg)" : undefined }}>
          <path d="M2 3.5 L5 6.5 L8 3.5" fill="none" stroke={t.text.tertiary} strokeWidth={1.4} />
        </svg>
      </button>
      {open ? <TodoList todos={todos} dimmedTodoIds={dimmedTodoIds} onTodoClick={onTodoClick} /> : null}
    </div>
  );
}
