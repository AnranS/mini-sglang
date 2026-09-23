// Resolves references into the mini-sglang source tree at build time.
//
// Pages never hard-code line numbers. They name a file plus a symbol or a text
// anchor, and this module finds the current line. A reference that no longer
// resolves throws, so the build fails instead of shipping a stale link.
//
// Paths are relative to `python/minisgl/` unless they start with `/`, in which
// case they are relative to the repository root (e.g. `/benchmark/offline/bench.py`).

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

const PKG_PREFIX = 'python/minisgl/';

function git(args, cwd) {
	try {
		return execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
	} catch {
		return '';
	}
}

// Astro bundles this module into dist/ during the build, so `import.meta.url` is
// not stable. The build always runs from docs/learn, so resolve from the cwd.
export const REPO_ROOT = git(['rev-parse', '--show-toplevel'], process.cwd()) || path.resolve(process.cwd(), '../..');
const REPO = process.env.GITHUB_REPOSITORY || 'AnranS/mini-sglang';
const SHA = process.env.GITHUB_SHA || git(['rev-parse', 'HEAD'], REPO_ROOT) || 'main';

export const SOURCE_INFO = {
	repo: REPO,
	sha: SHA,
	shortSha: SHA.slice(0, 7),
	repoUrl: `https://github.com/${REPO}`,
	blobBase: `https://github.com/${REPO}/blob/${SHA}/`,
	treeBase: `https://github.com/${REPO}/tree/${SHA}/`,
};

const PY_EXT = new Set(['.py']);
const C_EXT = new Set(['.c', '.cc', '.cpp', '.cu', '.cuh', '.h', '.hpp']);
const LANG_BY_EXT = {
	'.py': 'python',
	'.cu': 'cpp',
	'.cuh': 'cpp',
	'.cpp': 'cpp',
	'.cc': 'cpp',
	'.c': 'c',
	'.h': 'cpp',
	'.hpp': 'cpp',
	'.md': 'md',
	'.toml': 'toml',
	'.sh': 'bash',
	'.yaml': 'yaml',
	'.yml': 'yaml',
};

const fileCache = new Map();

/** Maps a doc-style path to a repository-relative path and checks it exists. */
export function resolvePath(p) {
	if (!p) throw new Error('[source] 缺少 path');
	const rel = p.startsWith('/') ? p.slice(1) : PKG_PREFIX + p;
	const abs = path.join(REPO_ROOT, rel);
	if (!existsSync(abs)) throw new Error(`[source] 文件不存在：${rel}`);
	return { rel, abs };
}

export function readLines(rel) {
	let lines = fileCache.get(rel);
	if (!lines) {
		lines = readFileSync(path.join(REPO_ROOT, rel), 'utf8').replace(/\r\n/g, '\n').split('\n');
		if (lines.at(-1) === '') lines.pop();
		fileCache.set(rel, lines);
	}
	return lines;
}

/** Number of lines in a source file, for reading lists. */
export function lineCount(p) {
	return readLines(resolvePath(p).rel).length;
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const indentOf = (line) => line.length - line.trimStart().length;
const isBlank = (line) => line.trim() === '';

function pySignatureEnd(lines, start) {
	let depth = 0;
	for (let i = start; i < lines.length; i++) {
		const code = lines[i].replace(/#.*$/, '');
		for (const ch of code) {
			if (ch === '(' || ch === '[' || ch === '{') depth++;
			else if (ch === ')' || ch === ']' || ch === '}') depth--;
		}
		if (depth <= 0 && /:\s*$/.test(code.trimEnd())) return i;
	}
	return start;
}

/** Last line (inclusive, 0-based) of the Python block that starts at `start`. */
function pyBlockEnd(lines, start) {
	const base = indentOf(lines[start]);
	const header = lines[start].trim();
	if (!/^(async\s+def|def|class)\b/.test(header)) return pyStatementEnd(lines, start);
	let last = pySignatureEnd(lines, start);
	for (let i = last + 1; i < lines.length; i++) {
		const line = lines[i];
		if (isBlank(line) || line.trim().startsWith('#')) continue;
		if (indentOf(line) <= base) break;
		last = i;
	}
	return last;
}

/** Last line of a (possibly multi-line) Python statement starting at `start`. */
function pyStatementEnd(lines, start) {
	let depth = 0;
	for (let i = start; i < lines.length; i++) {
		const code = lines[i].replace(/#.*$/, '');
		for (const ch of code) {
			if (ch === '(' || ch === '[' || ch === '{') depth++;
			else if (ch === ')' || ch === ']' || ch === '}') depth--;
		}
		if (depth <= 0 && !code.trimEnd().endsWith('\\')) return i;
	}
	return start;
}

/** Last line of a brace-delimited C/C++/CUDA block whose header starts at `start`. */
function cBlockEnd(lines, start) {
	let depth = 0;
	let opened = false;
	for (let i = start; i < lines.length; i++) {
		const code = lines[i].replace(/\/\/.*$/, '');
		for (const ch of code) {
			if (ch === '{') {
				depth++;
				opened = true;
			} else if (ch === '}') depth--;
		}
		if (opened && depth <= 0) return i;
		if (!opened && code.trimEnd().endsWith(';')) return i;
	}
	return start;
}

function findPySymbol(lines, sym, rel) {
	let lo = 0;
	let hi = lines.length - 1;
	let parentIndent = -1;
	let line = -1;
	for (const name of sym.split('.')) {
		const defRe = new RegExp(`^(\\s*)(?:async\\s+def|def|class)\\s+${escapeRe(name)}\\b`);
		const attrRe = new RegExp(`^(\\s*)(?:self\\.)?${escapeRe(name)}\\s*(?::|=[^=])`);
		let found = -1;
		for (const re of [defRe, attrRe]) {
			for (let i = lo; i <= hi; i++) {
				const m = re.exec(lines[i]);
				if (m && m[1].length > parentIndent) {
					found = i;
					break;
				}
			}
			if (found >= 0) break;
		}
		if (found < 0) throw new Error(`[source] 在 ${rel} 里找不到符号 ${sym}（卡在 ${name}）`);
		line = found;
		parentIndent = indentOf(lines[found]);
		lo = found + 1;
		hi = pyBlockEnd(lines, found);
	}
	return { start: line, end: pyBlockEnd(lines, line) };
}

function findCSymbol(lines, sym, rel) {
	const re = new RegExp(`\\b${escapeRe(sym)}\\s*(?:<[^;]*>)?\\s*\\(`);
	for (let i = 0; i < lines.length; i++) {
		const code = lines[i].replace(/\/\/.*$/, '');
		if (!re.test(code) || code.trimEnd().endsWith(';')) continue;
		let end = cBlockEnd(lines, i);
		if (end === i && !lines[i].includes('{')) continue;
		let start = i;
		while (start > 0 && /^\s*template\s*</.test(lines[start - 1])) start--;
		return { start, end };
	}
	throw new Error(`[source] 在 ${rel} 里找不到函数 ${sym}`);
}

function findText(lines, text, lo, hi, rel, what) {
	for (let i = lo; i <= hi; i++) if (lines[i].includes(text)) return i;
	throw new Error(`[source] 在 ${rel} 的 ${what} 里找不到文本：${JSON.stringify(text)}`);
}

function blockEnd(lines, start, ext) {
	return C_EXT.has(ext) ? cBlockEnd(lines, start) : pyBlockEnd(lines, start);
}

/**
 * Locates `sym` and/or `text` in a file. Returns 1-based line numbers.
 * `sym` narrows the search scope; `text` then picks the first matching line inside it.
 */
export function locate({ path: p, sym, text }) {
	const { rel } = resolvePath(p);
	const ext = path.extname(rel);
	const lines = readLines(rel);
	let start = 0;
	let end = lines.length - 1;
	let scope = '整个文件';
	if (sym) {
		const found = PY_EXT.has(ext) ? findPySymbol(lines, sym, rel) : findCSymbol(lines, sym, rel);
		({ start, end } = found);
		scope = `符号 ${sym}`;
	}
	if (text) {
		start = findText(lines, text, start, end, rel, scope);
		end = start;
	}
	return { rel, ext, lines, start: start + 1, end: end + 1 };
}

export function blobUrl(rel, start, end) {
	const anchor = start ? (end && end !== start ? `#L${start}-L${end}` : `#L${start}`) : '';
	return `${SOURCE_INFO.blobBase}${rel}${anchor}`;
}

/** A short inline reference: file + line + GitHub URL. */
export function resolveRef({ path: p, sym, text }) {
	const { rel, start } = locate({ path: p, sym, text });
	const short = rel.startsWith(PKG_PREFIX) ? rel.slice(PKG_PREFIX.length) : rel;
	return { rel, short, line: start, url: blobUrl(rel, sym || text ? start : 0) };
}

/**
 * What a link to `ref` shows: the symbol (or file) as the label, plus the
 * file name and line as a subdued location when a symbol is given.
 */
export function describeRef({ path: p, sym, text, label }) {
	const ref = resolveRef({ path: p, sym, text });
	const located = Boolean(sym || text);
	return {
		label: label ?? sym ?? ref.short + (located ? `:${ref.line}` : ''),
		loc: sym ? `${ref.short.split('/').pop()}:${ref.line}` : '',
		url: ref.url,
		title: located ? `${ref.rel} 第 ${ref.line} 行` : ref.rel,
	};
}

/** Number of lines of the block `sym` names, or of the whole file. */
export function spanLines({ path: p, sym }) {
	const { start, end } = locate({ path: p, sym });
	return end - start + 1;
}

/** Splits text on `code` spans: [{ text, code }]. */
export function splitInlineCode(text) {
	return text
		.split(/(`[^`]+`)/)
		.filter(Boolean)
		.map((part) =>
			part.startsWith('`') && part.endsWith('`') && part.length > 1
				? { text: part.slice(1, -1), code: true }
				: { text: part, code: false },
		);
}

function dedent(lines) {
	const indents = lines.filter((l) => !isBlank(l)).map(indentOf);
	const min = indents.length ? Math.min(...indents) : 0;
	return lines.map((l) => l.slice(min));
}

/**
 * A code excerpt. Without `from`/`to`, a symbol yields its whole block
 * (decorators included). `from` and `to` are text anchors, searched inside the
 * symbol when one is given; `to` is inclusive and, when it names a block
 * header, extends to the end of that block if `toEnd` is set.
 */
export function resolveExcerpt({ path: p, sym, from, to, toEnd = false, maxLines = 90 }) {
	const { rel, ext, lines, start: s0, end: e0 } = locate({ path: p, sym });
	let start = s0 - 1;
	let end = e0 - 1;
	const scope = sym ? `符号 ${sym}` : '整个文件';
	if (!sym && !from) throw new Error(`[source] 摘录 ${rel} 需要 sym 或 from`);
	if (from) start = findText(lines, from, start, end, rel, scope);
	if (to) {
		end = findText(lines, to, start, end, rel, scope);
		if (toEnd) end = blockEnd(lines, end, ext);
	} else if (from) {
		end = blockEnd(lines, start, ext);
	}
	if (sym && !from && PY_EXT.has(ext)) {
		const indent = indentOf(lines[start]);
		while (start > 0 && lines[start - 1].trim().startsWith('@') && indentOf(lines[start - 1]) === indent) start--;
	}
	while (end > start && isBlank(lines[end])) end--;
	const count = end - start + 1;
	if (count > maxLines) {
		throw new Error(`[source] ${rel} 的摘录有 ${count} 行，超过上限 ${maxLines}；用 from/to 缩小范围，或显式调大 maxLines`);
	}
	return {
		rel,
		lang: LANG_BY_EXT[ext] ?? 'text',
		start: start + 1,
		end: end + 1,
		code: dedent(lines.slice(start, end + 1)).join('\n'),
		url: blobUrl(rel, start + 1, end + 1),
	};
}
