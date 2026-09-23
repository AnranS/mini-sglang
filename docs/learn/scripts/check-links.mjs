// Post-build check: every same-site href/src in dist/ must point at a real file,
// and every #fragment at a real id on the target page.

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const DIST = path.resolve('dist');
const SITE = 'https://anrans.github.io';
const BASE = '/mini-sglang/';

function* walk(dir) {
	for (const name of readdirSync(dir)) {
		const abs = path.join(dir, name);
		if (statSync(abs).isDirectory()) yield* walk(abs);
		else yield abs;
	}
}

function fileFor(pathname) {
	if (!pathname.startsWith(BASE)) return null;
	const rel = decodeURIComponent(pathname.slice(BASE.length));
	const candidates = rel === '' || rel.endsWith('/')
		? [path.join(DIST, rel, 'index.html'), path.join(DIST, `${rel.slice(0, -1)}.html`)]
		: [path.join(DIST, rel), path.join(DIST, rel, 'index.html'), path.join(DIST, `${rel}.html`)];
	return candidates.find((c) => existsSync(c) && statSync(c).isFile()) ?? null;
}

const idCache = new Map();
function idsOf(file) {
	let ids = idCache.get(file);
	if (!ids) {
		ids = new Set([...readFileSync(file, 'utf8').matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
		idCache.set(file, ids);
	}
	return ids;
}

const unescape = (s) => s.replaceAll('&amp;', '&').replaceAll('&#38;', '&').replaceAll('&quot;', '"');

if (!existsSync(DIST)) {
	console.error('dist/ 不存在，先运行 astro build');
	process.exit(1);
}

const errors = [];
const pages = [...walk(DIST)].filter((f) => f.endsWith('.html'));
for (const file of pages) {
	const rel = path.relative(DIST, file);
	const pageUrl = new URL(BASE + rel.replace(/index\.html$/, '').split(path.sep).join('/'), SITE);
	const html = readFileSync(file, 'utf8');
	for (const m of html.matchAll(/\s(?:href|src)="([^"]*)"/g)) {
		const raw = unescape(m[1]);
		if (raw === '' || /^(mailto|javascript|data|tel):/i.test(raw)) continue;
		const url = new URL(raw, pageUrl);
		if (url.origin !== SITE) continue;
		const target = fileFor(url.pathname);
		if (!target) {
			errors.push(`${rel} → ${raw}`);
			continue;
		}
		const id = decodeURIComponent(url.hash.slice(1));
		if (id && target.endsWith('.html') && !idsOf(target).has(id)) {
			errors.push(`${rel} → ${raw}（目标页面没有 #${id}）`);
		}
	}
}

if (errors.length) {
	console.error(`发现 ${errors.length} 个失效链接：\n` + errors.map((e) => `  - ${e}`).join('\n'));
	process.exit(1);
}
console.log(`链接检查通过：${pages.length} 个页面`);
