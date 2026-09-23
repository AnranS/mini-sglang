// Fast pre-build check for the content: MDX syntax, source references, stage ids,
// stage titles and relative links between pages. Runs without building the site.
//
//   node scripts/check-refs.mjs                  # all pages
//   node scripts/check-refs.mjs stages/05-scheduler.mdx

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { mdxToMdast } from 'satteri';
import { DECODE_LOOP, STEPS } from '../src/data/journey.mjs';
import { MODELS } from '../src/data/models.mjs';
import { STAGE_BY_ID, STAGES } from '../src/data/stages.mjs';
import { locate, resolveExcerpt } from '../src/lib/source.mjs';

const DOCS = path.resolve('src/content/docs');
const errors = [];
const fail = (where, message) => errors.push(`${where}: ${message}`);

function* walk(dir) {
	for (const name of readdirSync(dir)) {
		const abs = path.join(dir, name);
		if (statSync(abs).isDirectory()) yield* walk(abs);
		else if (/\.mdx?$/.test(name)) yield abs;
	}
}

const slugOf = (abs) =>
	path
		.relative(DOCS, abs)
		.replace(/\.mdx?$/, '')
		.split(path.sep)
		.join('/')
		.replace(/(^|\/)index$/, '');

const allPages = new Set([...walk(DOCS)].map(slugOf));

function attrValue(attr) {
	if (attr.value === null || attr.value === undefined) return true;
	if (typeof attr.value === 'string') return attr.value;
	// An expression attribute such as files={[...]} or toEnd={true}.
	return Function(`"use strict"; return (${attr.value.value});`)();
}

function attrsOf(node) {
	const out = {};
	for (const attr of node.attributes ?? []) {
		if (attr.type === 'mdxJsxAttribute') out[attr.name] = attrValue(attr);
	}
	return out;
}

function check(where, fn) {
	try {
		fn();
	} catch (error) {
		fail(where, error.message);
	}
}

function checkLink(where, slug, url) {
	if (/^[a-z]+:/i.test(url) || url.startsWith('#') || url.startsWith('//')) return;
	if (url.startsWith('/')) {
		fail(where, `站内链接不要用绝对路径（缺少 base）：${url}；改成相对链接或 <StageLink>`);
		return;
	}
	const pagePath = `/${slug}${slug ? '/' : ''}`;
	const target = new URL(url, `https://x${pagePath}`).pathname.replace(/^\/|\/$/g, '');
	if (!url.split('#')[0].endsWith('/') && url.split('#')[0] !== '') {
		fail(where, `站内链接要以 / 结尾：${url}`);
	}
	if (!allPages.has(target)) fail(where, `链接指向不存在的页面：${url}（解析为 /${target}/）`);
}

function visit(node, fn) {
	fn(node);
	for (const child of node.children ?? []) visit(child, fn);
}

async function checkPage(abs) {
	const rel = path.relative(DOCS, abs);
	const slug = slugOf(abs);
	const source = readFileSync(abs, 'utf8');
	const fm = /^---\n([\s\S]*?)\n---/.exec(source);
	if (!fm) fail(rel, '缺少 frontmatter');

	const stage = STAGES.find((s) => s.slug === slug);
	if (slug.startsWith('stages/') && !stage) fail(rel, '这个阶段页面不在 src/data/stages.mjs 里');
	if (stage && fm) {
		const title = /^title:\s*(.+)$/m.exec(fm[1])?.[1].trim().replace(/^['"]|['"]$/g, '');
		if (title !== stage.title) fail(rel, `frontmatter 的 title「${title}」和 stages.mjs 的「${stage.title}」不一致`);
	}

	let tree;
	try {
		tree = await mdxToMdast(source);
	} catch (error) {
		fail(rel, `MDX 语法错误：${error.message}`);
		return;
	}

	visit(tree, (node) => {
		const line = node.position?.start?.line;
		const where = line ? `${rel}:${line}` : rel;
		if (node.type === 'link') checkLink(where, slug, node.url);
		if (node.type !== 'mdxJsxFlowElement' && node.type !== 'mdxJsxTextElement') return;
		let a;
		try {
			a = attrsOf(node);
		} catch (error) {
			fail(where, `<${node.name}> 的属性无法求值：${error.message}`);
			return;
		}
		switch (node.name) {
			case 'Src':
				check(where, () => locate({ path: a.path, sym: a.sym, text: a.text }));
				break;
			case 'SrcCode':
				check(where, () => resolveExcerpt(a));
				break;
			case 'FileTable':
				for (const f of a.files ?? []) check(where, () => locate({ path: f.path, sym: f.sym }));
				break;
			case 'StageHeader':
			case 'StageLink':
				if (!STAGE_BY_ID[a.id]) fail(where, `未知阶段 id：${a.id}`);
				if (node.name === 'StageHeader' && stage && a.id !== stage.id) {
					fail(where, `StageHeader 的 id=${a.id} 和页面对应的 ${stage.id} 不一致`);
				}
				break;
			case 'a':
				if (typeof a.href === 'string') checkLink(where, slug, a.href);
				break;
			default:
				break;
		}
	});
}

const only = process.argv.slice(2);
// Accept paths relative to the docs collection (stages/05-scheduler.mdx) or to the cwd.
const pages = only.length
	? only.map((p) => (existsSync(path.resolve(p)) ? path.resolve(p) : path.resolve(DOCS, p)))
	: [...walk(DOCS)];
for (const abs of pages) {
	if (!existsSync(abs)) fail(abs, '文件不存在');
	else await checkPage(abs);
}

if (only.length === 0) {
	STEPS.forEach((step, i) =>
		step.refs.forEach((r) => check(`data/journey.mjs 第 ${i + 1} 步`, () => locate(r))),
	);
	if (!STEPS[DECODE_LOOP.from] || !STEPS[DECODE_LOOP.to]) fail('data/journey.mjs', 'DECODE_LOOP 越界');
	MODELS.forEach((m) => m.refs.forEach((r) => check(`data/models.mjs「${m.title}」`, () => locate(r))));
	for (const s of STAGES) {
		if (!allPages.has(s.slug)) fail('data/stages.mjs', `${s.id} 的页面 ${s.slug}.mdx 不存在`);
		for (const p of s.prereq) if (!STAGE_BY_ID[p]) fail('data/stages.mjs', `${s.id} 的前置 ${p} 不存在`);
	}
}

if (errors.length) {
	console.error(`发现 ${errors.length} 个问题：\n` + errors.map((e) => `  - ${e}`).join('\n'));
	process.exit(1);
}
console.log(`检查通过：${pages.length} 个页面${only.length ? '' : '，以及首页数据'}`);
