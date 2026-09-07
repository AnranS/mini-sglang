import { groupFor, guideImageUrl, parseGuide, resolveChapter, sourceLink } from './document-model.mjs';

const byId = id => document.getElementById(id);
const article = byId('article');
const sourceDialog = byId('source-dialog');
const sourceContent = byId('source-content');
const mobileMenu = byId('mobile-menu');
const diagramDialog = byId('diagram-dialog');
const copies = new WeakMap();
let chapters = [];
let currentChapter;
let headings = [];
let renderVersion = 0;
let diagramQueue = Promise.resolve();
let sourceRequest;
let scrollPending = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function announce(message) {
  byId('announcer').textContent = message;
}

function buildChapterNav() {
  const fragment = document.createDocumentFragment();
  let previousGroup;
  let list;
  for (const chapter of chapters) {
    const group = groupFor(chapter.number);
    if (group !== previousGroup) {
      const section = element('div', 'nav-group');
      section.append(element('div', 'nav-group-label', group));
      list = element('ul');
      section.append(list);
      fragment.append(section);
      previousGroup = group;
    }
    const item = element('li');
    const link = element('a', 'chapter-link');
    link.href = `#${chapter.id}`;
    link.dataset.chapter = chapter.id;
    link.append(element('span', 'nav-number', chapter.number ? String(chapter.number).padStart(2, '0') : '—'));
    link.append(element('span', '', chapter.title));
    item.append(link);
    list.append(item);
  }
  byId('mobile-chapter-nav').replaceChildren(fragment.cloneNode(true));
  byId('chapter-nav').replaceChildren(fragment);
  byId('chapter-total').textContent = `${chapters.length - 1} 章`;
}

function buildOutline(chapter) {
  headings = [...article.querySelectorAll('h2, h3, h4')];
  const fragment = document.createDocumentFragment();
  headings.forEach((heading, index) => {
    heading.id = `${chapter.id}--${index + 1}`;
    const link = element('a', 'section-link', heading.textContent);
    link.href = `#${heading.id}`;
    fragment.append(link);
  });
  if (!headings.length) fragment.append(element('p', 'outline-empty', '本页为阅读说明'));
  byId('mobile-section-nav').replaceChildren(fragment.cloneNode(true));
  byId('section-nav').replaceChildren(fragment);
  byId('mobile-outline').open = false;
}

function buildPagination(chapter) {
  const index = chapters.indexOf(chapter);
  const fragment = document.createDocumentFragment();
  for (const [nextChapter, direction] of [[chapters[index - 1], 'previous'], [chapters[index + 1], 'next']]) {
    if (!nextChapter) continue;
    const link = element('a', direction);
    link.href = `#${nextChapter.id}`;
    link.append(element('span', 'pagination-label', direction === 'next' ? '下一章 →' : '← 上一章'));
    link.append(element('span', 'pagination-title', nextChapter.title));
    fragment.append(link);
  }
  byId('chapter-pagination').replaceChildren(fragment);
}

function copyButton(content) {
  const button = element('button', 'copy-button', '复制');
  button.type = 'button';
  button.setAttribute('aria-label', '复制代码');
  copies.set(button, content);
  return button;
}

function highlight(code, language) {
  if (language && window.hljs.getLanguage(language)) {
    code.innerHTML = window.hljs.highlight(code.textContent, { language, ignoreIllegals: true }).value;
    code.classList.add('hljs');
  }
}

function sourceUrl(path) {
  const url = new URL(location.href);
  url.searchParams.set('source', path);
  return `${url.pathname}${url.search}${url.hash}`;
}

function sourceAnchor(path, title) {
  const link = element('a', '', title);
  link.href = sourceUrl(path);
  link.dataset.source = path;
  return link;
}

function openDiagram(title, svg) {
  byId('diagram-title').textContent = title;
  const picture = element('img', 'diagram-preview');
  picture.alt = title;
  picture.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  const content = byId('diagram-content');
  content.replaceChildren(picture);
  content.scrollTop = 0;
  content.scrollLeft = 0;
  diagramDialog.showModal();
}

function enhanceContent(chapter, version) {
  for (const table of article.querySelectorAll('table')) {
    const wrapper = element('div', 'table-wrap');
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', '表格，可横向滚动');
    table.replaceWith(wrapper);
    wrapper.append(table);
  }
  for (const link of article.querySelectorAll('a[href]')) {
    const href = link.getAttribute('href');
    const path = sourceLink(href, location.origin);
    if (path) {
      link.dataset.source = path;
      link.href = sourceUrl(path);
      link.title = `查看本地源码：${path}`;
    } else if (/^https?:/i.test(href)) {
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
    }
  }
  for (const picture of article.querySelectorAll('img')) {
    const url = guideImageUrl(picture.getAttribute('src'), location.origin);
    if (!url) continue;
    const title = picture.title || picture.alt;
    const figure = element('figure', 'guide-figure');
    const link = element('a', 'guide-image-link');
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.setAttribute('aria-label', `${title}，在新标签页查看原图`);
    const parent = picture.parentElement;
    if (parent.tagName === 'P' && parent.children.length === 1 && !parent.textContent.trim()) parent.replaceWith(figure);
    else picture.replaceWith(figure);
    picture.src = url;
    picture.loading = 'lazy';
    picture.decoding = 'async';
    picture.removeAttribute('title');
    picture.addEventListener('load', scheduleScrollUpdate, { once: true });
    link.append(picture);
    const caption = element('figcaption');
    const original = element('a', '', '查看原图 ↗');
    original.href = url;
    original.target = '_blank';
    original.rel = 'noopener noreferrer';
    caption.append(element('span', '', title), original);
    figure.append(link, caption);
  }
  let diagramIndex = 0;
  for (const code of article.querySelectorAll('pre > code')) {
    const language = [...code.classList].find(name => name.startsWith('language-'))?.slice(9) || 'text';
    const content = code.textContent;
    const pre = code.parentElement;
    if (language === 'mermaid') {
      const figure = element('figure', 'diagram');
      const canvas = element('div', 'diagram-canvas');
      canvas.tabIndex = 0;
      canvas.setAttribute('role', 'region');
      canvas.setAttribute('aria-label', '图表，可横向或纵向滚动');
      canvas.append(element('p', 'load-state', '正在绘制图表…'));
      const labels = {
        '04-architecture': '共享分词模式：进程与设备关系',
        '05-lifecycle': '一次生成请求的完整流转',
        '10-model': '模型前向计算路径',
      };
      const caption = labels[chapter.id] || '流程图';
      figure.setAttribute('aria-label', caption);
      const captionBar = element('figcaption');
      captionBar.append(element('span', '', caption));
      figure.append(canvas, captionBar);
      const details = element('details');
      details.append(element('summary', '', '查看图表源码'));
      pre.replaceWith(figure);
      details.append(pre);
      figure.append(details);
      const diagramId = `diagram-${version}-${++diagramIndex}`;
      diagramQueue = diagramQueue.then(async () => {
        if (version !== renderVersion) return;
        try {
          const { svg } = await window.mermaid.render(diagramId, content);
          if (version !== renderVersion) return;
          canvas.innerHTML = window.DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true } });
          const rendered = canvas.querySelector('svg');
          if (!rendered || [...rendered.querySelectorAll('.node')].some(node => !node.textContent.trim())) {
            throw new Error('图表节点文字未完整保留');
          }
          rendered.setAttribute('aria-label', caption);
          // Preserve label size. Wide diagrams scroll instead of shrinking to illegible text.
          const naturalWidth = rendered.viewBox.baseVal.width;
          if (naturalWidth) rendered.style.width = `${Math.max(Math.min(naturalWidth, canvas.clientWidth - 48), naturalWidth * 0.9)}px`;
          const expand = element('button', 'diagram-expand', '放大查看');
          expand.type = 'button';
          expand.setAttribute('aria-label', `放大查看：${caption}`);
          const original = rendered.cloneNode(true);
          original.style.removeProperty('width');
          expand.addEventListener('click', () => openDiagram(caption, original.outerHTML));
          captionBar.append(expand);
        } catch {
          if (version !== renderVersion) return;
          canvas.replaceChildren(element('p', 'diagram-error', '图表未能绘制。可展开下方源码继续阅读，刷新页面后重试。'));
          details.open = true;
        }
        // Keep a linked subsection in view after the diagram gains its final height.
        if (location.hash.includes('--') && version === renderVersion) scrollToAnchor(resolveChapter(location.hash, chapters).anchor);
        scheduleScrollUpdate();
      });
      continue;
    }
    const frame = element('div', 'code-frame');
    const toolbar = element('div', 'code-toolbar');
    toolbar.append(element('span', 'code-language', language), copyButton(content));
    pre.replaceWith(frame);
    frame.append(toolbar, pre);
    pre.tabIndex = 0;
    pre.setAttribute('aria-label', `${language} 代码，可横向滚动`);
    highlight(code, language);
  }
}

function scrollToAnchor(anchor) {
  const target = document.getElementById(anchor);
  if (target && anchor.includes('--')) target.scrollIntoView({ block: 'start' });
  else window.scrollTo({ top: 0, behavior: 'auto' });
  scheduleScrollUpdate();
}

function navigate() {
  const { chapter, anchor } = resolveChapter(location.hash, chapters);
  if (currentChapter?.id === chapter.id) {
    scrollToAnchor(anchor);
    return;
  }
  currentChapter = chapter;
  const version = ++renderVersion;
  article.setAttribute('aria-busy', 'true');
  article.innerHTML = window.DOMPurify.sanitize(window.marked.parse(chapter.markdown));
  for (const oldAnchor of article.querySelectorAll('a[id]:not([href])')) oldAnchor.remove();
  // The Markdown uses h2 for chapters and h3 for sections; each page has one h1.
  if (chapter.number) {
    for (const heading of [...article.querySelectorAll('h2, h3, h4, h5, h6')]) {
      const replacement = element(`h${Number(heading.tagName.slice(1)) - 1}`);
      replacement.append(...heading.childNodes);
      heading.replaceWith(replacement);
    }
    article.querySelector('h1').textContent = chapter.title;
  }
  article.querySelector('h1').id = chapter.id;
  byId('chapter-group').textContent = groupFor(chapter.number);
  byId('chapter-number').textContent = chapter.number ? `${String(chapter.number).padStart(2, '0')} / ${chapters.length - 1}` : '阅读说明';
  document.title = `${chapter.title} · Mini-SGLang 学习指南`;
  for (const link of document.querySelectorAll('[data-chapter]')) {
    if (link.dataset.chapter === chapter.id) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  buildOutline(chapter);
  buildPagination(chapter);
  enhanceContent(chapter, version);
  article.setAttribute('aria-busy', 'false');
  byId('reading-content').focus({ preventScroll: true });
  scrollToAnchor(anchor);
  announce(`已打开${chapter.number ? `第 ${chapter.number} 章：` : ''}${chapter.title}`);
}

function updateScroll() {
  scrollPending = false;
  const available = document.documentElement.scrollHeight - window.innerHeight;
  const progress = available > 0 ? Math.min(1, Math.max(0, window.scrollY / available)) : 1;
  byId('reading-progress').style.width = `${progress * 100}%`;
  let activeId = headings[0]?.id;
  for (const heading of headings) {
    if (heading.getBoundingClientRect().top <= 145) activeId = heading.id;
  }
  for (const link of document.querySelectorAll('.section-link')) {
    if (link.hash === `#${activeId}`) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
}

function scheduleScrollUpdate() {
  if (scrollPending) return;
  scrollPending = true;
  requestAnimationFrame(updateScroll);
}

async function copyCode(button) {
  const content = copies.get(button);
  if (content === undefined) return;
  try {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(content);
    else {
      const input = element('textarea', 'clipboard-fallback', content);
      document.body.append(input);
      input.select();
      const copied = document.execCommand('copy');
      input.remove();
      button.focus({ preventScroll: true });
      if (!copied) throw new Error('copy unavailable');
    }
    button.textContent = '已复制';
    announce('代码已复制');
  } catch {
    button.textContent = '请手动复制';
    announce('浏览器未允许复制，请选中代码后手动复制');
  }
  window.setTimeout(() => { button.textContent = '复制'; }, 1800);
}

function sourceParent(path) {
  const parent = path.split('/').slice(0, -1).join('/');
  return ['python/minisgl', 'tests', 'benchmark', 'docs'].some(root => parent === root || parent.startsWith(`${root}/`)) ? parent : null;
}

async function openSource(path) {
  sourceRequest?.abort();
  const request = new AbortController();
  sourceRequest = request;
  byId('source-title').textContent = path;
  sourceContent.replaceChildren(element('p', 'load-state', '正在读取本地源码…'));
  sourceContent.scrollTop = 0;
  if (!sourceDialog.open) sourceDialog.showModal();
  history.replaceState(null, '', sourceUrl(path));
  try {
    const response = await fetch(`/api/source?path=${encodeURIComponent(path)}`, { signal: request.signal });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '源码读取失败');
    if (request.signal.aborted) return;
    const toolbar = element('div', 'code-toolbar source-toolbar');
    const parent = sourceParent(payload.path);
    if (parent) toolbar.append(sourceAnchor(parent, '← 上一级目录'));
    else toolbar.append(element('span', 'code-language', payload.kind === 'directory' ? '目录' : '源码'));
    if (payload.kind === 'directory') {
      const list = element('ul', 'source-file-list');
      for (const entry of payload.entries) {
        const item = element('li');
        const link = sourceAnchor(entry.path);
        link.append(element('span', 'source-file-kind', entry.kind === 'directory' ? 'DIR' : 'FILE'), element('span', '', entry.name));
        item.append(link);
        list.append(item);
      }
      sourceContent.replaceChildren(toolbar, list);
      if (!payload.entries.length) sourceContent.append(element('p', '', '此目录没有可浏览的源码文件。'));
    } else {
      toolbar.append(copyButton(payload.content));
      const frame = element('div', 'source-code');
      const count = payload.content.replace(/\n$/, '').split('\n').length;
      const numbers = element('pre', 'source-lines', Array.from({ length: count }, (_, i) => i + 1).join('\n'));
      numbers.setAttribute('aria-hidden', 'true');
      const pre = element('pre', 'source-text');
      pre.tabIndex = 0;
      pre.setAttribute('aria-label', '源码，可横向滚动');
      const code = element('code', '', payload.content);
      const suffix = payload.path.split('.').pop();
      const language = { py: 'python', md: 'markdown', toml: 'ini', cpp: 'cpp', cu: 'cpp', cuh: 'cpp', h: 'cpp', txt: 'text' }[suffix];
      highlight(code, language);
      pre.append(code);
      frame.append(numbers, pre);
      sourceContent.replaceChildren(toolbar, frame);
    }
  } catch (error) {
    if (request.signal.aborted) return;
    const message = element('div', 'error-state');
    message.append(element('p', '', `无法打开源码：${error.message}`));
    const retry = element('button', '', '重试');
    retry.type = 'button';
    retry.addEventListener('click', () => openSource(path));
    message.append(retry);
    sourceContent.replaceChildren(message);
  }
}

function bindEvents() {
  window.addEventListener('hashchange', navigate);
  window.addEventListener('scroll', scheduleScrollUpdate, { passive: true });
  window.addEventListener('resize', scheduleScrollUpdate, { passive: true });
  byId('open-menu').addEventListener('click', () => mobileMenu.showModal());
  sourceDialog.addEventListener('close', () => {
    sourceRequest?.abort();
    const url = new URL(location.href);
    url.searchParams.delete('source');
    history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
  });
  diagramDialog.addEventListener('close', () => byId('diagram-content').replaceChildren());
  for (const dialog of [mobileMenu, sourceDialog, diagramDialog]) {
    dialog.addEventListener('click', event => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    });
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('button');
    if (button?.dataset.close) byId(button.dataset.close).close();
    else if (button && copies.has(button)) void copyCode(button);
    const link = event.target.closest('a');
    if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    if (link.dataset.source) {
      event.preventDefault();
      void openSource(link.dataset.source);
    } else if (link.getAttribute('href') === '#reading-content') {
      event.preventDefault();
      byId('reading-content').focus({ preventScroll: true });
      window.scrollTo({ top: 0 });
    } else if (link.hash && link.origin === location.origin && link.pathname === location.pathname) {
      if (mobileMenu.open) mobileMenu.close();
      byId('mobile-outline').open = false;
      if (link.hash === location.hash) {
        event.preventDefault();
        navigate();
      }
    }
  });
}

async function start() {
  try {
    if (!window.marked || !window.DOMPurify || !window.hljs || !window.mermaid) throw new Error('本地渲染资源未能载入，请确认 docs/web/vendor 文件完整。');
    const response = await fetch('/api/guide');
    if (!response.ok) throw new Error('无法读取学习文档，请确认 Markdown 文件存在。');
    const guide = parseGuide(await response.text());
    chapters = guide.chapters;
    if (guide.baseline) {
      byId('baseline').textContent = guide.baseline;
      byId('baseline').title = `源码分析基线 ${guide.baseline} · 查看阅读说明`;
    }
    window.marked.use({ gfm: true, breaks: false });
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      // Mermaid 11.17 uses the root setting; flowchart.htmlLabels is deprecated.
      // SVG text survives our SVG-only sanitizer, unlike foreignObject HTML labels.
      htmlLabels: false,
      theme: 'base',
      fontFamily: '-apple-system, BlinkMacSystemFont, PingFang SC, Microsoft YaHei, sans-serif',
      themeVariables: {
        primaryColor: '#edf6f3', primaryTextColor: '#173d46', primaryBorderColor: '#81a9aa',
        lineColor: '#577780', secondaryColor: '#eff3f8', tertiaryColor: '#ffffff',
        actorBkg: '#edf6f3', actorBorder: '#81a9aa', actorTextColor: '#173d46',
        signalColor: '#577780', signalTextColor: '#354953', noteBkgColor: '#f7f3e9',
        noteBorderColor: '#c8bb96', noteTextColor: '#514c3b', fontSize: '16px',
      },
      flowchart: { curve: 'linear', useMaxWidth: false },
      sequence: { useMaxWidth: false, wrap: true, width: 190, actorFontSize: 16, messageFontSize: 16 },
    });
    buildChapterNav();
    bindEvents();
    if (!location.hash) history.replaceState(null, '', `${location.pathname}${location.search}#${chapters[1].id}`);
    navigate();
    const source = new URLSearchParams(location.search).get('source');
    if (source) void openSource(source);
  } catch (error) {
    article.setAttribute('aria-busy', 'false');
    const state = element('div', 'error-state');
    state.append(element('p', '', error.message || '文档载入失败，请检查本地服务是否仍在运行。'));
    const retry = element('button', '', '重新载入');
    retry.type = 'button';
    retry.addEventListener('click', () => location.reload());
    state.append(retry);
    const raw = element('a', '', '下载 Markdown 原文');
    raw.href = '/api/guide';
    const rawParagraph = element('p');
    rawParagraph.append(raw);
    state.append(rawParagraph);
    article.replaceChildren(element('h1', '', '暂时无法打开学习指南'), state);
  }
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
else void start();
