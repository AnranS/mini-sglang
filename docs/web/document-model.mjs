/** Pure document helpers, shared by the reader and its content checks. */
export function parseGuide(markdown) {
  const matches = [...markdown.matchAll(/^<a id="([^"]+)"><\/a>\s*\n## ([^\n]+)\n/gm)];
  if (!matches.length) throw new Error('文档中没有找到章节，请检查 Markdown 章节锚点。');
  const intro = markdown.slice(0, matches[0].index).replace(/\n## 目录\s*\n[\s\S]*$/, '');
  const chapters = matches.map((match, index) => ({
    id: match[1],
    number: index + 1,
    title: match[2].replace(/^\d+\.\s*/, ''),
    markdown: markdown.slice(match.index, matches[index + 1]?.index ?? markdown.length),
  }));
  const baseline = markdown.match(/短提交号 `([^`]+)`/)?.[1] ?? '';
  return { chapters: [{ id: 'intro', number: 0, title: '阅读说明', markdown: intro }, ...chapters], baseline };
}

export function groupFor(number) {
  if (number === 0) return '开始之前';
  if (number <= 6) return '建立全貌';
  if (number <= 9) return '调度与缓存';
  if (number <= 15) return '计算与加速';
  return '实践与进阶';
}

export function resolveChapter(hash, chapters) {
  let anchor;
  try { anchor = decodeURIComponent(hash.replace(/^#/, '')); } catch { anchor = ''; }
  const chapterId = anchor.split('--')[0];
  const chapter = chapters.find(item => item.id === chapterId) ?? chapters[1];
  return { chapter, anchor: chapter.id === chapterId ? anchor : chapter.id };
}

export function sourceLink(href, origin) {
  if (!href || href.startsWith('#') || /^[a-z][a-z\d+.-]*:/i.test(href)) return null;
  try {
    const url = new URL(href, `${origin}/docs/learning-guide.zh-CN.md`);
    if (url.origin !== origin) return null;
    return decodeURIComponent(url.pathname.replace(/^\//, ''));
  } catch {
    return null;
  }
}

export function guideImageUrl(href, origin) {
  const path = sourceLink(href, origin);
  if (!path?.startsWith('docs/web/figures/')) return null;
  return `/assets/${path.slice('docs/web/'.length).split('/').map(encodeURIComponent).join('/')}`;
}
