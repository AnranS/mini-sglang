import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { parseGuide, resolveChapter, sourceLink } from '../../docs/web/document-model.mjs';

const markdown = readFileSync(new URL('../../docs/learning-guide.zh-CN.md', import.meta.url), 'utf8');
const guide = parseGuide(markdown);

test('all 21 chapters retain the full Markdown body in their original order', () => {
  const chapters = guide.chapters.slice(1);
  assert.equal(chapters.length, 21);
  assert.equal(new Set(chapters.map(chapter => chapter.id)).size, 21);
  assert.equal(chapters.map(chapter => chapter.markdown).join(''), markdown.slice(markdown.indexOf('<a id="01-overview">')));
  assert.ok(guide.chapters[0].markdown.includes('**验证范围**'));
  assert.ok(!guide.chapters[0].markdown.includes('## 目录'));
  assert.match(guide.baseline, /^[a-f0-9]{7}$/);
  assert.equal(chapters.flatMap(chapter => [...chapter.markdown.matchAll(/```mermaid\n/g)]).length, 3);
});

test('chapter and subsection links resolve, including invalid URL fallbacks', () => {
  for (const chapter of guide.chapters) {
    assert.equal(resolveChapter(`#${chapter.id}`, guide.chapters).chapter.id, chapter.id);
    const subsection = resolveChapter(`#${chapter.id}--2`, guide.chapters);
    assert.equal(subsection.chapter.id, chapter.id);
    assert.equal(subsection.anchor, `${chapter.id}--2`);
  }
  for (const hash of ['', '#unknown', '#%E0%A4%A']) {
    assert.equal(resolveChapter(hash, guide.chapters).chapter.id, '01-overview');
  }
});

test('relative code links resolve against the Markdown location', () => {
  const origin = 'http://127.0.0.1:8765';
  assert.equal(sourceLink('../python/minisgl/core.py', origin), 'python/minisgl/core.py');
  assert.equal(sourceLink('../tests/', origin), 'tests/');
  assert.equal(sourceLink('../README.md', origin), 'README.md');
  for (const href of ['#01-overview', 'https://example.org/', '//example.org/file', 'mailto:a@example.org', 'javascript:alert(1)', 'data:text/plain,test', 'file:///etc/passwd', '%E0%A4%A']) {
    assert.equal(sourceLink(href, origin), null, href);
  }
});

test('missing chapter anchors produce a useful content error', () => {
  assert.throws(() => parseGuide('# Missing chapter markers'), /章节/);
});
