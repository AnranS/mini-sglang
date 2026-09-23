// @ts-check
import react from '@astrojs/react';
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

const REPO_URL = 'https://github.com/AnranS/mini-sglang';

export default defineConfig({
	site: 'https://anrans.github.io',
	base: '/mini-sglang',
	integrations: [
		starlight({
			title: 'Mini-SGLang 源码学习',
			description: '按一个请求的生命周期读懂 mini-sglang：11 个阶段，每章都有精读清单、讲解、自测题和动手实验。',
			defaultLocale: 'root',
			locales: { root: { label: '简体中文', lang: 'zh-CN' } },
			logo: { src: './src/assets/logo.svg', alt: 'Mini-SGLang' },
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: REPO_URL }],
			editLink: { baseUrl: `${REPO_URL}/edit/main/docs/learn/` },
			lastUpdated: true,
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{ label: '学习地图', link: '/' },
				{ label: '学习路线', items: [{ autogenerate: { directory: 'stages' } }] },
				{ label: '附录', items: [{ autogenerate: { directory: 'appendix' } }] },
			],
		}),
		react(),
	],
	vite: {
		build: {
			// Mermaid's largest chunk is ~650 kB; it is only fetched on pages that contain a diagram.
			chunkSizeWarningLimit: 800,
			rolldownOptions: {
				onwarn(warning, warn) {
					// Astro's MDX pipeline emits this for every page; it is harmless.
					if (warning.code === 'MODULE_LEVEL_DIRECTIVE') return;
					warn(warning);
				},
			},
		},
	},
});
