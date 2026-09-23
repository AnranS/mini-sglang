import type { MermaidConfig } from 'mermaid';

type Mermaid = (typeof import('mermaid'))['default'];

let mermaidPromise: Promise<Mermaid> | null = null;
let renderCount = 0;
let observing = false;

const load = () => (mermaidPromise ??= import('mermaid').then((m) => m.default));

function themeConfig(): MermaidConfig {
	const css = getComputedStyle(document.documentElement);
	const v = (name: string) => css.getPropertyValue(name).trim();
	const dark = document.documentElement.dataset.theme !== 'light';
	return {
		startOnLoad: false,
		securityLevel: 'strict',
		theme: 'base',
		fontFamily: getComputedStyle(document.body).fontFamily,
		themeVariables: {
			darkMode: dark,
			fontSize: '14px',
			background: v('--sl-color-bg'),
			primaryColor: v('--sl-color-accent-low'),
			primaryTextColor: v('--sl-color-white'),
			primaryBorderColor: v('--sl-color-accent'),
			secondaryColor: v('--sl-color-gray-6'),
			tertiaryColor: v('--sl-color-gray-7'),
			lineColor: v('--sl-color-gray-3'),
			textColor: v('--sl-color-white'),
			mainBkg: v('--sl-color-accent-low'),
			nodeBorder: v('--sl-color-accent'),
			clusterBkg: v('--sl-color-gray-6'),
			clusterBorder: v('--sl-color-gray-5'),
			edgeLabelBackground: v('--sl-color-bg'),
			actorBkg: v('--sl-color-accent-low'),
			actorBorder: v('--sl-color-accent'),
			actorTextColor: v('--sl-color-white'),
			actorLineColor: v('--sl-color-gray-4'),
			signalColor: v('--sl-color-gray-2'),
			signalTextColor: v('--sl-color-white'),
			labelBoxBkgColor: v('--sl-color-gray-6'),
			labelBoxBorderColor: v('--sl-color-gray-5'),
			labelTextColor: v('--sl-color-white'),
			loopTextColor: v('--sl-color-white'),
			noteBkgColor: v('--sl-color-gray-6'),
			noteBorderColor: v('--sl-color-gray-5'),
			noteTextColor: v('--sl-color-white'),
			activationBkgColor: v('--sl-color-gray-5'),
			activationBorderColor: v('--sl-color-gray-4'),
			sequenceNumberColor: v('--sl-color-black'),
		},
		flowchart: { curve: 'basis', padding: 12 },
		sequence: {
			mirrorActors: false,
			messageAlign: 'center',
			width: 120,
			actorMargin: 24,
			noteMargin: 8,
			boxMargin: 8,
		},
	};
}

async function renderAll() {
	const nodes = [...document.querySelectorAll<HTMLElement>('pre.mermaid[data-diagram]')];
	if (nodes.length === 0) return;
	const mermaid = await load();
	mermaid.initialize(themeConfig());
	for (const node of nodes) {
		const source = node.dataset.diagram ?? '';
		try {
			const { svg } = await mermaid.render(`mermaid-${++renderCount}`, source);
			node.innerHTML = svg;
			const width = node.querySelector('svg')?.viewBox.baseVal?.width;
			if (width) node.style.setProperty('--natural-width', `${width}px`);
			node.dataset.rendered = 'true';
		} catch (error) {
			node.dataset.rendered = 'error';
			console.error('Mermaid render failed', error);
		}
	}
}

export function renderDiagrams() {
	void renderAll();
	if (observing) return;
	observing = true;
	new MutationObserver(() => void renderAll()).observe(document.documentElement, {
		attributes: true,
		attributeFilter: ['data-theme'],
	});
}
