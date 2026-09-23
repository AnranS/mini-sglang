import { defineEcConfig } from '@astrojs/starlight/expressive-code';
import { pluginLineNumbers } from '@expressive-code/plugin-line-numbers';

export default defineEcConfig({
	plugins: [pluginLineNumbers()],
	defaultProps: { showLineNumbers: false },
	styleOverrides: {
		codeFontSize: '0.8125rem',
		codeLineHeight: '1.6',
	},
});
