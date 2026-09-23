import { build } from "esbuild";
import { copyFile, mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "..", "web", "learning-map");

const cursorCanvasShim = {
  name: "cursor-canvas-shim",
  setup(pluginBuild) {
    pluginBuild.onResolve({ filter: /^cursor\/canvas$/ }, () => ({ path: join(here, "canvas-shim.tsx") }));
  },
};

await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });

await build({
  entryPoints: [join(here, "main.tsx")],
  outfile: join(out, "app.js"),
  bundle: true,
  format: "esm",
  target: "es2020",
  minify: true,
  jsx: "automatic",
  legalComments: "none",
  define: { "process.env.NODE_ENV": '"production"' },
  plugins: [cursorCanvasShim],
  logLevel: "info",
});

await copyFile(join(here, "index.html"), join(out, "index.html"));
await copyFile(join(here, "..", "web", "favicon.svg"), join(out, "favicon.svg"));
// GitHub Pages serves the gh-pages branch as-is instead of running Jekyll.
await writeFile(join(out, ".nojekyll"), "");
