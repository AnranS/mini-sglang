/** Absolute URL of a docs page, including the site base (e.g. /mini-sglang/). */
export function pageUrl(slug: string): string {
	const base = import.meta.env.BASE_URL.replace(/\/$/, '');
	return `${base}/${slug.replace(/^\/|\/$/g, '')}/`.replace(/\/{2,}/g, '/');
}
