import { useSyncExternalStore } from 'react';

export type Status = 'todo' | 'doing' | 'done';
export type Progress = Record<string, Status>;

export const STATUS_ORDER: Status[] = ['todo', 'doing', 'done'];
export const STATUS_LABEL: Record<Status, string> = {
	todo: '未开始',
	doing: '进行中',
	done: '已完成',
};

const KEY = 'minisgl-learn:progress:v1';
const EVENT = 'minisgl-learn:progress';
const EMPTY: Progress = {};

// useSyncExternalStore needs a stable snapshot between changes.
let cached: { raw: string | null; value: Progress } = { raw: null, value: EMPTY };

function readProgress(): Progress {
	let raw: string | null;
	try {
		raw = localStorage.getItem(KEY);
	} catch {
		return EMPTY;
	}
	if (raw === cached.raw) return cached.value;
	let value = EMPTY;
	try {
		value = raw ? (JSON.parse(raw) as Progress) : EMPTY;
	} catch {
		value = EMPTY;
	}
	cached = { raw, value };
	return value;
}

function subscribe(onChange: () => void) {
	const onStorage = (event: StorageEvent) => {
		if (event.key === KEY) onChange();
	};
	window.addEventListener('storage', onStorage);
	window.addEventListener(EVENT, onChange);
	return () => {
		window.removeEventListener('storage', onStorage);
		window.removeEventListener(EVENT, onChange);
	};
}

export function setStatus(id: string, status: Status) {
	const next: Progress = { ...readProgress(), [id]: status };
	if (status === 'todo') delete next[id];
	try {
		localStorage.setItem(KEY, JSON.stringify(next));
	} catch {
		return;
	}
	window.dispatchEvent(new Event(EVENT));
}

export function useProgress(): Progress {
	return useSyncExternalStore(subscribe, readProgress, () => EMPTY);
}
